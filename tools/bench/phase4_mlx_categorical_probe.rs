//! Isolated Metal sampling experiment; included only in the archived source copy.
use mlx_rs::{
    Array, Device, Dtype,
    fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
    },
    with_device,
};
use std::time::Instant;

// Each 32-thread group owns one 4096-way row. Maxima and invalid flags are
// reduced in threadgroup memory; exponentials are computed once per element.
// The leader accumulates total and prefix in the original index order.
const COOPERATIVE: &str = r#"
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint base = row * 4096u;
    threadgroup float maxima[32];
    threadgroup uint bads[32];
    threadgroup float weights[4096];
    threadgroup float row_maximum;
    threadgroup uint invalid;
    float local_maximum = -INFINITY;
    bool bad = false;
    for (uint i = lane; i < 4096u; i += 32u) {
        float v = scores[base + i];
        bad = bad || isnan(v) || (isinf(v) && v > 0.0f);
        local_maximum = max(local_maximum, v);
    }
    maxima[lane] = local_maximum;
    bads[lane] = uint(bad);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float maximum = -INFINITY;
        uint flags = 0u;
        for (uint i = 0u; i < 32u; ++i) {
            maximum = max(maximum, maxima[i]);
            flags |= bads[i];
        }
        row_maximum = maximum;
        invalid = flags | uint(maximum == -INFINITY);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (invalid != 0u) {
        if (lane == 0u) output[row] = 0xffffffffu;
        return;
    }
    for (uint i = lane; i < 4096u; i += 32u)
        weights[i] = exp(scores[base + i] - row_maximum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        for (uint i = 0; i < 4096u; ++i) total += weights[i];
        uint bits = uniforms[row];
        uint total_bits = as_type<uint>(total);
        int total_exp = int((total_bits >> 23) & 255u);
        ulong rhs = ulong((total_bits & 0x7fffffu) | 0x800000u) * ulong(bits);
        float cumulative = 0.0f;
        uint choice = 0xffffffffu;
        uint last = 0xffffffffu;
        for (uint i = 0; i < 4096u; ++i) {
            float weight = weights[i];
            if (weight > 0.0f) last = i;
            cumulative += weight;
            if (cumulative > 0.0f) {
                uint cumulative_bits = as_type<uint>(cumulative);
                int exponent = int((cumulative_bits >> 23) & 255u);
                uint significand = (cumulative_bits & 0x7fffffu) | (exponent == 0 ? 0u : 0x800000u);
                int shift = (exponent == 0 ? 1 : exponent) - total_exp + 32;
                bool above = shift < 0 ? bits == 0u : (ulong(significand) << uint(shift)) > rhs;
                if (above) { choice = i; break; }
            }
        }
        output[row] = choice == 0xffffffffu ? last : choice;
    }
"#;

#[allow(unsafe_code)]
fn cooperative(logits: &Array, uniforms: &Array) -> crate::ops::Result<Array> {
    let [batch, length, vocab] = logits.shape() else {
        return Err("rank".into());
    };
    if !(1..=128).contains(batch)
        || !(1..=128).contains(length)
        || *vocab != 4096
        || uniforms.shape() != [*batch, *length]
        || logits.dtype() != Dtype::Float32
        || uniforms.dtype() != Dtype::Uint32
    {
        return Err("shape or dtype".into());
    }
    let rows = batch.checked_mul(*length).ok_or("row overflow")?;
    let grid = rows.checked_mul(32).ok_or("grid overflow")?;
    let kernel = MetalKernel::new(
        "ncm_categorical_cooperative_probe",
        &["scores", "uniforms"],
        &["output"],
        COOPERATIVE,
        MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    // SAFETY: each full 32-thread group accesses one checked row; every lane
    // follows the same barriers, and only the leader writes its distinct row.
    let mut outputs = unsafe {
        kernel.apply(
            &[logits, uniforms],
            &[MetalKernelOutput {
                shape: vec![*batch, *length],
                dtype: Dtype::Uint32,
            }],
            (grid, 1, 1),
            (32, 1, 1),
            MetalKernelApplyOptions {
                template_args: vec![("ROWS".into(), TemplateArg::Int(rows))],
                ..Default::default()
            },
        )?
    };
    if outputs.len() != 1 {
        return Err("unexpected output count".into());
    }
    Ok(outputs.remove(0))
}

#[test]
fn cooperative_sampling_matches_and_times_against_existing_kernel()
-> Result<(), Box<dyn std::error::Error>> {
    with_device(
        Device::gpu(),
        || -> Result<(), Box<dyn std::error::Error>> {
            for (batch, length) in [(4_i32, 64_i32), (128, 128)] {
                let rows = (batch * length) as usize;
                let mut state = 0x93a5_4321_u32;
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state
                };
                let mut logits: Vec<f32> = (0..rows * 4096)
                    .map(|_| (next() % 32001) as f32 / 1000.0 - 16.0)
                    .collect();
                let mut words: Vec<u32> = (0..rows).map(|_| next()).collect();
                logits[..4096].fill(f32::NEG_INFINITY);
                logits[4096] = f32::NAN;
                logits[8192] = f32::INFINITY;
                logits[3 * 4096..4 * 4096].fill(f32::NEG_INFINITY);
                logits[3 * 4096 + 4095] = 0.0;
                words[3] = u32::MAX;
                words[4] = 0;
                words[5] = u32::MAX;
                words[6] = 0x7fff_ffff;
                words[7] = 0x8000_0000;
                let x = Array::from_slice(&logits, &[batch, length, 4096]);
                let u = Array::from_slice(&words, &[batch, length]);
                drop(logits);
                drop(words);
                let old = super::draw(&x, &u)?;
                let new = cooperative(&x, &u)?;
                old.eval()?;
                new.eval()?;
                assert_eq!(old.try_as_slice::<u32>()?, new.try_as_slice::<u32>()?);
                let mut old_times = Vec::new();
                let mut new_times = Vec::new();
                for trial in 0..20 {
                    for candidate in if trial % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let start = Instant::now();
                        let output = if candidate {
                            cooperative(&x, &u)?
                        } else {
                            super::draw(&x, &u)?
                        };
                        output.eval()?;
                        let elapsed = start.elapsed().as_secs_f64();
                        if trial >= 5 {
                            if candidate {
                                new_times.push(elapsed)
                            } else {
                                old_times.push(elapsed)
                            }
                        }
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"batch":batch,"length":length,"rows":rows,
                "exact_outputs_match":true,"warmups":5,"trials":15,
                "baseline_seconds":old_times,"candidate_seconds":new_times})
                );
            }
            Ok(())
        },
    )
}
