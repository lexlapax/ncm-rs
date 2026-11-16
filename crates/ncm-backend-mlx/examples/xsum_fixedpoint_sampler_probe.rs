//! Isolated fixed-point categorical accumulation experiment; production is unchanged.
use mlx_rs::{
    Array, Device, Dtype,
    fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
    },
    with_device,
};
use safetensors::{Dtype as SafeDtype, SafeTensors};

// FP32 exponentials are converted to a 48-bit fixed-point scale. The 4096-term
// sum fits in 61 bits. Split multiplication computes floor(total * bits / 2^32)
// without 128-bit arithmetic. Tiny probabilities below 2^-48 are truncated;
// this candidate is exploratory, not an exact FP64 implementation.
const SOURCE: &str = r#"
    uint row = thread_position_in_grid.x;
    if (row >= uint(ROWS)) return;
    uint base = row * 4096u;
    float maximum = -INFINITY;
    bool bad = false;
    for (uint i = 0u; i < 4096u; ++i) {
        float v = scores[base+i];
        bad = bad || isnan(v) || (isinf(v) && v > 0.0f);
        maximum = max(maximum, v);
    }
    if (bad || maximum == -INFINITY) { output[row] = 0xffffffffu; return; }
    ulong total = 0ul;
    for (uint i = 0u; i < 4096u; ++i)
        total += ulong(exp(scores[base+i]-maximum) * 281474976710656.0f);
    uint bits = uniforms[row];
    ulong threshold = (total >> 32) * ulong(bits) + (((total & 0xfffffffful) * ulong(bits)) >> 32);
    ulong cumulative = 0ul;
    uint last = 0xffffffffu;
    for (uint i = 0u; i < 4096u; ++i) {
        float weight = exp(scores[base+i]-maximum);
        if (weight > 0.0f) last = i;
        if (bits == 0u && weight > 0.0f) { output[row] = i; return; }
        cumulative += ulong(weight * 281474976710656.0f);
        if (cumulative > threshold) { output[row] = i; return; }
    }
    output[row] = last;
"#;

fn next_word(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 32) as u32
}
#[allow(unsafe_code)]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: xsum_fixedpoint_sampler_probe CUDA_LOGITS CUDA_REFERENCE".into());
    }
    let raw = std::fs::read(&args[1])?;
    let tensors = SafeTensors::deserialize(&raw)?;
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let codes: Vec<u32> = serde_json::from_value(reference["codes"].clone())?;
    if codes.len() != 32 * 200 || reference["seed"] != 42 {
        return Err("invalid CUDA reference".into());
    }
    let kernel = MetalKernel::new(
        "ncm_xsum_fixedpoint_probe",
        &["scores", "uniforms"],
        &["output"],
        SOURCE,
        MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    let mut state = 42u64;
    let mut offset = 0usize;
    let mut differences = Vec::new();
    for level in 0..9 {
        let tensor = tensors.tensor(&format!("generator.level_{level:02}.logits"))?;
        let [32, length, 4096] = tensor.shape() else {
            return Err("invalid logit shape".into());
        };
        if tensor.dtype() != SafeDtype::F32 {
            return Err("invalid dtype".into());
        }
        let values: Vec<f32> = tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let words: Vec<u32> = (0..32 * length).map(|_| next_word(&mut state)).collect();
        let rows = i32::try_from(32 * length)?;
        let scores = Array::from_slice(&values, &[32, i32::try_from(*length)?, 4096]);
        let uniforms = Array::from_slice(&words, &[32, i32::try_from(*length)?]);
        // SAFETY: source and output dimensions are checked and each thread
        // owns one bounded output row.
        let out = unsafe {
            kernel.apply(
                &[&scores, &uniforms],
                &[MetalKernelOutput {
                    shape: vec![32, i32::try_from(*length)?],
                    dtype: Dtype::Uint32,
                }],
                (rows, 1, 1),
                (32, 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![("ROWS".into(), TemplateArg::Int(rows))],
                    ..Default::default()
                },
            )?
        };
        if out.len() != 1 {
            return Err("unexpected result count".into());
        }
        out[0].eval()?;
        let actual = out[0].try_as_slice::<u32>()?;
        for row in 0..32 {
            for pos in 0..*length {
                let index = offset + pos;
                let expected = codes[row * 200 + index];
                let got = actual[row * length + pos];
                if got != expected {
                    differences.push(serde_json::json!({"level":level,"row":row,"position":pos,
                        "flat_position":index,"expected":expected,"actual":got}));
                }
            }
        }
        offset += length;
    }
    if offset != 200 {
        return Err("invalid code lengths".into());
    }
    println!(
        "{}",
        serde_json::json!({"rows":32,"codes":6400,"scale_bits":48,"differences":differences})
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    with_device(Device::gpu(), run)
}
