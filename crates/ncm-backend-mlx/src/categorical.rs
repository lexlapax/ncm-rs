//! Bounded inverse-CDF categorical draws on an explicit Metal stream.
use mlx_rs::{
    Array, Dtype,
    fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
    },
};

// One independent thread owns each output row. The comparison keeps the full
// random word via an exact integer comparison of IEEE-754 FP32 boundaries.
// The sequential FP32 scan has monotone boundaries.
const SOURCE: &str = r#"
    uint row = thread_position_in_grid.x;
    if (row >= uint(ROWS)) return;
    uint base = row * 4096u;
    float maximum = -INFINITY;
    bool bad = false;
    for (uint i = 0; i < 4096u; ++i) {
        float v = scores[base + i];
        bad = bad || isnan(v) || (isinf(v) && v > 0.0f);
        maximum = max(maximum, v);
    }
    if (bad || maximum == -INFINITY) { output[row] = 0xffffffffu; return; }
    float total = 0.0f;
    for (uint i = 0; i < 4096u; ++i) total += exp(scores[base + i] - maximum);
    uint bits = uniforms[row];
    // total >= 1 because the maximum logit has weight one. Compare
    // cumulative * 2^32 against total * bits exactly using their FP32
    // significands; the largest intermediate occupies at most 56 bits.
    uint total_bits = as_type<uint>(total);
    int total_exp = int((total_bits >> 23) & 255u);
    ulong rhs = ulong((total_bits & 0x7fffffu) | 0x800000u) * ulong(bits);
    float cumulative = 0.0f;
    uint choice = 0xffffffffu;
    uint last = 0xffffffffu;
    for (uint i = 0; i < 4096u; ++i) {
        float weight = exp(scores[base + i] - maximum);
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
"#;

#[allow(unsafe_code)]
pub(crate) fn draw(logits: &Array, uniforms: &Array) -> crate::ops::Result<Array> {
    let [batch, length, vocab] = logits.shape() else {
        return Err("categorical logits must be rank three".into());
    };
    if !(1..=128).contains(batch)
        || !(1..=128).contains(length)
        || *vocab != 4096
        || uniforms.shape() != [*batch, *length]
        || logits.dtype() != Dtype::Float32
        || uniforms.dtype() != Dtype::Uint32
    {
        return Err("invalid categorical shape or dtype".into());
    }
    let rows = batch
        .checked_mul(*length)
        .ok_or("categorical size overflow")?;
    let kernel = MetalKernel::new(
        "ncm_categorical_4096",
        &["scores", "uniforms"],
        &["output"],
        SOURCE,
        MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    // SAFETY: both inputs are materialized row-contiguous; checked dimensions
    // bound every read, and each grid thread writes exactly one distinct row.
    let mut output = unsafe {
        kernel.apply(
            &[logits, uniforms],
            &[MetalKernelOutput {
                shape: vec![*batch, *length],
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
    if output.len() != 1 {
        return Err("unexpected categorical output count".into());
    }
    Ok(output.remove(0))
}
