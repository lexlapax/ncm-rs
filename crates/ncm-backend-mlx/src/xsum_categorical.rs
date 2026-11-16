//! XSum fixed-point inverse-CDF draws on an explicit Metal stream.
use mlx_rs::{
    Array, Dtype,
    fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
    },
};

// FP32 exponentials are quantized to 2^-48 and accumulated exactly in u64.
// The maximum 4096-term total is 2^60; split multiplication avoids u128.
// Tiny probabilities are truncated, so this is not an exact FP64 sampler.
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
        "ncm_xsum_fixedpoint_categorical_4096",
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
