//! Explicit finite 4096-way XSum selection on the caller's Metal stream.
use mlx_rs::{
    Array, Dtype,
    fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
    },
};

// Each thread owns one row. Strict comparisons keep the first index of a tie;
// the earlier profile rejects ties, and both profiles reject all nonfinite inputs.
const SOURCE: &str = r#"
    uint row = thread_position_in_grid.x;
    if (row >= uint(ROWS)) return;
    uint base = row * 4096u;
    float best = -INFINITY;
    uint index = 0u;
    bool bad = false;
    bool tied = false;
    for (uint i = 0u; i < 4096u; ++i) {
        float v = scores[base + i];
        bad = bad || !isfinite(v);
        if (v > best) { best = v; index = i; tied = false; }
        else if (v == best) { tied = true; }
    }
    output[row] = bad || (UNIQUE && tied) ? 0xffffffffu : index;
"#;

#[allow(unsafe_code)]
pub(crate) fn select(logits: &Array, unique: bool) -> crate::ops::Result<Array> {
    let [batch, length, vocab] = logits.shape() else {
        return Err("XSum top-1 logits must be rank three".into());
    };
    if !(1..=128).contains(batch)
        || !(1..=128).contains(length)
        || *vocab != 4096
        || logits.dtype() != Dtype::Float32
    {
        return Err("invalid XSum top-1 shape or dtype".into());
    }
    let rows = batch
        .checked_mul(*length)
        .ok_or("XSum top-1 size overflow")?;
    let kernel = MetalKernel::new(
        "ncm_xsum_top1_4096",
        &["scores"],
        &["output"],
        SOURCE,
        MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    // SAFETY: the input is made row-contiguous, dimensions bound every read,
    // and each thread writes exactly its own checked output row.
    let mut result = unsafe {
        kernel.apply(
            &[logits],
            &[MetalKernelOutput {
                shape: vec![*batch, *length],
                dtype: Dtype::Uint32,
            }],
            (rows, 1, 1),
            (32, 1, 1),
            MetalKernelApplyOptions {
                template_args: vec![
                    ("ROWS".into(), TemplateArg::Int(rows)),
                    ("UNIQUE".into(), TemplateArg::Int(i32::from(unique))),
                ],
                ..Default::default()
            },
        )?
    };
    if result.len() != 1 {
        return Err("unexpected XSum top-1 output count".into());
    }
    Ok(result.remove(0))
}
