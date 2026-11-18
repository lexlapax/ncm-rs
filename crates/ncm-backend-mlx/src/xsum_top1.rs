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

// One SIMD group reads each row in contiguous 32-score stripes. Reduce the
// maximum and its occurrence count separately so ties still return the sentinel.
const SIMD_SOURCE: &str = r#"
    uint row = thread_position_in_grid.x / 32u;
    uint lane = thread_position_in_grid.x % 32u;
    uint base = row * 4096u;
    bool simd = threads_per_simdgroup == 32u;
    if (!simd && lane != 0u) return;
    float best = -INFINITY;
    uint index = 0u;
    uint count = 0u;
    uint bad = 0u;
    for (uint i = lane; i < 4096u; i += simd ? 32u : 1u) {
        float value = scores[base + i];
        bad |= uint(!isfinite(value));
        if (value > best) { best = value; index = i; count = 1u; }
        else if (value == best) { ++count; }
    }
    if (simd) {
        float maximum = simd_max(best);
        count = simd_sum(best == maximum ? count : 0u);
        index = simd_min(best == maximum ? index : 0xffffffffu);
        bad = simd_sum(bad);
    }
    if (lane == 0u) output[row] = bad != 0u || count != 1u ? 0xffffffffu : index;
"#;

pub(crate) fn select(logits: &Array, unique: bool) -> crate::ops::Result<Array> {
    select_impl(logits, unique, false)
}

pub(crate) fn select_unique_simd(logits: &Array) -> crate::ops::Result<Array> {
    select_impl(logits, true, true)
}

#[allow(unsafe_code)]
fn select_impl(logits: &Array, unique: bool, simd: bool) -> crate::ops::Result<Array> {
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
    let (name, source, threads, template_args) = if simd {
        (
            "ncm_wmt14_unique_max_4096_simd",
            SIMD_SOURCE,
            rows * 32,
            vec![],
        )
    } else {
        (
            "ncm_xsum_top1_4096",
            SOURCE,
            rows,
            vec![
                ("ROWS".into(), TemplateArg::Int(rows)),
                ("UNIQUE".into(), TemplateArg::Int(i32::from(unique))),
            ],
        )
    };
    let kernel = MetalKernel::new(
        name,
        &["scores"],
        &["output"],
        source,
        MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    // SAFETY: the input is made row-contiguous, dimensions bound every read,
    // and each thread (serial) or lane zero (SIMD) owns one output row.
    let mut result = unsafe {
        kernel.apply(
            &[logits],
            &[MetalKernelOutput {
                shape: vec![*batch, *length],
                dtype: Dtype::Uint32,
            }],
            (threads, 1, 1),
            (32, 1, 1),
            MetalKernelApplyOptions {
                template_args,
                ..Default::default()
            },
        )?
    };
    if result.len() != 1 {
        return Err("unexpected XSum top-1 output count".into());
    }
    Ok(result.remove(0))
}
