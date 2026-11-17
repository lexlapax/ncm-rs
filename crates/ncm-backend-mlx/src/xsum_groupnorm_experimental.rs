//! Explicit XSum group-normalization diagnostic; not the default backend.
// Reduction order adapted from PyTorch v2.15.0; see third-party/pytorch-BSD.txt.
use mlx_rs::{
    Array, Dtype,
    fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput},
};

const HEADER: &str = r#"
struct GroupStats { float mean; float m2; float count; };
inline GroupStats group_add(GroupStats a, float x) {
    float n = a.count + 1.0f, delta = x - a.mean;
    float mean = a.mean + delta / n;
    return {mean, a.m2 + delta * (x - mean), n};
}
inline GroupStats group_combine(GroupStats a, GroupStats b) {
    if (a.count == 0.0f) return b;
    if (b.count == 0.0f) return a;
    float delta = b.mean - a.mean, n = a.count + b.count, ratio = b.count / n;
    return {a.mean + delta * ratio, a.m2 + b.m2 + delta * delta * a.count * ratio, n};
}
inline GroupStats group_reduce(GroupStats a) {
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        GroupStats b{simd_shuffle_down(a.mean, offset), simd_shuffle_down(a.m2, offset),
                     simd_shuffle_down(a.count, offset)};
        a = group_combine(a, b);
    }
    return a;
}
"#;
const SOURCE: &str = r#"
uint tid = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.x;
if (threads_per_simdgroup != 32u) {
    for (uint col = tid; col < 576u; col += 512u) out[row * 576u + col] = NAN;
    return;
}
GroupStats value{0.0f, 0.0f, 0.0f};
for (uint j = tid; j < 144u; j += 512u)
    for (uint k = 0; k < 4u; ++k) value = group_add(value, x[row * 576u + j * 4u + k]);
value = group_reduce(value);
threadgroup float means[16], m2s[16], counts[16];
if (tid % 32u == 0u) { means[tid / 32u] = value.mean; m2s[tid / 32u] = value.m2; counts[tid / 32u] = value.count; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 32u) {
    value = tid < 16u ? GroupStats{means[tid], m2s[tid], counts[tid]} : GroupStats{0.0f, 0.0f, 0.0f};
    value = group_reduce(value);
}
if (tid == 0u) { means[0] = value.mean; m2s[0] = rsqrt(value.m2 / value.count + 1.0e-5f); }
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint col = tid; col < 576u; col += 512u)
    out[row * 576u + col] = (x[row * 576u + col] - means[0]) * m2s[0];
"#;

#[allow(unsafe_code)]
pub(crate) fn apply(x: &Array) -> crate::ops::Result<Array> {
    let shape = x.shape();
    let &[batch, 12, 384] = shape else {
        return Err("invalid experimental XSum group norm shape".into());
    };
    if !(1..=128).contains(&batch) || x.dtype() != Dtype::Float32 {
        return Err("invalid experimental XSum group norm input".into());
    }
    let grouped = x.swap_axes(1, 2)?.reshape(&[batch * 8, 576])?;
    let threads = batch
        .checked_mul(8 * 512)
        .ok_or("group norm grid overflow")?;
    let kernel = MetalKernel::new(
        "ncm_xsum_groupnorm_reference_order",
        &["x"],
        &["out"],
        SOURCE,
        MetalKernelOptions {
            header: HEADER.into(),
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    // SAFETY: validated dimensions and contiguous grouping bound all 512-thread group accesses.
    let mut result = unsafe {
        kernel.apply(
            &[&grouped],
            &[MetalKernelOutput {
                shape: vec![batch * 8, 576],
                dtype: Dtype::Float32,
            }],
            (threads, 1, 1),
            (512, 1, 1),
            MetalKernelApplyOptions::default(),
        )?
    };
    if result.len() != 1 {
        return Err("unexpected group norm output".into());
    }
    Ok(result
        .remove(0)
        .reshape(&[batch, 384, 12])?
        .swap_axes(1, 2)?)
}
