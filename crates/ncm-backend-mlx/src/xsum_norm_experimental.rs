//! Explicit experimental XSum width-768 normalization; not the default backend.
// Reduction order adapted from PyTorch v2.15.0; see third-party/pytorch-BSD.txt.
use mlx_rs::{
    Array, Dtype,
    fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput},
};

const HEADER: &str = r#"
struct Stats { float mean; float sigma; float count; };
inline Stats online(float v, Stats s) {
    float d = v - s.mean; float n = s.count + 1.0f;
    float m = s.mean + d * (1.0f / n);
    return {m, s.sigma + d * (v - m), n};
}
inline Stats combine(Stats b, Stats a) {
    float d = b.mean - a.mean; float n = a.count + b.count;
    if (n <= 0.0f) return {0.0f, 0.0f, 0.0f};
    float c = 1.0f / n; float na = a.count * c; float nb = b.count * c;
    return {na * a.mean + nb * b.mean,
            a.sigma + b.sigma + d * d * a.count * nb, n};
}
"#;
const SOURCE: &str = r#"
uint tid = thread_position_in_threadgroup.x;
uint lane = tid % 32u;
uint warp = tid / 32u;
uint row = threadgroup_position_in_grid.x;
if (threads_per_simdgroup != 32u) {
    for (uint col = tid; col < 768u; col += 128u) out[row * 768u + col] = NAN;
    return;
}
threadgroup float mean_buf[4];
threadgroup float sigma_buf[4];
threadgroup float count_buf[4];
Stats s{0.0f, 0.0f, 0.0f};
for (uint v = tid; v < 192u; v += 128u) {
    for (uint j = 0; j < 4u; ++j) s = online(x[row * 768u + 4u * v + j], s);
}
for (uint offset = 16u; offset > 0u; offset >>= 1u) {
    Stats other{simd_shuffle_down(s.mean, offset), simd_shuffle_down(s.sigma, offset),
                simd_shuffle_down(s.count, offset)};
    s = combine(s, other);
}
for (uint offset = 2u; offset > 0u; offset >>= 1u) {
    if (lane == 0u && warp >= offset && warp < 2u * offset) {
        uint target = warp - offset; mean_buf[target] = s.mean;
        sigma_buf[target] = s.sigma; count_buf[target] = s.count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u && warp < offset) {
        Stats other{mean_buf[warp], sigma_buf[warp], count_buf[warp]};
        s = combine(s, other);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (tid == 0u) { mean_buf[0] = s.mean; sigma_buf[0] = s.sigma / 768.0f; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = rsqrt(sigma_buf[0] + 1.0e-6f);
for (uint v = tid; v < 192u; v += 128u) {
    for (uint j = 0; j < 4u; ++j) {
        uint col = 4u * v + j;
        out[row * 768u + col] = gamma[col] * (inv * (x[row * 768u + col] - mean_buf[0])) + beta[col];
    }
}
"#;

pub(crate) fn apply(x: &Array, gamma: &Array, beta: &Array) -> crate::ops::Result<Array> {
    let shape = x.shape();
    if shape.is_empty()
        || shape.last() != Some(&768)
        || gamma.shape() != [768]
        || beta.shape() != [768]
        || [x, gamma, beta].iter().any(|a| a.dtype() != Dtype::Float32)
    {
        return Err("invalid experimental XSum normalization input".into());
    }
    let rows = shape
        .iter()
        .try_fold(1i32, |n, &d| n.checked_mul(d))
        .ok_or("normalization size overflow")?
        / 768;
    if rows <= 0 || rows > 128 * 560 {
        return Err("invalid experimental XSum normalization row count".into());
    }
    let threads = rows.checked_mul(128).ok_or("normalization grid overflow")?;
    let kernel = MetalKernel::new(
        "ncm_xsum_welford_experimental",
        &["x", "gamma", "beta"],
        &["out"],
        SOURCE,
        MetalKernelOptions {
            header: HEADER.into(),
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    // SAFETY: validated shapes bound every read and write, row-contiguous inputs are
    // requested, and each 128-thread group initializes exactly one width-768 row.
    let mut output = unsafe {
        kernel.apply(
            &[x, gamma, beta],
            &[MetalKernelOutput {
                shape: shape.to_vec(),
                dtype: Dtype::Float32,
            }],
            (threads, 1, 1),
            (128, 1, 1),
            MetalKernelApplyOptions::default(),
        )?
    };
    if output.len() != 1 {
        return Err("unexpected experimental normalization output".into());
    }
    Ok(output.remove(0))
}
