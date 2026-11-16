//! Fused RMS normalization and split-half RoPE for width 64.
use mlx_rs::{
    Array, Dtype,
    fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput},
};

const SOURCE: &str = r#"
    uint tid = thread_position_in_grid.x;
    uint row = tid / 32u;
    uint lane = tid % 32u;
    if (row >= params[0]) return;
    uint pos = row % params[1];
    uint base = row * 64u;
    float a = x[base + lane];
    float b = x[base + lane + 32u];
    float sum = 0.0f;
    if (threads_per_simdgroup == 32u && params[2] == 0u) {
        sum = simd_sum(a * a + b * b);
    } else {
        for (uint i = 0u; i < 64u; ++i) {
            float v = x[base + i];
            sum += v * v;
        }
    }
    float inv = rsqrt(sum / 64.0f + epsilon[0]);
    float lo = (a * inv) * weight[lane];
    float hi = (b * inv) * weight[lane + 32u];
    float c = cosine[pos * 32u + lane];
    float s = sine[pos * 32u + lane];
    output[base + lane] = lo * c - hi * s;
    output[base + lane + 32u] = hi * c + lo * s;
"#;

pub(crate) fn apply(
    x: &Array,
    weight: &Array,
    cos: &Array,
    sin: &Array,
    eps: f64,
) -> crate::ops::Result<Array> {
    apply_inner(x, weight, cos, sin, eps, false)
}

#[allow(unsafe_code)]
fn apply_inner(
    x: &Array,
    weight: &Array,
    cos: &Array,
    sin: &Array,
    eps: f64,
    force_scalar: bool,
) -> crate::ops::Result<Array> {
    let [batch, heads, seq, width] = x.shape() else {
        return Err("RMS RoPE requires rank four".into());
    };
    if *batch <= 0
        || *heads <= 0
        || *seq <= 0
        || *width != 64
        || weight.shape() != [64]
        || cos.shape() != [*seq, 32]
        || sin.shape() != [*seq, 32]
        || [x, weight, cos, sin]
            .iter()
            .any(|a| a.dtype() != Dtype::Float32)
        || !eps.is_finite()
        || eps <= 0.
        || eps > f32::MAX as f64
    {
        return Err("invalid RMS RoPE input".into());
    }
    let rows = usize::try_from(*batch)?
        .checked_mul(usize::try_from(*heads)?)
        .and_then(|v| v.checked_mul(usize::try_from(*seq).ok()?))
        .ok_or("RMS RoPE size overflow")?;
    let threads = i32::try_from(rows.checked_mul(32).ok_or("RMS RoPE size overflow")?)?;
    let params = Array::from_slice(
        &[
            u32::try_from(rows)?,
            u32::try_from(*seq)?,
            u32::from(force_scalar),
        ],
        &[3],
    );
    let epsilon = Array::from_slice(&[eps as f32], &[1]);
    let kernel = MetalKernel::new(
        "ncm_qk_rms_rope",
        &["x", "weight", "cosine", "sine", "params", "epsilon"],
        &["output"],
        SOURCE,
        MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    // SAFETY: checked dimensions bound all reads and writes; one 32-thread
    // threadgroup handles a distinct width-64 row. The fast reduction is used
    // only for a 32-lane SIMD group; other widths sum the row independently.
    let mut output = unsafe {
        kernel.apply(
            &[x, weight, cos, sin, &params, &epsilon],
            &[MetalKernelOutput {
                shape: x.shape().to_vec(),
                dtype: Dtype::Float32,
            }],
            (threads, 1, 1),
            (32, 1, 1),
            MetalKernelApplyOptions::default(),
        )?
    };
    if output.len() != 1 {
        return Err("unexpected RMS RoPE output count".into());
    }
    Ok(output.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_width_fallback_matches_fast_width() {
        mlx_rs::with_device(mlx_rs::Device::gpu(), || -> crate::ops::Result<()> {
            let values: Vec<f32> = (0..16 * 3 * 64)
                .map(|i| ((i * 73 % 1009) as f32 - 504.) / 80.)
                .collect();
            let x = Array::from_slice(&values, &[16, 1, 3, 64]);
            let weight = Array::from_slice(&[1.0f32; 64], &[64]);
            let cosine = Array::from_slice(&[0.8f32; 96], &[3, 32]);
            let sine = Array::from_slice(&[0.6f32; 96], &[3, 32]);
            let fast = apply_inner(&x, &weight, &cosine, &sine, 1e-6, false)?;
            let scalar = apply_inner(&x, &weight, &cosine, &sine, 1e-6, true)?;
            fast.eval()?;
            scalar.eval()?;
            for (&a, &b) in fast
                .try_as_slice::<f32>()?
                .iter()
                .zip(scalar.try_as_slice::<f32>()?)
            {
                assert!((a - b).abs() <= 1e-5 + 1e-4 * a.abs());
            }
            Ok(())
        })
        .unwrap();
    }
}
