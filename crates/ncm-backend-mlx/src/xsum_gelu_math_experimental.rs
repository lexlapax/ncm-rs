//! Independent mathematical erf-table diagnostic; not a default GELU path.
use mlx_rs::{
    Array, Dtype,
    fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
    },
};

const SOURCE: &str = r#"
uint idx = thread_position_in_grid.x;
if (idx >= uint(N)) return;
float v = x[idx];
if (isnan(v)) { out[idx] = v; return; }
float z = v * 0.7071067811865475244f;
float y = metal::abs(z);
float e;
if (y >= 8.0f) { e = 1.0f; }
else {
    uint i = min(uint(y * 1024.0f), 8191u);
    float base = float(i) * (1.0f / 1024.0f);
    float delta = y - base;
    float high = table[i * 3];
    float low = table[i * 3 + 1];
    float slope = table[i * 3 + 2];
    float correction = fma(-base, delta, 1.0f);
    float remainder = fma(slope * delta, correction, low);
    e = high + remainder;
}
e = metal::copysign(e, z);
out[idx] = (v * 0.5f) * (1.0f + e);
"#;

/// Build erf and derivative values from the positive-term power series.
/// This table is mathematical data, not a captured model output or oracle.
pub(crate) fn table_values() -> Vec<f32> {
    let scale = 2.0 / std::f64::consts::PI.sqrt();
    let mut table = Vec::with_capacity(8193 * 3);
    for i in 0..=8192 {
        let x = f64::from(i) / 1024.0;
        let mut term = x;
        let mut sum = term;
        for n in 1..=255 {
            term *= 2.0 * x * x / f64::from(2 * n + 1);
            let next = sum + term;
            if next == sum {
                break;
            }
            sum = next;
        }
        let derivative = scale * (-x * x).exp();
        let value = derivative * sum;
        let high = value as f32;
        table.push(high);
        table.push((value - f64::from(high)) as f32);
        table.push(derivative as f32);
    }
    table
}

#[allow(unsafe_code)]
pub(crate) fn apply(x: &Array, table: &Array) -> crate::ops::Result<Array> {
    if x.dtype() != Dtype::Float32 || table.dtype() != Dtype::Float32 || table.shape() != [8193, 3]
    {
        return Err("invalid experimental GELU input or table".into());
    }
    let n = x
        .shape()
        .iter()
        .try_fold(1i32, |n, &d| if d > 0 { n.checked_mul(d) } else { None })
        .ok_or("invalid experimental GELU dimensions")?;
    let kernel = MetalKernel::new(
        "ncm_xsum_math_gelu_experimental",
        &["x", "table"],
        &["out"],
        SOURCE,
        MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )?;
    // SAFETY: contiguous inputs and checked shapes bound reads; each thread writes
    // one output, and its table index is bounded to the 8,193 checked rows.
    let mut result = unsafe {
        kernel.apply(
            &[x, table],
            &[MetalKernelOutput {
                shape: x.shape().to_vec(),
                dtype: Dtype::Float32,
            }],
            (n, 1, 1),
            (256, 1, 1),
            MetalKernelApplyOptions {
                template_args: vec![("N".into(), TemplateArg::Int(n))],
                ..Default::default()
            },
        )?
    };
    if result.len() != 1 {
        return Err("unexpected experimental GELU output".into());
    }
    Ok(result.remove(0))
}

#[cfg(test)]
mod tests {
    use super::table_values;

    #[test]
    fn mathematical_table_has_finite_accurate_reference_points() {
        let values = table_values();
        assert_eq!(values.len(), 8193 * 3);
        assert!(values.iter().all(|v| v.is_finite()));
        for (i, expected) in [
            (0, 0.0),
            (512, 0.520_499_877_813_046_5),
            (1024, 0.842_700_792_949_714_9),
            (2048, 0.995_322_265_018_952_7),
        ] {
            let actual = f64::from(values[3 * i]) + f64::from(values[3 * i + 1]);
            assert!((actual - expected).abs() < 1e-13);
        }
    }
}
