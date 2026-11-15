//! Isolated Metal interpolation diagnostic; not production inference.
use mlx_rs::{
    Array, Device, Dtype,
    fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput},
    memory, with_device,
};
use ncm_backend_mlx::ops;
use ncm_io::sha256;
use serde_json::json;
use std::{error::Error, fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[allow(unsafe_code)]
fn fused(kernel: &MetalKernel, x: &Array, length: usize) -> Result<Array> {
    let &[batch, source, channels] = x.shape() else {
        return Err("expected BLC".into());
    };
    if x.dtype() != Dtype::Float32
        || batch <= 0
        || source <= 0
        || channels <= 0
        || length == 0
        || length > 1_000_000
    {
        return Err("invalid BLC interpolation shape".into());
    }
    let mut low = Vec::with_capacity(length);
    let mut high = Vec::with_capacity(length);
    let mut left = Vec::with_capacity(length);
    let mut right = Vec::with_capacity(length);
    for i in 0..length {
        let pos = ((i as f64 + 0.5) * f64::from(source) / length as f64 - 0.5).max(0.);
        let lo = (pos.floor() as i32).min(source - 1);
        let weight = pos - f64::from(lo);
        low.push(lo);
        high.push((lo + 1).min(source - 1));
        left.push((1. - weight) as f32);
        right.push(weight as f32);
    }
    let length = i32::try_from(length)?;
    let count = batch
        .checked_mul(length)
        .and_then(|n| n.checked_mul(channels))
        .ok_or("grid overflow")?;
    let low = Array::from_slice(&low, &[length]);
    let high = Array::from_slice(&high, &[length]);
    let left = Array::from_slice(&left, &[length]);
    let right = Array::from_slice(&right, &[length]);
    // SAFETY: one thread per output element; coordinates are computed from validated
    // positive BLC dimensions, bounded host-generated indices, and contiguous inputs.
    // The four index/weight arrays each have exactly `length` elements. No input is
    // written, and each thread writes a distinct element of the complete output.
    let mut outputs = unsafe {
        kernel.apply(
            &[x, &low, &high, &left, &right],
            &[MetalKernelOutput {
                shape: vec![batch, length, channels],
                dtype: Dtype::Float32,
            }],
            (count, 1, 1),
            (256, 1, 1),
            MetalKernelApplyOptions::default(),
        )?
    };
    Ok(outputs.remove(0))
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3
        || !["gather", "fused"].contains(&args[1].as_str())
        || cfg!(debug_assertions)
        || Path::new(&args[2]).exists()
    {
        return Err("usage (release): mlx_linear_fused_poc gather|fused NEW_REPORT".into());
    }
    let kernel = MetalKernel::new(
        "ncm_linear_fused_poc_v1",
        &["x", "low", "high", "left", "right"],
        &["out"],
        r#"int i = int(thread_position_in_grid.x);
           int c = x_shape[2]; int n = low_shape[0]; int s = x_shape[1];
           int channel = i % c; int pos = (i / c) % n; int batch = i / (c * n);
           float a = x[(batch * s + low[pos]) * c + channel] * left[pos];
           float b = x[(batch * s + high[pos]) * c + channel] * right[pos];
           out[i] = a + b;"#,
        MetalKernelOptions {
            header: "#pragma clang fp contract(off)\n".into(),
            ensure_row_contiguous: true,
            atomic_outputs: false,
        },
    )?;
    let mut rows = Vec::new();
    for (batch, source, length, channels) in [
        (1, 1, 89, 256),
        (1, 2, 89, 256),
        (1, 13, 89, 256),
        (1, 50, 89, 256),
        (1, 89, 13, 256),
        (4, 13, 89, 256),
        (4, 89, 13, 256),
        (2, 37, 67, 33),
        (2, 67, 37, 33),
        (1, 5, 1, 33),
    ] {
        let data: Vec<f32> = (0..batch * source * channels)
            .map(|i| ((i * 73 % 509) as f32 - 254.) / 37.)
            .collect();
        let x = Array::from_slice(&data, &[batch as i32, source as i32, channels as i32]);
        let cold_start = Instant::now();
        let cold_output = if args[1] == "fused" {
            fused(&kernel, &x, length)?
        } else {
            ops::linear1d(&x, length)?
        };
        cold_output.eval()?;
        let first_eval_ms = cold_start.elapsed().as_secs_f64() * 1000.;
        drop(cold_output);
        let reference = ops::linear1d(&x, length)?;
        let candidate = fused(&kernel, &x, length)?;
        reference.eval()?;
        candidate.eval()?;
        let a = reference.try_as_slice::<f32>()?;
        let b = candidate.try_as_slice::<f32>()?;
        if a.len() != b.len() {
            return Err("output length mismatch".into());
        }
        let mut max_abs = 0f32;
        let mut bit_differences = 0usize;
        for (&expected, &actual) in a.iter().zip(b) {
            if !expected.is_finite() || !actual.is_finite() {
                return Err("nonfinite finite-case output".into());
            }
            max_abs = max_abs.max((expected - actual).abs());
            bit_differences += usize::from(expected.to_bits() != actual.to_bits());
            if (actual - expected).abs() > 1e-5 + 1e-4 * expected.abs() {
                return Err("finite parity mismatch".into());
            }
        }
        drop(reference);
        drop(candidate);
        memory::clear_cache()?;
        memory::reset_peak_memory()?;
        let mut times = Vec::new();
        let mut first = 0f64;
        for i in 0..36 {
            let start = Instant::now();
            let y = if args[1] == "fused" {
                fused(&kernel, &x, length)?
            } else {
                ops::linear1d(&x, length)?
            };
            y.eval()?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            if i == 0 {
                first = elapsed;
            } else if i > 5 {
                times.push(elapsed);
            }
        }
        rows.push(json!({"batch":batch,"source":source,"length":length,"channels":channels,"max_abs_difference":max_abs,"bit_differences":bit_differences,"first_eval_ms":first_eval_ms,"post_parity_first_ms":first,"trials_ms":times,"peak_active_bytes":memory::peak_memory()?,"active_bytes":memory::active_memory()?,"cache_bytes":memory::cache_memory()?}));
    }
    let source_path = "crates/ncm-backend-mlx/examples/mlx_linear_fused_poc.rs";
    let report = json!({"schema_version":1,"mode":args[1],"source_sha256":sha256(&fs::read(source_path)?),"scope":"isolated contiguous synthetic interpolation, not model parity","cases":rows});
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?,
        &report,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    with_device(Device::gpu(), run)
}
