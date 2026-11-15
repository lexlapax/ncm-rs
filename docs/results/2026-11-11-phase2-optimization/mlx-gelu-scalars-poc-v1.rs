//! Compare scalar-array construction with immutable reuse in exact GELU.
use mlx_rs::{Array, Device, error::Exception, with_device};
use serde_json::json;
use std::{error::Error, fs::OpenOptions, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn scalars() -> [Array; 3] {
    [
        Array::from_f32(0.5),
        Array::from_f32(std::f32::consts::SQRT_2),
        Array::from_f32(1.),
    ]
}
fn graph(x: &Array, constants: &[Array; 3]) -> std::result::Result<Array, Exception> {
    Ok(x.multiply(&constants[0])?
        .multiply(mlx_rs::ops::erf(x.divide(&constants[1])?)?.add(&constants[2])?)?)
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 || cfg!(debug_assertions) || Path::new(&args[1]).exists() {
        return Err("usage (release): mlx_gelu_scalars_poc NEW_REPORT".into());
    }
    let constants = scalars();
    let mut rows = Vec::new();
    for batch in [1_usize, 2, 4] {
        let values: Vec<f32> = (0..batch * 89 * 1024)
            .map(|i| ((i * 73 % 1009) as f32 - 504.) / 80.)
            .collect();
        let x = Array::from_slice(&values, &[batch as i32, 89, 1024]);
        x.eval()?;
        let a = graph(&x, &scalars())?;
        let b = graph(&x, &constants)?;
        mlx_rs::transforms::eval([&a, &b])?;
        let av = a.try_as_slice::<f32>()?;
        let bv = b.try_as_slice::<f32>()?;
        if av.len() != bv.len()
            || av
                .iter()
                .zip(bv)
                .any(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
        {
            return Err("scalar reuse output mismatch or nonfinite value".into());
        }
        drop(a);
        drop(b);
        let mut fresh = Vec::new();
        let mut reused = Vec::new();
        for i in 0..35 {
            for reuse in if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let result = if reuse {
                    graph(&x, &constants)?
                } else {
                    graph(&x, &scalars())?
                };
                result.eval()?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                if i >= 5 {
                    if reuse {
                        reused.push(elapsed);
                    } else {
                        fresh.push(elapsed);
                    }
                }
            }
        }
        rows.push(json!({"batch":batch,"shape":[batch,89,1024],"fresh_ms":fresh,"reused_ms":reused,"bitwise_equal_finite":true}));
    }
    serde_json::to_writer_pretty(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[1])?,
        &json!({"schema_version":1,"warmups":5,"trials":30,"scope":"isolated exact GELU; unchanged operation order; not acceptance","cases":rows}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    with_device(Device::gpu(), run)
}
