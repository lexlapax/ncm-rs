//! Isolated exact-GELU compilation diagnostic; no model or production changes.
use mlx_rs::{Array, Device, error::Exception, memory, transforms::compile::compile, with_device};
use serde_json::json;
use std::{error::Error, fs, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// All arrays, including constants, are explicit transform inputs. Operation
// order is the existing MLX backend's exact-erf GELU, not the tanh variant.
fn graph(inputs: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let (x, half, sqrt_two, one) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
    Ok(vec![x.multiply(half)?.multiply(
        mlx_rs::ops::erf(x.divide(sqrt_two)?)?.add(one)?,
    )?])
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 || cfg!(debug_assertions) || Path::new(&args[1]).exists() {
        return Err("usage (release): mlx_gelu_compile_poc NEW_REPORT".into());
    }
    let mut rows = Vec::new();
    for batch in [1_usize, 2, 4] {
        let values: Vec<f32> = (0..batch * 89 * 1024)
            .map(|i| ((i * 73 % 1009) as f32 - 504.) / 80.)
            .collect();
        let inputs = [
            Array::from_slice(&values, &[batch as i32, 89, 1024]),
            Array::from_f32(0.5),
            Array::from_f32(std::f32::consts::SQRT_2),
            Array::from_f32(1.0),
        ];
        mlx_rs::transforms::eval(inputs.iter())?;
        memory::clear_cache()?;
        memory::reset_peak_memory()?;
        let mut compiled = compile(graph, None);
        let start = Instant::now();
        let first = compiled(&inputs[..])?;
        let first_graph_ms = start.elapsed().as_secs_f64() * 1000.;
        first[0].eval()?;
        let first_total_ms = start.elapsed().as_secs_f64() * 1000.;
        let first_peak = memory::peak_memory()?;
        let first_cache = memory::cache_memory()?;
        let plain = graph(&inputs)?;
        plain[0].eval()?;
        if plain[0].size() != first[0].size() {
            return Err("GELU output element counts differ".into());
        }
        let mut max_abs = 0f32;
        for (&a, &b) in plain[0]
            .try_as_slice::<f32>()?
            .iter()
            .zip(first[0].try_as_slice::<f32>()?)
        {
            if !a.is_finite() || !b.is_finite() {
                return Err(format!("non-finite GELU at batch {batch}: {a} vs {b}").into());
            }
            let diff = (a - b).abs();
            max_abs = max_abs.max(diff);
            if diff > 1e-5 + 1e-4 * a.abs() {
                return Err(format!("GELU mismatch at batch {batch}: {a} vs {b}").into());
            }
        }
        drop(first);
        drop(plain);
        let mut plain_total = Vec::new();
        let mut compiled_total = Vec::new();
        for i in 0..35 {
            // Reverse order on alternating trials to reduce systematic drift.
            for use_compiled in if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let result = if use_compiled {
                    compiled(&inputs[..])?
                } else {
                    graph(&inputs)?
                };
                result[0].eval()?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                if i >= 5 {
                    if use_compiled {
                        compiled_total.push(elapsed)
                    } else {
                        plain_total.push(elapsed)
                    }
                }
            }
        }
        rows.push(json!({"batch": batch, "shape": [batch, 89, 1024],
            "first_graph_ms": first_graph_ms, "first_total_ms": first_total_ms,
            "first_peak_bytes": first_peak, "first_cache_bytes": first_cache,
            "final_peak_bytes": memory::peak_memory()?, "final_cache_bytes": memory::cache_memory()?,
            "max_abs_difference": max_abs, "plain_total_ms": plain_total, "compiled_total_ms": compiled_total}));
    }
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[1])?,
        &json!({"schema_version": 1, "warmups": 5, "trials": 30, "cases": rows,
            "scope": "isolated exact GELU; all arrays explicit; compiled first call included separately; not acceptance"}),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    with_device(Device::gpu(), run)
}
