//! Isolated gather-based interpolation experiment; production remains unchanged.
use mlx_rs::{Array, Device, memory, with_device};
use ncm_backend_mlx::ops;
use ncm_io::sha256;
use serde_json::json;
use std::{error::Error, fs, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn gathered(x: &Array, length: usize) -> Result<Array> {
    let source = x.shape()[1];
    let mut low = Vec::with_capacity(length);
    let mut high = Vec::with_capacity(length);
    let mut left = Vec::with_capacity(length);
    let mut right = Vec::with_capacity(length);
    for i in 0..length {
        // Match the existing MLX path's f64 coordinate calculation and f32 casts.
        let pos = ((i as f64 + 0.5) * f64::from(source) / length as f64 - 0.5).max(0.);
        let lo = (pos.floor() as i32).min(source - 1);
        let weight = pos - f64::from(lo);
        low.push(lo);
        high.push((lo + 1).min(source - 1));
        left.push((1. - weight) as f32);
        right.push(weight as f32);
    }
    let length = i32::try_from(length)?;
    let a = x.take_axis(Array::from_slice(&low, &[length]), 1)?;
    let b = x.take_axis(Array::from_slice(&high, &[length]), 1)?;
    Ok(a.multiply(Array::from_slice(&left, &[1, length, 1]))?
        .add(b.multiply(Array::from_slice(&right, &[1, length, 1]))?)?)
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3
        || !["reference", "gather"].contains(&args[1].as_str())
        || cfg!(debug_assertions)
    {
        return Err("usage (release): mlx_linear_poc reference|gather NEW_REPORT".into());
    }
    if Path::new(&args[2]).exists() {
        return Err("report already exists".into());
    }
    let mut rows = Vec::new();
    for (batch, source, length) in [
        (1_usize, 1_usize, 89_usize),
        (1, 2, 89),
        (1, 13, 89),
        (1, 50, 89),
        (1, 89, 13),
        (4, 13, 89),
    ] {
        let data: Vec<f32> = (0..batch * source * 64)
            .map(|i| ((i * 73 % 509) as f32 - 254.) / 37.)
            .collect();
        let x = Array::from_slice(&data, &[batch as i32, source as i32, 64]);
        let old = ops::linear1d(&x, length)?;
        let new = gathered(&x, length)?;
        old.eval()?;
        new.eval()?;
        if old.shape() != new.shape() {
            return Err("gather result shape differs".into());
        }
        let mut max_abs = 0.0_f32;
        for (&a, &b) in old
            .try_as_slice::<f32>()?
            .iter()
            .zip(new.try_as_slice::<f32>()?)
        {
            let diff = (a - b).abs();
            max_abs = max_abs.max(diff);
            if diff > 1e-5 + 1e-4 * a.abs() {
                return Err(
                    format!("parity failure for {batch}/{source}/{length}: {a} vs {b}").into(),
                );
            }
        }
        drop(old);
        drop(new);
        memory::clear_cache()?;
        memory::reset_peak_memory()?;
        let mut graph_ms = Vec::new();
        let mut total_ms = Vec::new();
        for iteration in 0..35 {
            let start = Instant::now();
            let y = if args[1] == "reference" {
                ops::linear1d(&x, length)?
            } else {
                gathered(&x, length)?
            };
            let graph = start.elapsed().as_secs_f64() * 1000.;
            y.eval()?;
            let total = start.elapsed().as_secs_f64() * 1000.;
            if iteration >= 5 {
                graph_ms.push(graph);
                total_ms.push(total);
            }
        }
        rows.push(json!({"batch":batch,"source_length":source,"output_length":length,"channels":64,
            "max_abs_difference":max_abs,"graph_ms":graph_ms,"total_ms":total_ms,
            "peak_mlx_bytes":memory::peak_memory()?,"active_mlx_bytes":memory::active_memory()?,"cache_mlx_bytes":memory::cache_memory()?}));
    }
    let sources = [
        "crates/ncm-backend-mlx/examples/mlx_linear_poc.rs",
        "crates/ncm-backend-mlx/src/ops.rs",
        "Cargo.lock",
    ];
    let hashes: std::collections::BTreeMap<_, _> = sources
        .into_iter()
        .map(|p| Ok((p, sha256(&fs::read(p)?))))
        .collect::<Result<_>>()?;
    let report = json!({"schema_version":1,"mode":args[1],"warmups":5,"trials":30,
        "scope":"isolated MLX interpolation; graph construction and graph plus eval; output drop excluded",
        "source_sha256":hashes,"cases":rows});
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
