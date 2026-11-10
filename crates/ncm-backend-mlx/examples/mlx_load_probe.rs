//! Checked unified-memory load, exact readback and allocator observations.
use mlx_rs::{Array, Device, memory, with_device};
use ncm_io::{Artifact, sha256};
use safetensors::{SafeTensors, tensor::TensorView};
use serde_json::json;
use std::{fs, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn load(root: &str) -> Result<(Artifact, Vec<Array>)> {
    let artifact = Artifact::load(root)?;
    let mut arrays = Vec::new();
    for (name, records) in &artifact.manifest().tensors {
        let tensors = SafeTensors::deserialize(artifact.file(name)?)?;
        for record in records {
            let view = tensors.tensor(&record.name)?;
            for &dim in view.shape() {
                i32::try_from(dim)?;
            }
            let array = Array::try_from(view)?;
            array.eval()?;
            arrays.push(array);
        }
    }
    Ok((artifact, arrays))
}
fn main() -> Result<()> {
    with_device(Device::gpu(), run)
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || cfg!(debug_assertions) {
        return Err("usage (release): mlx_load_probe <bundle> <new-report.json>".into());
    }
    memory::clear_cache()?;
    memory::reset_peak_memory()?;
    let before = memory::active_memory()?;
    let start = Instant::now();
    let loaded = load(&args[0])?;
    let first = start.elapsed().as_secs_f64() * 1000.;
    let resident = memory::active_memory()?;
    // Verify every array against its source bytes without arithmetic changing signed zeros.
    let mut index = 0;
    for (name, records) in &loaded.0.manifest().tensors {
        let tensors = SafeTensors::deserialize(loaded.0.file(name)?)?;
        for record in records {
            let view = tensors.tensor(&record.name)?;
            let actual = TensorView::try_from(&loaded.1[index])?;
            if actual.shape() != view.shape()
                || actual.dtype() != view.dtype()
                || actual.data() != view.data()
            {
                return Err("MLX tensor readback differs".into());
            }
            index += 1;
        }
    }
    // Force a real Metal kernel to consume a loaded F32 weight. Arrays use unified memory,
    // so eval of a leaf alone must not be described as a CUDA-style upload.
    let x = loaded
        .1
        .iter()
        .find(|a| a.dtype() == mlx_rs::Dtype::Float32)
        .ok_or("no F32 weights")?;
    let squared = x.square()?;
    squared.eval()?;
    drop(squared);
    drop(loaded);
    let after_drop = memory::active_memory()?;
    for _ in 0..5 {
        drop(load(&args[0])?);
    }
    let mut trials = Vec::new();
    for _ in 0..30 {
        let start = Instant::now();
        let loaded = load(&args[0])?;
        trials.push(start.elapsed().as_secs_f64() * 1000.);
        drop(loaded);
    }
    let report = json!({"schema_version":1,"runtime":"mlx-rs-0.34.0/mlx-0.32.2","device":"metal","tensor_count":index,"byte_exact_readback":true,
        "first_load_ms":first,"trials_ms":trials,"warmups":5,"manifest_sha256":sha256(&fs::read(std::path::Path::new(&args[0]).join("manifest.json"))?),
        "scope":"read + integrity verification + copied unified-memory arrays; readback/kernel check excluded",
        "active_memory_bytes":{"before":before,"loaded":resident,"after_drop":after_drop,"after_trials":memory::active_memory()?},
        "peak_memory_bytes":memory::peak_memory()?,"cache_memory_bytes":memory::cache_memory()?,
        "source_sha256":{"probe":sha256(include_bytes!("mlx_load_probe.rs")),"loader":sha256(include_bytes!("../../ncm-io/src/lib.rs")),"lock":sha256(include_bytes!("../../../Cargo.lock"))},
        "cold_disk_cache":false,"model_quality":null});
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[1])?,
        &report,
    )?;
    println!("Metal unified memory: {index} tensors exact");
    Ok(())
}
