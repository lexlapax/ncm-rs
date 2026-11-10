//! Diagnostic isolation of checked host loading and native copied-array residency.
#[path = "../../../tests/support/host_memory.rs"]
mod host_memory;
use mlx_rs::{Array, memory};
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use safetensors::SafeTensors;
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn observation() -> Result<Value> {
    Ok(
        json!({"host":host_memory::host()?,"active":memory::active_memory()?,"cache":memory::cache_memory()?,"peak":memory::peak_memory()?}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || cfg!(debug_assertions) {
        return Err(
            "usage (release): mlx_memory_probe ARTIFACT artifact|arrays|codec NEW_REPORT".into(),
        );
    }
    if Path::new(&args[3]).exists() {
        return Err("output exists".into());
    }
    if !["artifact", "arrays", "codec"].contains(&args[2].as_str()) {
        return Err("unknown mode".into());
    }
    let b = MlxBackend::gpu()?;
    memory::clear_cache()?;
    memory::reset_peak_memory()?;
    let mut rows = Vec::new();
    let before = observation()?;
    for trial in 0..36 {
        let start = Instant::now();
        let artifact = Artifact::load(&args[1])?;
        let host_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut arrays = Vec::new();
        let mut model = None;
        match args[2].as_str() {
            "arrays" => {
                for (name, records) in &artifact.manifest().tensors {
                    let tensors = SafeTensors::deserialize(artifact.file(name)?)?;
                    for record in records {
                        let array = Array::try_from(tensors.tensor(&record.name)?)?;
                        array.eval()?;
                        arrays.push(array);
                    }
                }
            }
            "codec" => {
                model = Some(Codec::from_artifact(b.clone(), &artifact)?);
                b.evaluate(
                    &model
                        .as_ref()
                        .unwrap()
                        .weights()
                        .values()
                        .cloned()
                        .collect::<Vec<_>>(),
                )?;
            }
            _ => {}
        }
        let loaded_ms = start.elapsed().as_secs_f64() * 1000.;
        let loaded = observation()?;
        drop(model);
        drop(arrays);
        drop(artifact);
        rows.push(json!({"iteration":trial,"artifact_ms":host_ms,"total_load_ms":loaded_ms,"loaded":loaded,"dropped":observation()?}));
    }
    let sources: [&str; 5] = [
        "crates/ncm-backend-mlx/examples/mlx_memory_probe.rs",
        "tests/support/host_memory.rs",
        "crates/ncm-io/src/lib.rs",
        "crates/ncm-model/src/lib.rs",
        "Cargo.lock",
    ];
    let hashes: std::collections::BTreeMap<_, _> = sources
        .into_iter()
        .map(|p| Ok((p, sha256(&fs::read(p)?))))
        .collect::<Result<_>>()?;
    let report = json!({"schema_version":1,"scope":"diagnostic, not a matched throughput benchmark; host snapshots invoke ps outside load timing","mode":args[2],"before":before,"iterations":rows,"source_sha256":hashes});
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?,
        &report,
    )?;
    Ok(())
}
