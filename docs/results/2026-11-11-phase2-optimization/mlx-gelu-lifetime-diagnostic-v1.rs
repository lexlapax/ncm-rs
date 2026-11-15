//! Bounded exact-GELU candidate lifecycle observation; not acceptance.
#[path = "../../../tests/support/host_memory.rs"]
mod host_memory;
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, OpenOptions},
    path::Path,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn observe() -> Result<Value> {
    Ok(
        json!({"host": host_memory::host()?, "active": memory::active_memory()?, "cache": memory::cache_memory()?, "peak": memory::peak_memory()?}),
    )
}
fn tokens(batch: usize) -> Vec<u32> {
    let mut result = Vec::new();
    for i in 0..batch {
        for row in 0..9 {
            if row > 0 {
                result.push(10);
            }
            for col in 0..9 {
                result.push(((row * 3 + row / 3 + col + i % 2) % 9 + 1) as u32);
            }
        }
    }
    result
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4
        || cfg!(debug_assertions)
        || !["eager", "compiled"].contains(&args[2].as_str())
        || Path::new(&args[3]).exists()
    {
        return Err("usage (release): diagnostic BUNDLE eager|compiled NEW_REPORT".into());
    }
    let before = observe()?;
    let mut cycles = Vec::new();
    let mut hashes = BTreeMap::<usize, String>::new();
    for cycle in 0..3 {
        let artifact = Artifact::load(&args[1])?;
        let backend = MlxBackend::gpu()?;
        backend.set_compiled_gelu_for_diagnostic(args[2] == "compiled");
        let start = Instant::now();
        let model = Codec::from_artifact(backend.clone(), &artifact)?;
        backend.evaluate(&model.weights().values().cloned().collect::<Vec<_>>())?;
        let load_ms = start.elapsed().as_secs_f64() * 1000.;
        let loaded = observe()?;
        let mut requests = Vec::new();
        for i in 0..16 {
            let batch = [1_usize, 2, 3, 4, 5][i % 5];
            let start = Instant::now();
            let (encoded, decoded) = model.reconstruct(&tokens(batch), batch)?;
            let mut outputs = encoded.snapshots.clone();
            outputs.extend(decoded.snapshots.clone());
            outputs.push(decoded.logits.clone());
            backend.evaluate(&outputs)?;
            let mut ids = backend.read_indices(&encoded.codes)?;
            ids.extend(backend.read_indices(&decoded.ids)?);
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let digest = sha256(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>());
            if let Some(previous) = hashes.insert(batch, digest.clone()) {
                if previous != digest {
                    return Err("nonrepeatable IDs".into());
                }
            }
            drop((encoded, decoded, outputs));
            requests.push(json!({"iteration":i,"batch":batch,"elapsed_ms":elapsed,"ids_sha256":digest,"after":observe()?}));
        }
        let before_drop = observe()?;
        drop(model);
        drop(backend);
        drop(artifact);
        cycles.push(json!({"cycle":cycle,"load_ms":load_ms,"loaded":loaded,"requests":requests,"before_drop":before_drop,"after_drop":observe()?}));
    }
    let manifest_sha256 = sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?);
    serde_json::to_writer_pretty(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?,
        &json!({"schema_version":1,"scope":"finite backend/model lifetime diagnostic; not acceptance","mode":args[2],"manifest_sha256":manifest_sha256,"before":before,"cycles":cycles,"ids_by_batch":hashes}),
    )?;
    Ok(())
}
