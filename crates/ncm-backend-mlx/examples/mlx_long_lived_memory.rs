//! Bounded diagnostic for repeated inference and model lifetime on Metal.
#[path = "../../../tests/support/host_memory.rs"]
mod host_memory;
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn observe() -> Result<Value> {
    Ok(
        json!({"active":memory::active_memory()?, "cache":memory::cache_memory()?,
        "peak":memory::peak_memory()?, "host":host_memory::host()?}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || cfg!(debug_assertions) || Path::new(&args[2]).exists() {
        return Err("usage (release): mlx_long_lived_memory ARTIFACT NEW_REPORT".into());
    }
    let manifest = fs::read(Path::new(&args[1]).join("manifest.json"))?;
    if serde_json::from_slice::<Value>(&manifest)?
        != serde_json::from_str::<Value>(include_str!(
            "../../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
        ))?
    {
        return Err("unpinned artifact".into());
    }
    let backend = MlxBackend::gpu()?;
    memory::clear_cache()?;
    memory::reset_peak_memory()?;
    let before = observe()?;
    let mut cycles = Vec::new();
    let mut expected_hash: Option<String> = None;
    for cycle in 0..3 {
        let artifact = Artifact::load(&args[1])?;
        let model = Codec::from_artifact(backend.clone(), &artifact)?;
        backend.evaluate(&model.weights().values().cloned().collect::<Vec<_>>())?;
        let loaded = observe()?;
        let batch = 4;
        let mut tokens = Vec::new();
        for i in 0..batch {
            for row in 0..9 {
                if row > 0 {
                    tokens.push(10);
                }
                for col in 0..9 {
                    tokens.push(((row * 3 + row / 3 + col + i % 2) % 9 + 1) as u32);
                }
            }
        }
        let mut samples = Vec::new();
        for request in 1..=128 {
            let (encoded, decoded) = model.reconstruct(&tokens, batch)?;
            let mut outputs = encoded.snapshots.clone();
            outputs.extend(decoded.snapshots.iter().cloned());
            outputs.push(decoded.logits.clone());
            backend.evaluate(&outputs)?;
            let mut ids = backend.read_indices(&encoded.codes)?;
            ids.extend(backend.read_indices(&decoded.ids)?);
            let hash = sha256(
                &ids.into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
            match &expected_hash {
                Some(expected) if expected != &hash => return Err("output IDs changed".into()),
                None => expected_hash = Some(hash),
                _ => {}
            }
            drop(outputs);
            drop(encoded);
            drop(decoded);
            if [1, 8, 32, 64, 128].contains(&request) {
                samples.push(json!({"request":request, "memory":observe()?}));
            }
        }
        drop(model);
        drop(artifact);
        cycles
            .push(json!({"cycle":cycle+1,"loaded":loaded,"requests":samples,"dropped":observe()?}));
    }
    let sources: BTreeMap<_, _> = [
        "crates/ncm-backend-mlx/examples/mlx_long_lived_memory.rs",
        "tests/support/host_memory.rs",
        "crates/ncm-backend-mlx/src/codec.rs",
        "crates/ncm-model/src/blocks.rs",
        "crates/ncm-core/src/lib.rs",
        "Cargo.lock",
    ]
    .into_iter()
    .map(|p| Ok((p, sha256(&fs::read(p)?))))
    .collect::<Result<_>>()?;
    let report = json!({"schema_version":1,"scope":"Three loads, 128 batch-4 reconstruction requests per retained model; observed allocation snapshots, not a performance benchmark or proof of an infinite bound.",
        "manifest_sha256":sha256(&manifest),"before":before,"cycles":cycles,"output_ids_sha256":expected_hash,"source_sha256":sources});
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?,
        &report,
    )?;
    Ok(())
}
