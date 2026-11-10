//! Fresh-process lifecycle diagnostic, separate from steady-state acceptance timing.
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use serde_json::json;
use std::{error::Error, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn emit(phase: &str, start: Instant, hash: Option<&str>) {
    println!(
        "{}",
        json!({"schema_version":1,"phase":phase,"internal_ms":start.elapsed().as_secs_f64()*1000.,"output_ids_sha256":hash})
    );
}
pub fn run<B: CodecBackend + Clone>(create: impl FnOnce() -> Result<B>) -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 || cfg!(debug_assertions) {
        return Err("usage (release): codec_lifecycle CANONICAL_BUNDLE".into());
    }
    let start = Instant::now();
    let b = create()?;
    b.evaluate(&[])?;
    emit("context", start, None);
    let start = Instant::now();
    let artifact = Artifact::load(&args[1])?;
    let anchor: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
    ))?;
    if serde_json::to_value(artifact.manifest())? != anchor {
        return Err("unpinned bundle".into());
    }
    let model = Codec::from_artifact(b.clone(), &artifact)?;
    b.evaluate(&model.weights().values().cloned().collect::<Vec<_>>())?;
    // Match checked-load API ownership: host artifact is not retained by the model.
    drop(artifact);
    emit("load", start, None);
    let mut ids = Vec::new();
    for row in 0..9 {
        if row > 0 {
            ids.push(10);
        }
        for col in 0..9 {
            ids.push((row * 3 + row / 3 + col) % 9 + 1);
        }
    }
    let start = Instant::now();
    let (encoded, decoded) = model.reconstruct(&ids, 1)?;
    let mut floats = encoded.snapshots;
    floats.extend(decoded.snapshots);
    floats.push(decoded.logits);
    b.evaluate(&floats)?;
    let mut codes = b.read_indices(&encoded.codes)?;
    let output = b.read_indices(&decoded.ids)?;
    if codes.len() != 322 || output != ids {
        return Err("incorrect solved-board output".into());
    }
    codes.extend(output);
    let hash = sha256(
        &codes
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    emit("execute", start, Some(&hash));
    let start = Instant::now();
    drop(floats);
    drop(encoded.codes);
    drop(decoded.ids);
    drop(codes);
    drop(model);
    b.evaluate(&[])?;
    emit("drop", start, None);
    Ok(())
}
