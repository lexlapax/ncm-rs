//! Small fixed reconstruction workload for external CUPTI tracing, NOT timing.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::Codec;
use std::{error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(2..=3).contains(&args.len()) || cfg!(debug_assertions) {
        return Err("usage (release): codec_profile CANONICAL_BUNDLE [block4]".into());
    }
    let block4 = args.get(2).map(|s| s.as_str());
    if block4.is_some_and(|s| s != "block4") {
        return Err("unknown profile scope".into());
    }
    let manifest = fs::read(Path::new(&args[1]).join("manifest.json"))?;
    let anchor = include_bytes!(
        "../../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
    );
    if serde_json::from_slice::<serde_json::Value>(&manifest)?
        != serde_json::from_slice::<serde_json::Value>(anchor)?
    {
        return Err("unpinned artifact".into());
    }
    let backend = CandleBackend::cuda(0)?;
    if block4.is_some() {
        candle_core::cuda_backend::cudarc::driver::profiler_stop()?;
    }
    let artifact = Artifact::load(&args[1])?;
    let model = Codec::from_artifact(backend, &artifact)?;
    let mut ids = Vec::new();
    for row in 0..9 {
        if row > 0 {
            ids.push(10);
        }
        for col in 0..9 {
            ids.push((row * 3 + row / 3 + col) % 9 + 1);
        }
    }
    if block4.is_some() {
        let encoded = model.encode(&ids.repeat(4), 4)?;
        let latent = encoded.snapshots.last().ok_or("missing latent")?.clone();
        model.backend().evaluate(std::slice::from_ref(&latent))?;
        drop(encoded);
        for _ in 0..5 {
            let output = model.decoder_block(&latent)?;
            model.backend().evaluate(&[output])?;
        }
        candle_core::cuda_backend::cudarc::driver::profiler_start()?;
        for _ in 0..30 {
            let output = model.decoder_block(&latent)?;
            model.backend().evaluate(&[output])?;
        }
        candle_core::cuda_backend::cudarc::driver::profiler_stop()?;
        println!("PROFILE_WORKLOAD: 30 resident B4 blocks after 5 warmups; no performance claim");
        return Ok(());
    }
    // Whole-process profile includes load and exactly two B1 reconstructions.
    // Synchronization/readback matches the retained-output benchmark contract.
    for _ in 0..2 {
        let (encoded, decoded) = model.reconstruct(&ids, 1)?;
        let mut floats = encoded.snapshots;
        floats.extend(decoded.snapshots);
        floats.push(decoded.logits);
        model.backend().evaluate(&floats)?;
        let codes = model.backend().read_indices(&encoded.codes)?;
        let output = model.backend().read_indices(&decoded.ids)?;
        if codes.len() != 322 || output != ids {
            return Err("unexpected solved-board reconstruction".into());
        }
    }
    println!("PROFILE_WORKLOAD: checked load + two B1 reconstructions; no performance claim");
    Ok(())
}
