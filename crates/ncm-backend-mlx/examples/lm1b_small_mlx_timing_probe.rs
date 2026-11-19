//! Synchronized LM1B-small Metal sampling time and memory for one fixed batch.
//!
//! Times `sample_seeded` (the `ncm sample-lm1b-small ... mlx` default path) plus code readback;
//! IDs are already on the host. Model load is reported separately.
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Lm1bSmallTeacher, lm1b_small_reference_rope};
use std::{error::Error, fs, path::Path, time::Instant};

const ARTIFACT_MANIFEST_SHA256: &str =
    "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76";
const ROTARY_TABLES_SHA256: &str =
    "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact_path, rope, batch, warmups, trials] = args.as_slice() else {
        return Err(
            "usage: lm1b_small_mlx_timing_probe ARTIFACT ROTARY_TABLES BATCH WARMUPS TRIALS".into(),
        );
    };
    let (batch, warmups, trials): (usize, usize, usize) =
        (batch.parse()?, warmups.parse()?, trials.parse()?);
    if !(1..=32).contains(&batch) || trials == 0 {
        return Err("batch must be 1..=32 and trials positive".into());
    }
    let artifact_path = Path::new(artifact_path);
    if sha256(&fs::read(artifact_path.join("manifest.json"))?) != ARTIFACT_MANIFEST_SHA256 {
        return Err("unexpected artifact manifest".into());
    }
    let rope_bytes = fs::read(rope)?;
    if sha256(&rope_bytes) != ROTARY_TABLES_SHA256 {
        return Err("unexpected rotary tables".into());
    }
    let load_start = Instant::now();
    let artifact = Artifact::load_inference(artifact_path)?;
    let backend = MlxBackend::gpu()?;
    let (cosine, sine) = lm1b_small_reference_rope(&rope_bytes)?;
    let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?
        .with_reference_rope_diagnostic(&cosine, &sine)?;
    drop(artifact);
    let load_seconds = load_start.elapsed().as_secs_f64();
    let prefixes = vec![101; batch];
    let mut seconds = Vec::with_capacity(trials);
    let mut signatures = Vec::with_capacity(trials);
    for trial in 0..warmups + trials {
        // Fixed per-trial request seeds; separate from the quality campaign streams.
        let seed = (11_u64 << 32) | trial as u64;
        let start = Instant::now();
        let result = model.sample_seeded(&prefixes, batch, seed)?;
        let codes = backend.read_indices(&result.codes)?;
        let elapsed = start.elapsed().as_secs_f64();
        if codes.len() != batch * 641 || result.ids.len() != batch * 128 {
            return Err("unexpected LM1B-small output shape".into());
        }
        if trial >= warmups {
            seconds.push(elapsed);
            let mut bytes = Vec::with_capacity(4 * (codes.len() + result.ids.len()));
            for value in codes.iter().chain(&result.ids) {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            signatures.push(sha256(&bytes));
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "runtime": "rust", "batch": batch, "warmups": warmups, "trials": trials,
            "scope": "sample_seeded generation, decoding penalties and code readback; model load excluded",
            "sampler": "host FP64 inverse CDF (CLI default)",
            "load_seconds": load_seconds, "seconds": seconds, "signatures": signatures,
            "mlx_memory": {"active": memory::active_memory()?, "peak": memory::peak_memory()?,
                           "cache": memory::cache_memory()?},
        })
    );
    Ok(())
}
