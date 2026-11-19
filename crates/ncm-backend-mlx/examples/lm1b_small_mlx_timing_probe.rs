//! Synchronized LM1B-small Metal sampling time and memory for one fixed batch.
//!
//! Times `sample_seeded` (host FP64 draws) or `sample_seeded_device_fp32` plus code readback;
//! IDs are already on the host. Model load is reported separately.
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Lm1bSmallTeacher, lm1b_small_reference_rope};
use std::{error::Error, fs, path::Path, time::Instant};

/// Pinned small, medium and large artifact manifests; the model reads its size from the profile.
const ARTIFACT_MANIFESTS: [&str; 3] = [
    "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76",
    "2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c",
    "86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a",
];
const ROTARY_TABLES_SHA256: &str =
    "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact_path, rope, batch, warmups, trials, sampler] = args.as_slice() else {
        return Err(
            "usage: lm1b_small_mlx_timing_probe ARTIFACT ROTARY_TABLES BATCH WARMUPS TRIALS host|device-fp32"
                .into(),
        );
    };
    let device_fp32 = match sampler.as_str() {
        "host" => false,
        "device-fp32" => true,
        _ => return Err("sampler must be host or device-fp32".into()),
    };
    let (batch, warmups, trials): (usize, usize, usize) =
        (batch.parse()?, warmups.parse()?, trials.parse()?);
    if !(1..=32).contains(&batch) || trials == 0 {
        return Err("batch must be 1..=32 and trials positive".into());
    }
    let artifact_path = Path::new(artifact_path);
    if !ARTIFACT_MANIFESTS
        .contains(&sha256(&fs::read(artifact_path.join("manifest.json"))?).as_str())
    {
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
        let result = if device_fp32 {
            model.sample_seeded_device_fp32(&prefixes, batch, seed)?
        } else {
            model.sample_seeded(&prefixes, batch, seed)?
        };
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
            "sampler": if device_fp32 { "device FP32 inverse CDF" } else { "host FP64 inverse CDF" },
            "load_seconds": load_seconds, "seconds": seconds, "signatures": signatures,
            "mlx_memory": {"active": memory::active_memory()?, "peak": memory::peak_memory()?,
                           "cache": memory::cache_memory()?},
        })
    );
    Ok(())
}
