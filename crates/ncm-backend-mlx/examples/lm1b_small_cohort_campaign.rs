//! Capture an explicit LM1B-small seed list on Metal with one Rust model load.
//!
//! Uses the same model, pinned rotary tables and default device FP32 sampler as
//! `ncm sample-lm1b-small ... mlx`. Request seeds follow the CUDA campaign runner.
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Lm1bSmallTeacher, lm1b_small_reference_rope};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

const ARTIFACT_MANIFEST_SHA256: &str =
    "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76";
const ROTARY_TABLES_SHA256: &str =
    "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact_path, rope, seeds, output] = args.as_slice() else {
        return Err(
            "usage: lm1b_small_cohort_campaign ARTIFACT ROTARY_TABLES SEEDS OUTPUT_DIRECTORY"
                .into(),
        );
    };
    let seeds = Path::new(seeds);
    let seed_bytes = fs::read(seeds)?;
    let seed_values: Vec<u64> = serde_json::from_slice(&seed_bytes)?;
    let distinct: std::collections::BTreeSet<_> = seed_values.iter().copied().collect();
    if !(1..=80).contains(&seed_values.len())
        || distinct.len() != seed_values.len()
        || seed_values.iter().any(|&seed| seed > u64::from(u32::MAX))
    {
        return Err("capture requires 1..80 distinct unsigned 32-bit seeds".into());
    }
    let output_dir = Path::new(output).canonicalize()?;
    if !output_dir.is_dir() || !output_dir.starts_with(Path::new("reports").canonicalize()?) {
        return Err("output directory under reports required".into());
    }
    for seed in &seed_values {
        let output = output_dir.join(format!("rust-seed-{seed}.json"));
        if output.exists() || output.is_symlink() {
            return Err("capture paths must be new".into());
        }
    }
    let artifact_path = Path::new(artifact_path);
    if sha256(&fs::read(artifact_path.join("manifest.json"))?) != ARTIFACT_MANIFEST_SHA256 {
        return Err("unexpected artifact manifest".into());
    }
    let rope_bytes = fs::read(rope)?;
    if sha256(&rope_bytes) != ROTARY_TABLES_SHA256 {
        return Err("unexpected rotary tables".into());
    }
    let binary_sha256 = sha256(&fs::read(std::env::current_exe()?)?);
    let artifact = Artifact::load_inference(artifact_path)?;
    let backend = MlxBackend::gpu()?;
    let (cosine, sine) = lm1b_small_reference_rope(&rope_bytes)?;
    let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?
        .with_reference_rope_diagnostic(&cosine, &sine)?;
    drop(artifact);
    for seed in seed_values {
        let mut codes = Vec::with_capacity(256);
        let mut raw_ids = Vec::with_capacity(256);
        for batch in 0..32_u64 {
            // Distinct deterministic request streams; do not claim PyTorch seed equivalence.
            let request_seed = (seed << 32) | batch;
            let result = model.sample_seeded_device_fp32(&[101; 8], 8, request_seed)?;
            let values = backend.read_indices(&result.codes)?;
            if values.len() != 8 * 641
                || result.ids.len() != 8 * 128
                || values.iter().any(|&x| x >= 8192)
                || result.ids.iter().any(|&x| x >= 30528)
            {
                return Err("invalid sample dimensions or IDs".into());
            }
            codes.extend(values.as_chunks::<641>().0.iter().map(|row| row.to_vec()));
            raw_ids.extend(
                result
                    .ids
                    .as_chunks::<128>()
                    .0
                    .iter()
                    .map(|row| row.to_vec()),
            );
            eprintln!("completed batch {}/32", batch + 1);
        }
        let report = serde_json::json!({"scope": "LM1B-small cohort Rust Metal capture",
            "size": "small", "seed": seed, "count": 256, "batch_size": 8, "device": "metal",
            "rng": "SplitMix64 request seed (heldout_seed << 32) | zero_based_batch; host FP64 level zero, then 24-bit midpoint uniforms",
            "sampler": "device FP32 inverse CDF", "penalty": "host FP64", "convolution_math": "MLX Metal",
            "artifact_manifest_sha256": ARTIFACT_MANIFEST_SHA256, "binary_sha256": binary_sha256,
            "seed_manifest_sha256": sha256(&seed_bytes),
            "diagnostic": {"rotary_tables_sha256": ROTARY_TABLES_SHA256},
            "codes": codes, "raw_ids": raw_ids});
        let mut bytes = serde_json::to_vec(&report)?;
        bytes.push(b'\n');
        if fs::read(seeds)? != seed_bytes {
            return Err("seed manifest changed".into());
        }
        let output = output_dir.join(format!("rust-seed-{seed}.json"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        println!("capture sha256 {}", sha256(&bytes));
    }
    Ok(())
}
