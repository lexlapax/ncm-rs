//! Capture an explicit LM1B seed list (small, medium or large, from the artifact) with one Rust model load.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Lm1bSmallTeacher, lm1b_small_reference_rope};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

/// Pinned artifact manifests for each LM1B size.
const SIZES: [(&str, &str); 3] = [
    (
        "small",
        "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76",
    ),
    (
        "medium",
        "2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c",
    ),
    (
        "large",
        "86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a",
    ),
];

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact_path, seeds, output] = args.as_slice() else {
        return Err("usage: lm1b_small_cohort_campaign ARTIFACT SEEDS OUTPUT_DIRECTORY".into());
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
    let manifest_sha256 = sha256(&fs::read(artifact_path.join("manifest.json"))?);
    let Some(&(size, _)) = SIZES.iter().find(|(_, sha)| *sha == manifest_sha256) else {
        return Err("unexpected artifact manifest".into());
    };
    let binary_sha256 = sha256(&fs::read(std::env::current_exe()?)?);
    let artifact = Artifact::load_inference(artifact_path)?;
    let rope_path = std::env::var_os("NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC");
    if rope_path.is_none() {
        return Err("explicit diagnostic rotary tables required".into());
    }
    let backend = CandleBackend::cuda_lm1b_efficient_experimental(0)?;
    let mut model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
    if let Some(path) = &rope_path {
        let (cosine, sine) = lm1b_small_reference_rope(&fs::read(path)?)?;
        model = model.with_reference_rope_diagnostic(&cosine, &sine)?;
    }
    drop(artifact);
    for seed in seed_values {
        let mut codes = Vec::with_capacity(256);
        let mut raw_ids = Vec::with_capacity(256);
        for batch in 0..32_u64 {
            // Distinct deterministic request streams; do not claim PyTorch seed equivalence.
            let request_seed = (seed << 32) | batch;
            let result = model.sample_seeded_fully_device(&[101; 8], 8, request_seed)?;
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
        let mut report = serde_json::json!({"scope": "LM1B cohort Rust capture",
            "size": size, "seed": seed, "count": 256, "batch_size": 8,
            "rng": "SplitMix64 request seed (heldout_seed << 32) | zero_based_batch; CUDA FP64 inverse CDF with CPU level-zero sampling",
            "penalty": "CUDA FP64",
            "artifact_manifest_sha256": manifest_sha256, "binary_sha256": binary_sha256,
            "codes": codes, "raw_ids": raw_ids});
        report["convolution_math"] = serde_json::json!("cuDNN graph and efficient attention");
        report["seed_manifest_sha256"] = serde_json::json!(sha256(&seed_bytes));
        if rope_path.is_some() {
            report["diagnostic"] = serde_json::json!({"rotary_tables_sha256": "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc", "backend": "reference operators with 1 MiB QKV preference"});
        }
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
