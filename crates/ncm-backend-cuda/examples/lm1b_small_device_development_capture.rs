//! Bounded non-held-out LM1B-small development capture, not a quality gate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Lm1bSmallTeacher;
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

fn development_seed(seed: u64) -> Result<u64, Box<dyn Error>> {
    if [11, 12, 13, 14, 15].contains(&seed) {
        Ok(seed)
    } else {
        Err("only non-held-out development seeds 11-15 are supported".into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: lm1b_small_device_development_capture ARTIFACT SEED OUTPUT".into());
    }
    let seed = development_seed(args[2].parse()?)?;
    let output = Path::new(&args[3]);
    let parent = output
        .parent()
        .ok_or("output needs a parent")?
        .canonicalize()?;
    if output.exists()
        || output.is_symlink()
        || !parent.starts_with(Path::new("reports").canonicalize()?)
    {
        return Err("new output under reports required".into());
    }
    let artifact_path = Path::new(&args[1]);
    if sha256(&fs::read(artifact_path.join("manifest.json"))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("unexpected artifact manifest".into());
    }
    let artifact = Artifact::load_inference(artifact_path)?;
    let backend = CandleBackend::cuda_cudnn(0)?;
    let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut codes = Vec::with_capacity(256);
    let mut raw_ids = Vec::with_capacity(256);
    for batch in 0..32_u64 {
        // Distinct deterministic request streams; do not claim PyTorch seed equivalence.
        let request_seed = (seed << 32) | batch;
        let result = model.sample_seeded_device(&[101; 8], 8, request_seed)?;
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
    let report = serde_json::json!({"scope": "experimental Rust LM1B-small device sampling non-held-out development; not a quality gate",
        "size": "small", "seed": seed, "count": 256, "batch_size": 8,
        "rng": "SplitMix64 request seed (development_seed << 32) | zero_based_batch; CUDA FP64 inverse CDF with CPU level-zero sampling",
        "artifact_manifest_sha256": "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76",
        "codes": codes, "raw_ids": raw_ids});
    let mut bytes = serde_json::to_vec(&report)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    println!("capture sha256 {}", sha256(&bytes));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::development_seed;

    #[test]
    fn only_nonheldout_development_seeds_are_accepted() {
        for seed in 11..=15 {
            assert_eq!(development_seed(seed).unwrap(), seed);
        }
        for seed in [0, 1, 2, 3, 10, 16, 42, u64::MAX] {
            assert!(development_seed(seed).is_err());
        }
    }
}
