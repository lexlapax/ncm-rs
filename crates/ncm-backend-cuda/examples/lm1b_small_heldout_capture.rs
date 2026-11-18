//! Frozen LM1B-small held-out candidate capture; scoring is separate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Lm1bSmallTeacher;
use safetensors::{Dtype, SafeTensors};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

fn heldout_seed(seed: u64) -> Result<u64, Box<dyn Error>> {
    if [42, 0, 1, 2, 3].contains(&seed) {
        Ok(seed)
    } else {
        Err("only held-out seeds 42, 0, 1, 2 and 3 are supported".into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact_path, seed, output, options @ ..] = args.as_slice() else {
        return Err(
            "usage: lm1b_small_heldout_capture ARTIFACT SEED OUTPUT [cudnn-default-math|cudnn-graph]".into(),
        );
    };
    let convolution = match options {
        [] => None,
        [option] if option == "cudnn-default-math" => Some("cuDNN default math"),
        [option] if option == "cudnn-graph" => Some("cuDNN graph"),
        _ => return Err("invalid LM1B capture option".into()),
    };
    let seed = heldout_seed(seed.parse()?)?;
    let output = Path::new(output);
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
    let artifact_path = Path::new(artifact_path);
    if sha256(&fs::read(artifact_path.join("manifest.json"))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("unexpected artifact manifest".into());
    }
    let binary_sha256 = sha256(&fs::read(std::env::current_exe()?)?);
    let artifact = Artifact::load_inference(artifact_path)?;
    let rope_path = std::env::var_os("NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC");
    if rope_path.is_none() {
        return Err("explicit diagnostic rotary tables required".into());
    }
    let backend = match convolution {
        Some("cuDNN graph") => CandleBackend::cuda_lm1b_cudnn_graph_experimental(0)?,
        Some(_) => CandleBackend::cuda_lm1b_cudnn_default_math_experimental(0)?,
        None => CandleBackend::cuda_lm1b_qkv_reference_experimental(0)?,
    };
    let mut model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
    if let Some(path) = &rope_path {
        let data = fs::read(path)?;
        if sha256(&data) != "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc" {
            return Err("unexpected diagnostic rotary tables".into());
        }
        let tables = SafeTensors::deserialize(&data)?;
        let values = |name| -> Result<Vec<f32>, Box<dyn Error>> {
            let tensor = tables.tensor(name)?;
            if tensor.dtype() != Dtype::F32 || tensor.shape() != [128, 32] {
                return Err("unexpected rotary tensor shape or type".into());
            }
            Ok(tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect())
        };
        model = model.with_reference_rope_diagnostic(&values("cosine")?, &values("sine")?)?;
    }
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
    let mut report = serde_json::json!({"scope": "experimental Rust LM1B-small held-out candidate samples; scoring is separate",
        "size": "small", "seed": seed, "count": 256, "batch_size": 8,
        "rng": "SplitMix64 request seed (heldout_seed << 32) | zero_based_batch; CUDA FP64 inverse CDF with CPU level-zero sampling",
        "penalty": "CUDA FP64",
        "artifact_manifest_sha256": "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76", "binary_sha256": binary_sha256,
        "codes": codes, "raw_ids": raw_ids});
    if let Some(convolution) = convolution {
        report["convolution_math"] = serde_json::json!(convolution);
        let scope = if convolution == "cuDNN graph" {
            "observed-seed experimental cuDNN graph retest"
        } else {
            "observed-seed experimental cuDNN default-math retest"
        };
        report["scope"] = serde_json::json!(scope);
    }
    if rope_path.is_some() {
        report["diagnostic"] = serde_json::json!({"rotary_tables_sha256": "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc", "backend": "reference operators with 1 MiB QKV preference"});
    }
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
    use super::heldout_seed;

    #[test]
    fn only_heldout_seeds_are_accepted() {
        for seed in [42, 0, 1, 2, 3] {
            assert_eq!(heldout_seed(seed).unwrap(), seed);
        }
        for seed in [4, 10, 11, 12, 13, 14, 15, 16, u64::MAX] {
            assert!(heldout_seed(seed).is_err());
        }
    }
}
