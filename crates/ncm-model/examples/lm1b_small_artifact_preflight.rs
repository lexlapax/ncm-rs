//! Host-only inspection of the pinned checked LM1B-small artifact.
use ncm_io::{Artifact, sha256};
use ncm_model::{Lm1bSmallCodecInspection, Lm1bSmallGeneratorInspection};
use std::{env, error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        return Err("usage: lm1b_small_artifact_preflight ARTIFACT".into());
    }
    if sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("unexpected LM1B-small manifest".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    if artifact.manifest().profile != "lm1b-small-inference-v1"
        || artifact.file("training.safetensors").is_ok()
    {
        return Err("invalid profile or training lifetime".into());
    }
    let config = artifact.file("config.json")?;
    let generator = Lm1bSmallGeneratorInspection::from_preparation_bytes(
        config,
        artifact.file("generator.safetensors")?,
        artifact.file("inference.safetensors")?,
    )?;
    let codec = Lm1bSmallCodecInspection::from_preparation_bytes(
        config,
        artifact.file("codec.safetensors")?,
    )?;
    println!(
        "{}",
        serde_json::json!({"scope":"checked artifact and host schema only; no model or GPU qualification",
        "generator_tensors":generator.generator_tensors,"codec_tensors":codec.codec_tensors,
        "inference_tensors":generator.inference_tensors})
    );
    Ok(())
}
