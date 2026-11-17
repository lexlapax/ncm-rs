//! Validate the pinned LM1B-small artifact on CPU without constructing a model.
use ncm_io::{Artifact, sha256};
use std::{env, error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 || !matches!(args[2].as_str(), "full" | "inference") {
        return Err("usage: lm1b_small_artifact_probe ARTIFACT full|inference".into());
    }
    if sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("unexpected LM1B-small manifest".into());
    }
    let artifact = if args[2] == "full" {
        Artifact::load(&args[1])?
    } else {
        Artifact::load_inference(&args[1])?
    };
    let manifest = artifact.manifest();
    if manifest.profile != "lm1b-small-inference-v1"
        || manifest.tensors["codec.safetensors"].len() != 100
        || manifest.tensors["generator.safetensors"].len() != 213
        || manifest.tensors["training.safetensors"].len() != 36
        || manifest.tensors["inference.safetensors"].len() != 2
        || artifact.file("training.safetensors").is_ok() != (args[2] == "full")
        || artifact.file("inference.safetensors").is_err()
    {
        return Err("unexpected LM1B-small inventory or training lifetime".into());
    }
    println!(
        "{}",
        serde_json::json!({"profile":manifest.profile,"mode":args[2],
        "codec_tensors":100,"generator_tensors":213,"training_tensors":36,"inference_tensors":2,
        "scope":"checked artifact load only; no model or GPU qualification"})
    );
    Ok(())
}
