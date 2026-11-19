//! Validate a pinned LM1B (small, medium or large) artifact on CPU without constructing a model.
use ncm_io::{Artifact, sha256};
use std::{env, error::Error, fs, path::Path};

/// Profile, pinned manifest SHA-256 and generator tensor count for each exported size.
const SIZES: [(&str, &str, usize); 3] = [
    (
        "lm1b-small-inference-v1",
        "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76",
        213,
    ),
    (
        "lm1b-medium-inference-v1",
        "2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c",
        381,
    ),
    (
        "lm1b-large-inference-v1",
        "86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a",
        549,
    ),
];

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 || !matches!(args[2].as_str(), "full" | "inference") {
        return Err("usage: lm1b_small_artifact_probe ARTIFACT full|inference".into());
    }
    let digest = sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?);
    let Some(&(profile, _, generator_tensors)) = SIZES.iter().find(|(_, sha, _)| *sha == digest)
    else {
        return Err("unexpected LM1B manifest".into());
    };
    let artifact = if args[2] == "full" {
        Artifact::load(&args[1])?
    } else {
        Artifact::load_inference(&args[1])?
    };
    let manifest = artifact.manifest();
    if manifest.profile != profile
        || manifest.tensors["codec.safetensors"].len() != 100
        || manifest.tensors["generator.safetensors"].len() != generator_tensors
        || manifest.tensors["training.safetensors"].len() != 36
        || manifest.tensors["inference.safetensors"].len() != 2
        || artifact.file("training.safetensors").is_ok() != (args[2] == "full")
        || artifact.file("inference.safetensors").is_err()
    {
        return Err("unexpected LM1B inventory or training lifetime".into());
    }
    println!(
        "{}",
        serde_json::json!({"profile":manifest.profile,"mode":args[2],
        "codec_tensors":100,"generator_tensors":generator_tensors,"training_tensors":36,"inference_tensors":2,
        "scope":"checked artifact load only; no model or GPU qualification"})
    );
    Ok(())
}
