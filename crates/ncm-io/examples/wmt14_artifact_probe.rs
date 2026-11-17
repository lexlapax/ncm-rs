//! Validate the pinned WMT14 artifact on CPU without constructing a model.
use ncm_io::{Artifact, sha256};
use std::{env, error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 || !matches!(args[2].as_str(), "full" | "inference") {
        return Err("usage: wmt14_artifact_probe ARTIFACT full|inference".into());
    }
    if sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected WMT14 manifest".into());
    }
    let artifact = if args[2] == "full" {
        Artifact::load(&args[1])?
    } else {
        Artifact::load_inference(&args[1])?
    };
    let manifest = artifact.manifest();
    if manifest.profile != "wmt14-inference-v1"
        || manifest.tensors["codec.safetensors"].len() != 63
        || manifest.tensors["generator.safetensors"].len() != 186
        || manifest.tensors["training.safetensors"].len() != 33
        || artifact.file("training.safetensors").is_ok() != (args[2] == "full")
    {
        return Err("unexpected WMT14 inventory or training lifetime".into());
    }
    println!(
        "{}",
        serde_json::json!({"profile":manifest.profile,"mode":args[2],
        "codec_tensors":63,"generator_tensors":186,"training_tensors":33,
        "scope":"checked artifact load only; no model or GPU validation"})
    );
    Ok(())
}
