//! Validate the pinned WMT14 generator on CPU without constructing a model.
use ncm_io::{Artifact, sha256};
use ncm_model::Wmt14GeneratorInspection;
use std::{env, error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        return Err("usage: wmt14_generator_preflight ARTIFACT".into());
    }
    if sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected WMT14 manifest".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    let inspection = Wmt14GeneratorInspection::from_artifact(&artifact)?;
    if inspection.tensor_count != 186 {
        return Err("unexpected WMT14 generator inventory".into());
    }
    println!(
        "{}",
        serde_json::json!({"tensors":inspection.tensor_count,
        "scope":"CPU generator preflight only; no model or GPU execution"})
    );
    Ok(())
}
