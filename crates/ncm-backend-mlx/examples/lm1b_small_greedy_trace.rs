//! Experimental LM1B-small Metal greedy trace; results require a separate oracle comparison.
#[path = "../../../tests/support/codec_capture.rs"]
mod capture;
use capture::Captured;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Lm1bSmallTeacher;
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, serialize},
};
use std::{collections::BTreeMap, error::Error, fs, path::Path};

fn indices(t: TensorView<'_>) -> Result<Vec<u32>, Box<dyn Error>> {
    if t.dtype() != Dtype::I64 {
        return Err("expected I64 inputs".into());
    }
    t.data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|x| Ok(u32::try_from(i64::from_le_bytes(*x))?))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    assert_eq!(capture::tolerance("decode.logits"), (5e-4, 2e-4));
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: lm1b_small_greedy_trace ARTIFACT REFERENCE OUTPUT".into());
    }
    let artifact_path = Path::new(&args[1]);
    let reference = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("output must not exist".into());
    }
    if sha256(&fs::read(artifact_path.join("manifest.json"))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("unexpected artifact manifest".into());
    }
    if sha256(&fs::read(reference.join("report.json"))?)
        != "874772e2a12028ab2432e5e5a0a6edff6cb8d30c43d06564e53338b68a16e8b8"
    {
        return Err("unexpected reference report".into());
    }
    let data = fs::read(reference.join("trace.safetensors"))?;
    if sha256(&data) != "03b598e97609d01f46671523709a55836c4894050bbd7055c857ea9e7d6cea78" {
        return Err("unexpected reference trace".into());
    }
    let reference_tensors = SafeTensors::deserialize(&data)?;
    let input = reference_tensors.tensor("generator.prefix_ids")?;
    let supplied = reference_tensors.tensor("generator.initial_codes")?;
    let tokens = indices(input.clone())?;
    let codes = indices(supplied.clone())?;
    let artifact = Artifact::load_inference(artifact_path)?;
    let backend = MlxBackend::gpu()?;
    let teacher = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
    let b = &backend;
    let mut captured = Captured::default();
    teacher.generate_greedy_traced(&tokens, &codes, input.shape()[0], &mut captured)?;
    let mut all = BTreeMap::new();
    for (name, tensor) in captured.tensors {
        let values = b.read_f32(&tensor)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("nonfinite {name}").into());
        }
        all.insert(
            name,
            (
                Dtype::F32,
                b.shape(&tensor),
                values
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        );
    }
    for (name, tensor) in captured.indices {
        all.insert(
            name,
            (
                Dtype::I64,
                b.indices_shape(&tensor),
                b.read_indices(&tensor)?
                    .into_iter()
                    .flat_map(|x| i64::from(x).to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        );
    }
    let views = all
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            Ok((name, TensorView::new(*dtype, shape.clone(), bytes)?))
        })
        .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
    let serialized = serialize(views, None)?;
    fs::create_dir_all(output)?;
    fs::write(output.join("traces.safetensors"), &serialized)?;
    println!(
        "{} tensors, sha256 {}; experimental capture only",
        all.len(),
        sha256(&serialized)
    );
    Ok(())
}
