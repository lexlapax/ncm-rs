//! Experimental LM1B-small Metal trace; results require a separate oracle comparison.
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
        return Err("usage: lm1b_small_teacher_trace ARTIFACT REFERENCE OUTPUT".into());
    }
    let artifact_path = Path::new(&args[1]);
    let reference = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("output must not exist".into());
    }
    let manifest = sha256(&fs::read(artifact_path.join("manifest.json"))?);
    let small = manifest == "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76";
    if !small
        && manifest != "2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c"
        && manifest != "86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a"
    {
        return Err("unexpected artifact manifest".into());
    }
    let report = sha256(&fs::read(reference.join("report.json"))?);
    let data = fs::read(reference.join("trace.safetensors"))?;
    // Small keeps its original pinned reference; larger sizes print the reference they consumed
    // and the comparator checks its trace hash explicitly.
    if small
        && (report != "00f20a4a67127293d5d01d3c146e0e3cf3cc9a2aa1992570e36901599b886d0c"
            || sha256(&data) != "fd815da45bc20fde58aa44e030aeb94e032a42ba71baa1f355f09fc91fd5223b")
    {
        return Err("unexpected reference report or trace".into());
    }
    eprintln!("reference report {report}, trace {}", sha256(&data));
    let reference_tensors = SafeTensors::deserialize(&data)?;
    let input = reference_tensors.tensor("generator.prefix_ids")?;
    let supplied = reference_tensors.tensor("generator.forced_codes")?;
    let tokens = indices(input.clone())?;
    let codes = indices(supplied.clone())?;
    let artifact = Artifact::load_inference(artifact_path)?;
    let backend = MlxBackend::gpu()?;
    let teacher = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
    let b = &backend;
    let mut captured = Captured::default();
    teacher.run_traced(&tokens, &codes, input.shape()[0], &mut captured)?;
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
