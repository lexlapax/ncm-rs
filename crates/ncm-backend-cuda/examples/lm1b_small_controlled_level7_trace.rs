//! Experimental LM1B-small CUDA trace; results require a separate oracle comparison.
#[path = "../../../tests/support/codec_capture.rs"]
mod capture;
use capture::Captured;
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend};
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
        return Err("usage: lm1b_small_controlled_level7_trace ARTIFACT REFERENCE OUTPUT".into());
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
        != "aa2bd1f8db4da63cfba04bbb857997a31855c79a63d55ce51f86a46b89b1b643"
    {
        return Err("unexpected reference report".into());
    }
    let data = fs::read(reference.join("trace.safetensors"))?;
    if sha256(&data) != "173f1be79510d9fabad465652e748a43d5d71d447c20acaa581e4c1476689c64" {
        return Err("unexpected reference trace".into());
    }
    let reference_tensors = SafeTensors::deserialize(&data)?;
    let input = reference_tensors.tensor("generator.prefix_ids")?;
    let tokens = indices(input.clone())?;
    let artifact = Artifact::load_inference(artifact_path)?;
    let backend = if std::env::var_os("NCM_LM1B_QKV_DIAGNOSTIC").is_some() {
        CandleBackend::cuda_lm1b_qkv_reference_experimental(0)?
    } else {
        CandleBackend::cuda_cudnn(0)?.xsum_reference_backend()?
    };
    let mut teacher =
        Lm1bSmallTeacher::from_artifact_level_seven_diagnostic(backend.clone(), &artifact)?;
    if let Some(path) = std::env::var_os("NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC") {
        let data = fs::read(path)?;
        if sha256(&data) != "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc" {
            return Err("unexpected diagnostic rotary tables".into());
        }
        let tensors = SafeTensors::deserialize(&data)?;
        let values = |name| -> Result<Vec<f32>, Box<dyn Error>> {
            let tensor = tensors.tensor(name)?;
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
        teacher = teacher.with_reference_rope_diagnostic(&values("cosine")?, &values("sine")?)?;
    }
    let b = &backend;
    let mut captured = Captured::default();
    let result = teacher.sample_seeded_traced(&tokens, input.shape()[0], 11, &mut captured)?;
    if b.read_indices(&result.codes)?
        != b.read_indices(&captured.indices["generator.generated_codes"])?
        || result.ids != b.read_indices(&captured.indices["generator.penalized_ids"])?
    {
        return Err("returned sample differs from trace".into());
    }
    let untraced = teacher.sample_seeded(&tokens, input.shape()[0], 11)?;
    if b.read_indices(&result.codes)? != b.read_indices(&untraced.codes)?
        || result.ids != untraced.ids
    {
        return Err("untraced sample differs from trace".into());
    }
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
