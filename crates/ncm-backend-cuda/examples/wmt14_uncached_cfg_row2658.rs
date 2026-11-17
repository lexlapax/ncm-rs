//! Uncached WMT14 CUDA trace; results require a separate oracle comparison.
#[path = "../../../tests/support/codec_capture.rs"]
mod capture;
use capture::Captured;
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Wmt14Teacher;
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
        return Err("usage: wmt14_uncached_cfg_row2658 ARTIFACT REFERENCE OUTPUT".into());
    }
    let artifact_path = Path::new(&args[1]);
    let reference = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("output must not exist".into());
    }
    if sha256(&fs::read(artifact_path.join("manifest.json"))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected artifact manifest".into());
    }
    if sha256(&fs::read(reference.join("report.json"))?)
        != "5e2ed2024e83052f677090e3be192c29cac623afe1b94f2fe423b8e0f7d44fb4"
    {
        return Err("unexpected reference report".into());
    }
    let data = fs::read(reference.join("trace.safetensors"))?;
    if sha256(&data) != "07dcf16e7029f40c97b88d384ddecc0feb299ffdfe566bc372aadfb388a6c14f" {
        return Err("unexpected reference trace".into());
    }
    let reference_tensors = SafeTensors::deserialize(&data)?;
    let input = reference_tensors.tensor("generator.prefix_ids")?;
    let tokens = indices(input.clone())?;
    let artifact = Artifact::load_inference(artifact_path)?;
    let backend = CandleBackend::cuda_wmt14_reference(0)?;
    let teacher = Wmt14Teacher::from_artifact(backend.clone(), &artifact)?;
    let b = &backend;
    let mut captured = Captured::default();
    teacher.generate_cfg_top1_traced(&tokens, input.shape()[0], 1.0, &mut captured)?;
    let plain = teacher.generate_cfg_top1(&tokens, input.shape()[0], 1.0)?;
    if b.read_indices(&plain.codes)? != b.read_indices(&captured.indices["generator.codes"])?
        || b.read_indices(&plain.ids)?
            != b.read_indices(&captured.indices["generator.greedy_ids"])?
        || b.read_f32(&plain.logits)? != b.read_f32(&captured.tensors["generator.final_logits"])?
    {
        return Err("untraced WMT14 CFG generation differs".into());
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
