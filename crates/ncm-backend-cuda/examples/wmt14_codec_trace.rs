//! Experimental WMT14 CUDA trace; results require a separate oracle comparison.
#[path = "../../../tests/support/codec_capture.rs"]
mod capture;
use capture::Captured;
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Codec, TraceSink};
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
        return Err("usage: wmt14_codec_trace ARTIFACT REFERENCE OUTPUT".into());
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
        != "164dc6b368dd71f6b6541a99aa50c2c514b3349a6fb6b2e787b3d5a5fbeef586"
    {
        return Err("unexpected reference report".into());
    }
    let data = fs::read(reference.join("traces.safetensors"))?;
    if sha256(&data) != "b4b66ab0b58910e501aed704d50df9493bad65d691e78738f8689d14649662fd" {
        return Err("unexpected reference trace".into());
    }
    let reference_tensors = SafeTensors::deserialize(&data)?;
    let input = reference_tensors.tensor("input.ids")?;
    let supplied = reference_tensors.tensor("supplied.codes")?;
    let tokens = indices(input.clone())?;
    let codes = indices(supplied.clone())?;
    let artifact = Artifact::load_inference(artifact_path)?;
    let codec = Codec::from_wmt14_artifact_experimental(CandleBackend::cuda(0)?, &artifact)?;
    let b = codec.backend();
    let mut captured = Captured::default();
    captured.indices("input.ids", &b.upload_indices(&tokens, input.shape())?)?;
    captured.indices(
        "supplied.codes",
        &b.upload_indices(&codes, supplied.shape())?,
    )?;
    codec.reconstruct_traced(&tokens, input.shape()[0], &mut captured)?;
    codec.decode_traced(&codes, input.shape()[0], "forced", &mut captured)?;
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
