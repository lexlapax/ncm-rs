//! Isolated candidate probe: run its first residual on retained CUDA inputs.
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::Codec;
use safetensors::{Dtype, SafeTensors};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: probe ARTIFACT CUDA_REFERENCE OUTPUT".into());
    }
    let bytes = std::fs::read(&args[2])?;
    let reference = SafeTensors::deserialize(&bytes)?;
    let input = reference.tensor("cascade.3")?;
    if input.dtype() != Dtype::F32 || input.shape() != [128, 128, 384] {
        return Err("unexpected reference input".into());
    }
    let count = 16 * 128 * 384;
    let values: Vec<f32> = input.data().as_chunks::<4>().0[..count]
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let backend = MlxBackend::gpu()?;
    let artifact = Artifact::load(&args[1])?;
    let codec = Codec::from_artifact(backend.clone(), &artifact)?;
    let x = backend.upload_f32(&values, &[16, 128, 384])?;
    let conv = backend.first_refinement_conv1d(
        &x,
        codec
            .weights()
            .get("vq.quant_resi.0.weight")
            .ok_or("missing weight")?,
        codec
            .weights()
            .get("vq.quant_resi.0.bias")
            .ok_or("missing bias")?,
    )?;
    let refined = backend.add(&backend.scale(&x, 0.5)?, &backend.scale(&conv, 0.5)?)?;
    let result = backend.read_f32(&refined)?;
    let input_bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    let result_bytes: Vec<u8> = result.into_iter().flat_map(f32::to_le_bytes).collect();
    let shape = vec![16, 128, 384];
    let views = vec![
        (
            "cascade.3",
            safetensors::tensor::TensorView::new(Dtype::F32, shape.clone(), &input_bytes)?,
        ),
        (
            "refined",
            safetensors::tensor::TensorView::new(Dtype::F32, shape, &result_bytes)?,
        ),
    ];
    let output = safetensors::tensor::serialize(views, None)?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?
        .write_all(&output)?;
    Ok(())
}
