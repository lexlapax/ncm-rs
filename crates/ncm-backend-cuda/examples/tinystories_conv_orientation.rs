//! Diagnostic: compare a padded 2-D CUDA convolution with upstream conv1d.
use candle_core::Tensor;
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::Artifact;
use ncm_model::Codec;
use safetensors::{Dtype, SafeTensors};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: tinystories_conv_orientation ARTIFACT REFERENCE".into());
    }
    let raw = std::fs::read(&args[2])?;
    let reference = SafeTensors::deserialize(&raw)?;
    let input = reference.tensor("cascade.3")?;
    let expected = reference.tensor("residual.conv")?;
    if input.dtype() != Dtype::F32
        || expected.dtype() != Dtype::F32
        || input.shape() != [128, 128, 384]
        || input.shape() != expected.shape()
    {
        return Err("invalid trace".into());
    }
    let values: Vec<f32> = input
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    let backend = CandleBackend::cuda(0)?;
    let artifact = Artifact::load(&args[1])?;
    let codec = Codec::from_artifact(backend.clone(), &artifact)?;
    let weight = codec
        .weights()
        .get("vq.quant_resi.0.weight")
        .ok_or("missing weight")?;
    let bias = codec
        .weights()
        .get("vq.quant_resi.0.bias")
        .ok_or("missing bias")?;
    let x = backend.upload_f32(&values, &[128, 128, 384])?;
    let x = x
        .transpose(1, 2)?
        .unsqueeze(2)?
        .contiguous()?
        .pad_with_zeros(3, 1, 1)?;
    let weight = weight.unsqueeze(2)?.contiguous()?;
    let result: Tensor = x
        .conv2d(&weight, 0, 1, 1, 1)?
        .squeeze(2)?
        .transpose(1, 2)?
        .broadcast_add(bias)?;
    let values = backend.read_f32(&result)?;
    let mut max = 0.0f64;
    let mut squared = 0.0;
    let mut different = 0usize;
    for (a, b) in values.iter().zip(expected.data().as_chunks::<4>().0) {
        let b = f32::from_le_bytes(*b);
        let diff = (f64::from(*a) - f64::from(b)).abs();
        max = max.max(diff);
        squared += diff * diff;
        different += usize::from(*a != b);
    }
    println!(
        "{}",
        serde_json::json!({"elements":values.len(),"max_abs":max,"rms":(squared / values.len() as f64).sqrt(),"different":different,"variant":"pre-padded [batch,channels,1,length] conv2d"})
    );
    Ok(())
}
