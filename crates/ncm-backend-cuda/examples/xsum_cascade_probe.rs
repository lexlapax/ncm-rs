//! Compare XSum's four-position first cascade with a pinned CUDA trace.
use candle_core::Tensor;
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::Artifact;
use ncm_model::Codec;
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;

fn compare(
    backend: &CandleBackend,
    reference: &SafeTensors<'_>,
    name: &str,
    value: &Tensor,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let expected = reference.tensor(name)?;
    let actual = backend.read_f32(value)?;
    if expected.dtype() != Dtype::F32
        || expected.shape() != backend.shape(value)
        || expected.data().len() != actual.len() * 4
    {
        return Err(format!("shape or dtype differs: {name}").into());
    }
    let mut outside = 0usize;
    let mut max_abs: f64 = 0.0;
    let mut sum = 0.0;
    for (x, bytes) in actual.iter().zip(expected.data().as_chunks::<4>().0) {
        let y = f32::from_le_bytes(*bytes);
        if !x.is_finite() || !y.is_finite() {
            return Err("nonfinite trace".into());
        }
        let delta = (f64::from(*x) - f64::from(y)).abs();
        max_abs = max_abs.max(delta);
        sum += delta * delta;
        outside += usize::from(delta > 1e-4 + 2e-4 * f64::from(y).abs());
    }
    Ok(
        serde_json::json!({"elements": actual.len(), "max_abs": max_abs,
        "rms": (sum / actual.len() as f64).sqrt(), "component_band_outside": outside}),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: xsum_cascade_probe ARTIFACT REFERENCE".into());
    }
    let data = std::fs::read(&args[2])?;
    let reference = SafeTensors::deserialize(&data)?;
    let backend = CandleBackend::cuda_cudnn(0)?;
    let artifact = Artifact::load_inference(&args[1])?;
    let codec = Codec::from_artifact(backend.clone(), &artifact)?;
    let ids = backend.upload_indices(&[0, 37, 2048, 4095, 123, 456, 789, 42], &[2, 4])?;
    let book = codec
        .weights()
        .get("vq.embeddings.0.weight")
        .ok_or("missing book")?;
    let embedding = backend.embedding(book, &ids)?;
    let up = codec.cascade(&embedding, true)?;
    let refined = codec.refine(&up, 0)?;
    let values: Vec<f32> = (0..2 * 48 * 384)
        .map(|i| ((i % 127) as f32 - 63.0) / 64.0)
        .collect();
    let down_input = backend.upload_f32(&values, &[2, 48, 384])?;
    let down = codec.cascade(&down_input, false)?;
    let mut result = BTreeMap::new();
    for (name, value) in [
        ("embedding", &embedding),
        ("up.2", &up),
        ("refined", &refined),
        ("down.input", &down_input),
        ("down.3", &down),
    ] {
        result.insert(name, compare(&backend, &reference, name, value)?);
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
