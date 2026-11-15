//! Compare each stage of the TinyStories first-level CUDA cascade.
use candle_core::Tensor;
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::Codec;
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeMap, io::Write};
fn compare(
    backend: &CandleBackend,
    reference: &SafeTensors<'_>,
    name: &str,
    tensor: &Tensor,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let expected = reference.tensor(name)?;
    let actual = backend.read_f32(tensor)?;
    if expected.dtype() != Dtype::F32
        || expected.shape() != backend.shape(tensor)
        || expected.data().len() != actual.len() * 4
    {
        return Err(format!("invalid reference shape: {name}").into());
    }
    let mut max: f64 = 0.0;
    let mut rms = 0.0;
    let mut outside = 0usize;
    let mut max_normalized: f64 = 0.0;
    for (x, bytes) in actual.iter().zip(expected.data().as_chunks::<4>().0) {
        let y = f32::from_le_bytes(*bytes);
        if !x.is_finite() || !y.is_finite() {
            return Err("nonfinite tensor".into());
        }
        let delta = (f64::from(*x) - f64::from(y)).abs();
        let normalized = delta / (5e-4 + 2e-4 * f64::from(y).abs());
        max = max.max(delta);
        rms += delta * delta;
        outside += usize::from(normalized > 1.0);
        max_normalized = max_normalized.max(normalized);
    }
    Ok(
        serde_json::json!({"elements":actual.len(),"max_abs":max,"rms":(rms/actual.len() as f64).sqrt(),"phase2_accumulated_band_outside":outside,"max_normalized":max_normalized}),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: tinystories_cascade_probe ARTIFACT REFERENCE OUTPUT [SAMPLES]".into());
    }
    let bytes = std::fs::read(&args[2])?;
    let reference = SafeTensors::deserialize(&bytes)?;
    #[cfg(feature = "cuda-cudnn")]
    let backend = CandleBackend::cuda_cudnn(0)?;
    #[cfg(not(feature = "cuda-cudnn"))]
    let backend = CandleBackend::cuda(0)?;
    let artifact = Artifact::load(&args[1])?;
    let codec = Codec::from_artifact(backend.clone(), &artifact)?;
    let weights = codec.weights();
    let initial = if let Some(path) = args.get(4) {
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        data["codes"]
            .as_array()
            .ok_or("missing codes")?
            .iter()
            .take(128)
            .map(|row| {
                row.as_array()
                    .and_then(|r| r.first())
                    .and_then(serde_json::Value::as_u64)
                    .filter(|&i| i < 4096)
                    .map(|i| i as u32)
                    .ok_or("invalid code")
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![0, 37]
    };
    if initial.is_empty() {
        return Err("empty code list".into());
    }
    let ids = backend.upload_indices(&initial, &[initial.len(), 1])?;
    let mut x = backend.embedding(
        weights
            .get("vq.embeddings.0.weight")
            .ok_or("missing book")?,
        &ids,
    )?;
    let mut results = BTreeMap::new();
    let mut captured: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
    let mut save = |name: &str, x: &Tensor| -> Result<(), Box<dyn std::error::Error>> {
        let values = backend.read_f32(x)?;
        let data = values.into_iter().flat_map(f32::to_le_bytes).collect();
        captured.insert(name.to_owned(), (backend.shape(x), data));
        Ok(())
    };
    save("embedding", &x)?;
    results.insert(
        "embedding".to_owned(),
        compare(&backend, &reference, "embedding", &x)?,
    );
    let normalized = codec
        .config()
        .raw()
        .vector_quantizer_config
        .learned_l1_cascade_norm;
    for (i, stride) in codec.config().strides().iter().rev().enumerate() {
        if normalized && i > 0 {
            x = backend.group_norm(&x, 8, 1e-5)?;
            let name = format!("cascade.{}", i * 2 - 1);
            results.insert(name.clone(), compare(&backend, &reference, &name, &x)?);
            save(&name, &x)?;
        }
        let index = if normalized { i * 2 } else { i };
        let w = weights
            .get(&format!("vq.learned_up_l1.{index}.weight"))
            .ok_or("missing weight")?;
        let b = weights
            .get(&format!("vq.learned_up_l1.{index}.bias"))
            .ok_or("missing bias")?;
        x = backend.conv1d(&x, w, b, *stride, 0, true)?;
        let name = format!("cascade.{index}");
        results.insert(name.clone(), compare(&backend, &reference, &name, &x)?);
        save(&name, &x)?;
    }
    let w = weights
        .get("vq.quant_resi.0.weight")
        .ok_or("missing residual weight")?;
    let bias = weights
        .get("vq.quant_resi.0.bias")
        .ok_or("missing residual bias")?;
    let mut timings = Vec::new();
    for trial in 0..35 {
        backend.device().synchronize()?;
        let start = std::time::Instant::now();
        let value = backend.conv1d(&x, w, bias, 1, 1, false)?;
        backend.evaluate(&[value])?;
        if trial >= 5 {
            timings.push(start.elapsed().as_secs_f64());
        }
    }
    results.insert(
        "residual.benchmark".to_owned(),
        serde_json::json!({"samples_seconds": timings, "warmups": 5}),
    );
    let conv = backend.conv1d(&x, w, bias, 1, 1, false)?;
    for (name, value) in [
        ("residual.conv", conv.clone()),
        ("residual.input_half", backend.scale(&x, 0.5)?),
        ("residual.conv_half", backend.scale(&conv, 0.5)?),
    ] {
        if reference.tensor(name).is_ok() {
            results.insert(
                name.to_owned(),
                compare(&backend, &reference, name, &value)?,
            );
            save(name, &value)?;
        }
    }
    x = codec.refine(&x, 0)?;
    results.insert(
        "refined".to_owned(),
        compare(&backend, &reference, "refined", &x)?,
    );
    save("refined", &x)?;
    let views = captured
        .iter()
        .map(|(name, (shape, data))| {
            Ok((
                name.as_str(),
                safetensors::tensor::TensorView::new(Dtype::F32, shape.clone(), data)?,
            ))
        })
        .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
    let bytes = safetensors::tensor::serialize(views, None)?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?
        .write_all(&bytes)?;
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
