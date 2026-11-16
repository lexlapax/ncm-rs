//! Retain CUDA XSum head logits under saved seeded code histories.
use candle_core::Tensor;
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, Error};
use ncm_io::Artifact;
use ncm_model::{TraceSink, XsumTeacher};
use safetensors::{Dtype, tensor::TensorView};
use std::{collections::BTreeMap, io::Write};

struct Capture {
    backend: CandleBackend,
    heads: BTreeMap<String, (Vec<usize>, Vec<u8>)>,
}
impl TraceSink<CandleBackend> for Capture {
    fn tensor(&mut self, name: &str, value: &Tensor) -> ncm_core::Result<()> {
        if name.starts_with("generator.level_") && name.ends_with(".logits") {
            let data = self.backend.read_f32(value)?;
            if data.iter().any(|v| !v.is_finite()) {
                return Err(Error::Weights("nonfinite shared-history logits".into()));
            }
            self.heads.insert(
                name.to_owned(),
                (
                    self.backend.shape(value),
                    data.into_iter().flat_map(f32::to_le_bytes).collect(),
                ),
            );
        }
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &Tensor) -> ncm_core::Result<()> {
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: xsum_shared_history_capture ARTIFACT INPUT REFERENCE OUTPUT".into());
    }
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let codes: Vec<u32> = serde_json::from_value(reference["codes"].clone())?;
    if rows.len() != 32
        || rows.iter().any(|r| r.len() != 512)
        || codes.len() != 32 * 200
        || reference["seed"] != 42
        || reference["code_differences"] != 0
        || reference["id_differences"] != 0
    {
        return Err("invalid shared-history inputs".into());
    }
    let prefix: Vec<u32> = rows.into_iter().flatten().collect();
    let backend = CandleBackend::cuda_cudnn(0)?;
    let artifact = Artifact::load_inference(&args[1])?;
    let model = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut capture = Capture {
        backend,
        heads: BTreeMap::new(),
    };
    model.run_traced(&prefix, &codes, 32, &mut capture)?;
    if capture.heads.len() != 9 {
        return Err("incomplete shared-history logits".into());
    }
    let views = capture
        .heads
        .iter()
        .map(|(name, (shape, data))| {
            Ok((
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), data)?,
            ))
        })
        .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
    let bytes = safetensors::tensor::serialize(views, None)?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?
        .write_all(&bytes)?;
    println!(
        "{}",
        serde_json::json!({"rows":32,"levels":9,"forced_codes":codes.len(),"bytes":bytes.len()})
    );
    Ok(())
}
