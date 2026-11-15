//! Capture only level-one logits for controlled-sampling diagnosis.
use candle_core::Tensor;
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, Error};
use ncm_io::Artifact;
use ncm_model::{TinyStoriesTeacher, TraceSink};
use std::io::Write;
struct Capture {
    backend: CandleBackend,
    path: String,
    captured: bool,
}
impl TraceSink<CandleBackend> for Capture {
    fn tensor(&mut self, name: &str, x: &Tensor) -> ncm_core::Result<()> {
        if name != "generator.level_01.logits" {
            return Ok(());
        }
        let values = self.backend.read_f32(x)?;
        let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            self.backend.shape(x),
            &bytes,
        )
        .map_err(|e| Error::Weights(e.to_string()))?;
        let data = safetensors::tensor::serialize([("logits", view)], None)
            .map_err(|e| Error::Weights(e.to_string()))?;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)
            .and_then(|mut f| f.write_all(&data))
            .map_err(|e| Error::Weights(e.to_string()))?;
        self.captured = true;
        Err(Error::Input("intentional first-level capture stop".into()))
    }
    fn indices(&mut self, _: &str, _: &Tensor) -> ncm_core::Result<()> {
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("usage: tinystories_first_logits ARTIFACT SAMPLES OUTPUT".into());
    }
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let rows: Vec<Vec<u32>> = serde_json::from_value(data["codes"].clone())?;
    if rows.len() < 128
        || rows
            .iter()
            .take(128)
            .any(|r| r.len() != 641 || r.iter().any(|&x| x >= 4096))
    {
        return Err("invalid code rows".into());
    }
    let codes: Vec<u32> = rows[..128].iter().flatten().copied().collect();
    #[cfg(feature = "cuda-cudnn")]
    let backend = CandleBackend::cuda_cudnn(0)?;
    #[cfg(not(feature = "cuda-cudnn"))]
    let backend = CandleBackend::cuda(0)?;
    let artifact = Artifact::load(&a[1])?;
    let teacher = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut trace = Capture {
        backend,
        path: a[3].clone(),
        captured: false,
    };
    let result = teacher.run_traced(&[0; 128], &codes, 128, &mut trace);
    if !trace.captured
        || !matches!(result,Err(Error::Input(ref e)) if e=="intentional first-level capture stop")
    {
        return Err("capture failed".into());
    }
    Ok(())
}
