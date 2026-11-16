//! Exploratory strict-cuDNN XSum timing; not a quality or acceptance gate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::{TraceSink, XsumTeacher};
use std::time::Instant;

struct StageTiming<'a> {
    backend: &'a CandleBackend,
    start: Instant,
    decoder_done: Option<f64>,
}
impl TraceSink<CandleBackend> for StageTiming<'_> {
    fn tensor(&mut self, _: &str, _: &candle_core::Tensor) -> ncm_core::Result<()> {
        Ok(())
    }
    fn indices(&mut self, name: &str, _: &candle_core::Tensor) -> ncm_core::Result<()> {
        if name == "generator.greedy_ids" {
            self.backend
                .device()
                .synchronize()
                .map_err(|e| ncm_core::Error::Backend(e.to_string()))?;
            self.decoder_done = Some(self.start.elapsed().as_secs_f64());
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=6).contains(&args.len())
        || args.get(4).is_some_and(|s| s != "device")
        || args.get(5).is_some_and(|s| s != "30")
    {
        return Err("usage: xsum_perf_screen ARTIFACT INPUT_JSON BATCH [device [30]]".into());
    }
    let measured = if args.len() == 6 { 30 } else { 5 };
    let warmups = if measured == 30 { 5 } else { 2 };
    let device_selection = args.len() >= 5;
    let batch: usize = args[3].parse()?;
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if !(1..=32).contains(&batch)
        || rows.len() < batch
        || rows[..batch].iter().any(|r| r.len() != 512)
    {
        return Err("need 1..=32 rows of 512 tokens".into());
    }
    let input: Vec<u32> = rows[..batch].iter().flatten().copied().collect();
    let backend = CandleBackend::cuda_cudnn(0)?;
    let artifact = Artifact::load_inference(&args[1])?;
    let model = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let mut samples = Vec::new();
    let mut decoder_samples = Vec::new();
    let mut signature = None;
    for trial in 0..warmups + measured {
        backend.device().synchronize()?;
        let start = Instant::now();
        let mut trace = StageTiming {
            backend: &backend,
            start,
            decoder_done: None,
        };
        let output = if device_selection {
            model.generate_top1_device(&input, batch)?
        } else {
            model.generate_top1_traced(&input, batch, &mut trace)?
        };
        let codes = backend.read_indices(&output.codes)?;
        let seconds = start.elapsed().as_secs_f64();
        if codes.len() != batch * 200 || output.ids.len() != batch * 48 {
            return Err("wrong generation output shape".into());
        }
        let current = (codes, output.ids);
        if signature.as_ref().is_some_and(|prior| prior != &current) {
            return Err("generation changed between trials".into());
        }
        signature = Some(current);
        if trial >= warmups {
            samples.push(seconds);
            if !device_selection {
                decoder_samples.push(trace.decoder_done.ok_or("missing decoder timing")?);
            }
        }
    }
    let (codes, ids) = signature.ok_or("no measured output")?;
    println!(
        "{}",
        serde_json::json!({"batch":batch,"warmups":warmups,"trials_seconds":samples,
        "decoder_seconds":decoder_samples,
        "scope":"generation, FP64 final selection and code/ID host readback; load excluded; CPU mode has an extra decoder synchronization",
        "status":"exploratory","backend":"strict-cudnn", "selection":if device_selection {"device"} else {"cpu"}, "codes":codes,"ids":ids})
    );
    Ok(())
}
