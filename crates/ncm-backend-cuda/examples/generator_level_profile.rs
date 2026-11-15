//! Synchronizing diagnostic timings by level; not an uninstrumented benchmark.
use candle_core::Tensor;
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::{Generator, TraceSink};
use std::{error::Error, fs, time::Instant};
struct Capture {
    backend: CandleBackend,
    previous: Instant,
    levels: Vec<f64>,
}
impl TraceSink<CandleBackend> for Capture {
    fn tensor(&mut self, _: &str, value: &Tensor) -> ncm_core::Result<()> {
        self.backend.read_f32(value)?;
        let now = Instant::now();
        self.levels
            .push(now.duration_since(self.previous).as_secs_f64());
        self.previous = now;
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &Tensor) -> ncm_core::Result<()> {
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: generator_level_profile ARTIFACT INPUT_JSON BATCH".into());
    }
    let batch: usize = args[3].parse()?;
    if !(1..=128).contains(&batch) {
        return Err("invalid batch".into());
    }
    let input: Vec<Vec<u32>> = serde_json::from_slice(&fs::read(&args[2])?)?;
    if input.len() < batch || input[..batch].iter().any(|r| r.len() != 91) {
        return Err("invalid inputs".into());
    }
    let tokens: Vec<u32> = input[..batch].iter().flatten().copied().collect();
    let b = CandleBackend::cuda(0)?;
    let model = Generator::from_artifact(b.clone(), &Artifact::load(&args[1])?)?;
    let mut records = Vec::new();
    for _ in 0..4 {
        let mut trace = Capture {
            backend: b.clone(),
            previous: Instant::now(),
            levels: Vec::new(),
        };
        let out = model.generate_traced(&tokens, batch, &mut trace)?;
        b.read_indices(&out.codes)?;
        b.read_indices(&out.ids)?;
        let tail = trace.previous.elapsed().as_secs_f64();
        records.push(serde_json::json!({"levels_seconds":trace.levels,"tail_seconds":tail}));
    }
    println!(
        "{}",
        serde_json::json!({"batch":batch,"runs":records,"scope":"instrumented: each level reads all logits and synchronizes"})
    );
    Ok(())
}
