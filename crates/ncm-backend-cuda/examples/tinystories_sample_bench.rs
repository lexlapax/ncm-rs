//! Exploratory native TinyStories timing; not a quality or formal gate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::TinyStoriesTeacher;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: tinystories_sample_bench ARTIFACT BATCH".into());
    }
    let batch: usize = args[2].parse()?;
    if !(1..=128).contains(&batch) {
        return Err("batch must be in 1..=128".into());
    }
    let backend = CandleBackend::cuda(0)?;
    let artifact = Artifact::load(&args[1])?;
    let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let prefix = vec![0; batch];
    let mut samples = Vec::with_capacity(30);
    for trial in 0..35 {
        backend.device().synchronize()?;
        let start = Instant::now();
        let result = model.sample_seeded(&prefix, batch, 42)?;
        let codes = backend.read_indices(&result.codes)?;
        let ids = backend.read_indices(&result.ids)?;
        let elapsed = start.elapsed().as_secs_f64();
        if codes.len() != batch * 641 || ids.len() != batch * 128 {
            return Err("wrong output shape".into());
        }
        if trial >= 5 {
            samples.push(elapsed);
        }
    }
    println!(
        "{}",
        serde_json::json!({"batch":batch,"warmups":5,"trials_seconds":samples,"scope":"generation plus code and final-ID host readback; model loading excluded","rng":"SplitMix64 and inverse CDF, seed 42; not PyTorch's RNG","status":"exploratory"})
    );
    Ok(())
}
