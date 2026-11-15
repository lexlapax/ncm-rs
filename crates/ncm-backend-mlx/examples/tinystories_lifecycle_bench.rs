//! Exploratory TinyStories output-lifetime barrier and optional cache cap.
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::Artifact;
use ncm_model::TinyStoriesTeacher;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: tinystories_lifecycle_bench ARTIFACT BATCH default|CACHE_MIB".into());
    }
    let batch: usize = args[2].parse()?;
    if !(1..=128).contains(&batch) {
        return Err("batch must be in 1..=128".into());
    }
    let cache_limit = if args[3] == "default" {
        None
    } else {
        let mib: usize = args[3].parse()?;
        if mib > 16 * 1024 {
            return Err("cache limit exceeds diagnostic bound".into());
        }
        Some(mib.checked_mul(1024 * 1024).ok_or("cache limit overflow")?)
    };
    let previous_cache_limit = cache_limit.map(memory::set_cache_limit).transpose()?;
    let backend = MlxBackend::gpu()?;
    let artifact = Artifact::load_inference(&args[1])?;
    let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let marker = backend.upload_f32(&[1.0], &[1])?;
    let prefix = vec![0; batch];
    let mut samples = Vec::with_capacity(30);
    for trial in 0..35 {
        let start = Instant::now();
        let result = model.sample_seeded(&prefix, batch, 42)?;
        drop(result.logits);
        let codes = backend.read_indices(&result.codes)?;
        let ids = backend.read_indices(&result.ids)?;
        drop(result.codes);
        drop(result.ids);
        let barrier = backend.add(&marker, &marker)?;
        backend.evaluate(&[barrier])?;
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
        serde_json::json!({"batch":batch,"warmups":5,"trials_seconds":samples,"scope":"generation plus code and final-ID host readback; model loading excluded","rng":"SplitMix64 and inverse CDF, seed 42; not PyTorch's RNG","status":"exploratory","backend":"mlx", "cleanup":"drop outputs and evaluate GPU barrier within timing", "cache_limit_bytes":cache_limit,"previous_cache_limit_bytes":previous_cache_limit, "mlx_memory":{"active":memory::active_memory()?,"peak":memory::peak_memory()?,"cache":memory::cache_memory()?}})
    );
    Ok(())
}
