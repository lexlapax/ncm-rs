//! Source-bound LM1B-small batch-eight timing probe; not a release gate alone.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Lm1bSmallTeacher;
use serde::Deserialize;
use std::{error::Error, fs, time::Instant};

#[derive(Deserialize)]
struct Capture {
    size: String,
    seed: u64,
    count: usize,
    batch_size: usize,
    codes: Vec<Vec<u32>>,
    raw_ids: Vec<Vec<u32>>,
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) || args.get(3).is_some_and(|x| x != "device-penalty") {
        return Err(
            "usage: lm1b_small_performance_probe ARTIFACT DEVELOPMENT_CAPTURE [device-penalty]"
                .into(),
        );
    }
    if sha256(&fs::read(format!("{}/manifest.json", args[1]))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("artifact differs".into());
    }
    let bytes = fs::read(&args[2])?;
    if sha256(&bytes) != "68fa3135e880e141a683fbbdfe2b49b6aa6834dfb0748ebe5f15eec31641df3e" {
        return Err("capture differs".into());
    }
    let reference: Capture = serde_json::from_slice(&bytes)?;
    if (
        reference.size.as_str(),
        reference.seed,
        reference.count,
        reference.batch_size,
    ) != ("small", 11, 256, 8)
        || reference.codes.len() != 256
        || reference.raw_ids.len() != 256
    {
        return Err("capture metadata differs".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    let backend = CandleBackend::cuda_cudnn(0)?;
    let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut seconds = Vec::new();
    for batch in [30_u64, 31, 0, 1, 2, 3, 4, 5, 6, 7] {
        let start = Instant::now();
        let out = if args.len() == 4 {
            model.sample_seeded_fully_device(&[101; 8], 8, (11_u64 << 32) | batch)?
        } else {
            model.sample_seeded_device(&[101; 8], 8, (11_u64 << 32) | batch)?
        };
        let codes = backend.read_indices(&out.codes)?;
        let elapsed = start.elapsed().as_secs_f64();
        let index = batch as usize * 8;
        let expected_codes: Vec<u32> = reference.codes[index..index + 8]
            .iter()
            .flatten()
            .copied()
            .collect();
        let expected_ids: Vec<u32> = reference.raw_ids[index..index + 8]
            .iter()
            .flatten()
            .copied()
            .collect();
        if codes != expected_codes || out.ids != expected_ids {
            return Err(format!("outputs differ for batch {batch}").into());
        }
        if batch < 8 {
            seconds.push(elapsed);
        }
    }
    let status = fs::read_to_string("/proc/self/status")?;
    let hwm = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .ok_or("missing VmHWM")?
        .to_owned();
    println!(
        "{}",
        serde_json::json!({"scope":"LM1B development timing; not release gate", "runtime":"rust", "batch":8, "warmups":2, "trials":8, "seconds":seconds, "output_matches_capture":true, "vmhwm":hwm,
            "penalty": if args.len() == 4 { "device" } else { "cpu" }})
    );
    Ok(())
}
