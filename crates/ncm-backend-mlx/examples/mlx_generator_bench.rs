//! Synchronized greedy Sudoku generation and readback on explicit Metal.
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Generator;
use std::{error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: mlx_generator_bench ARTIFACT INPUT_JSON BATCH_SIZE".into());
    }
    let batch: usize = args[3].parse()?;
    if !(1..=128).contains(&batch) {
        return Err("batch must be 1..=128".into());
    }
    let input: Vec<Vec<u32>> = serde_json::from_slice(&fs::read(&args[2])?)?;
    if input.len() != batch || input.iter().any(|x| x.len() != 91) {
        return Err("input must contain exactly batch rows of length 91".into());
    }
    let prefix: Vec<u32> = input.iter().flatten().copied().collect();
    let backend = MlxBackend::gpu()?;
    let artifact = Artifact::load(&args[1])?;
    let generator = Generator::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let mut samples = Vec::with_capacity(30);
    let mut expected = None;
    for trial in 0..35 {
        let start = Instant::now();
        let out = generator.generate(&prefix, batch)?;
        let codes = backend.read_indices(&out.codes)?;
        let ids = backend.read_indices(&out.ids)?;
        let elapsed = start.elapsed().as_secs_f64();
        let bytes: Vec<_> = codes
            .iter()
            .chain(&ids)
            .flat_map(|&v| i64::from(v).to_le_bytes())
            .collect();
        let hash = sha256(&bytes);
        if let Some(ref previous) = expected {
            if previous != &hash {
                return Err("outputs changed between trials".into());
            }
        } else {
            expected = Some(hash);
        }
        if trial >= 5 {
            samples.push(elapsed);
        }
    }
    println!(
        "{}",
        serde_json::json!({"batch":batch,"warmups":5,"trials":samples,
            "outputs_sha256_i64le":expected,
            "scope":"generate plus code and final-ID host readback; excludes model load",
            "mlx_memory":{"active":memory::active_memory()?,"peak":memory::peak_memory()?,"cache":memory::cache_memory()?}})
    );
    Ok(())
}
