//! Bounded alternating CPU/device sampler screen, not a performance gate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Lm1bSample, Lm1bSmallTeacher};
use std::{error::Error, fs, time::Instant};

fn sample(
    model: &Lm1bSmallTeacher<CandleBackend>,
    device: bool,
    seed: u64,
) -> Result<Lm1bSample<CandleBackend>, Box<dyn Error>> {
    if device {
        Ok(model.sample_seeded_device(&[101; 8], 8, seed)?)
    } else {
        Ok(model.sample_seeded(&[101; 8], 8, seed)?)
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("expected artifact path")?;
    if sha256(&fs::read(format!("{path}/manifest.json"))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("unexpected artifact".into());
    }
    let artifact = Artifact::load_inference(path)?;
    let backend = CandleBackend::cuda_cudnn(0)?;
    let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut rows = Vec::new();
    for round in 0..4_u64 {
        let order = if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut outputs = Vec::new();
        for device in order {
            let start = Instant::now();
            let result = sample(&model, device, (11_u64 << 32) | round)?;
            let codes = backend.read_indices(&result.codes)?;
            let elapsed = start.elapsed().as_secs_f64();
            outputs.push((codes, result.ids));
            if round > 0 {
                rows.push(serde_json::json!({"round":round,"device":device,"seconds":elapsed}));
            }
        }
        if outputs[0] != outputs[1] {
            return Err(format!("CPU/device outputs differ at round {round}").into());
        }
    }
    println!(
        "{}",
        serde_json::json!({"scope":"alternating batch-8 development screen; not a performance gate", "warmup_rounds":1,"measured_rounds":3,"outputs_equal":true,"measurements":rows})
    );
    Ok(())
}
