//! Non-held-out LM1B batch capacity diagnostic; not a release gate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Lm1bSmallTeacher;
use safetensors::{Dtype, SafeTensors};
use std::{error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: lm1b_small_capacity_probe ARTIFACT BATCH".into());
    }
    let batch: usize = args[2].parse()?;
    if ![1, 4, 16].contains(&batch) {
        return Err("only diagnostic batches 1, 4 and 16 supported".into());
    }
    if sha256(&fs::read(format!("{}/manifest.json", args[1]))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("artifact differs".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    let rope_path = std::env::var_os("NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC");
    let backend = if rope_path.is_some() {
        CandleBackend::cuda_lm1b_qkv_reference_experimental(0)?
    } else {
        CandleBackend::cuda_cudnn(0)?
    };
    let mut model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
    if let Some(path) = &rope_path {
        let data = fs::read(path)?;
        if sha256(&data) != "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc" {
            return Err("unexpected diagnostic rotary tables".into());
        }
        let tables = SafeTensors::deserialize(&data)?;
        let values = |name| -> Result<Vec<f32>, Box<dyn Error>> {
            let tensor = tables.tensor(name)?;
            if tensor.dtype() != Dtype::F32 || tensor.shape() != [128, 32] {
                return Err("unexpected rotary tensor shape or type".into());
            }
            Ok(tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect())
        };
        model = model.with_reference_rope_diagnostic(&values("cosine")?, &values("sine")?)?;
    }
    let prefix = vec![101; batch];
    let mut times = Vec::new();
    let mut signatures = Vec::new();
    for round in [30_u64, 31, 0, 1, 2, 3, 4, 5, 6, 7] {
        let started = Instant::now();
        let output = model.sample_seeded_fully_device(&prefix, batch, (11_u64 << 32) | round)?;
        let codes = backend.read_indices(&output.codes)?;
        let elapsed = started.elapsed().as_secs_f64();
        if codes.len() != batch * 641
            || output.ids.len() != batch * 128
            || codes.iter().any(|&x| x >= 8192)
            || output.ids.iter().any(|&x| x >= 30528)
        {
            return Err("invalid output".into());
        }
        if round < 8 {
            times.push(elapsed);
            signatures.push(sha256(&serde_json::to_vec(&(codes, output.ids))?));
        }
    }
    let status = fs::read_to_string("/proc/self/status")?;
    let hwm = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .ok_or("missing VmHWM")?;
    println!(
        "{}",
        serde_json::json!({"scope":"non-held-out LM1B capacity screen", "runtime":"rust", "batch":batch, "warmups":2, "trials":8, "seconds":times, "signatures":signatures, "vmhwm":hwm, "diagnostic": if rope_path.is_some() { "reference tables and 1 MiB QKV preference" } else { "default" }})
    );
    Ok(())
}
