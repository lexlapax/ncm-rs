//! Non-held-out LM1B batch capacity diagnostic; not a release gate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Lm1bSmallTeacher;
use safetensors::{Dtype, SafeTensors};
use std::{error::Error, fs, time::Instant};

#[path = "support/cuda_pool.rs"]
mod cuda_pool;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact, batch, options @ ..] = args.as_slice() else {
        return Err(
            "usage: lm1b_small_capacity_probe ARTIFACT BATCH [repeated] [cudnn-default-math|cudnn-graph|cudnn-graph-efficient]"
                .into(),
        );
    };
    let mut repeated = false;
    let mut convolution = "strict FMA";
    for option in options {
        match option.as_str() {
            "repeated" if !repeated => repeated = true,
            "cudnn-default-math" if convolution == "strict FMA" => {
                convolution = "cuDNN default math"
            }
            "cudnn-graph" if convolution == "strict FMA" => convolution = "cuDNN graph",
            "cudnn-graph-efficient" if convolution == "strict FMA" => {
                convolution = "cuDNN graph and efficient attention"
            }
            _ => return Err("invalid or repeated LM1B capacity option".into()),
        }
    }
    let (warmups, trials) = if repeated { (5, 30) } else { (2, 8) };
    let batch: usize = batch.parse()?;
    if ![1, 4, 16].contains(&batch) {
        return Err("only diagnostic batches 1, 4 and 16 supported".into());
    }
    if sha256(&fs::read(format!("{artifact}/manifest.json"))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("artifact differs".into());
    }
    let load_start = Instant::now();
    let artifact = Artifact::load_inference(artifact)?;
    let rope_path = std::env::var_os("NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC");
    if convolution != "strict FMA" && rope_path.is_none() {
        return Err("experimental cuDNN modes require the reference rotary diagnostic".into());
    }
    let backend = if convolution == "cuDNN graph and efficient attention" {
        CandleBackend::cuda_lm1b_efficient_experimental(0)?
    } else if convolution == "cuDNN graph" {
        CandleBackend::cuda_lm1b_cudnn_graph_experimental(0)?
    } else if convolution == "cuDNN default math" {
        CandleBackend::cuda_lm1b_cudnn_default_math_experimental(0)?
    } else if rope_path.is_some() {
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
    backend.device().synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let prefix = vec![101; batch];
    let mut times = Vec::new();
    let mut signatures = Vec::new();
    for round in (30..30 + warmups).chain(0..trials) {
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
        if round < trials {
            times.push(elapsed);
            signatures.push(sha256(&serde_json::to_vec(&(codes, output.ids))?));
        }
    }
    backend.device().synchronize()?;
    let pool = cuda_pool::snapshot(&backend)?;
    let status = fs::read_to_string("/proc/self/status")?;
    let hwm = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .ok_or("missing VmHWM")?;
    println!(
        "{}",
        serde_json::json!({"scope":"non-held-out LM1B capacity screen", "runtime":"rust", "batch":batch, "warmups":warmups, "trials":trials, "load_seconds":load_seconds, "cuda_default_pool":pool, "convolution_math":convolution, "seconds":times, "signatures":signatures, "vmhwm":hwm, "diagnostic": if rope_path.is_some() { "reference tables and 1 MiB QKV preference" } else { "default" }})
    );
    Ok(())
}
