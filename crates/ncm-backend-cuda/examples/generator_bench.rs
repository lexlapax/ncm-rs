//! Synchronized eager Sudoku generation timing on one explicit backend.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Generator;
use std::{error::Error, fs, time::Instant};
// Safety: the context is bound to this thread, the CUDA-owned pool handle is
// not destroyed, and each attribute writes exactly one live u64 output value.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn pool(backend: &CandleBackend) -> Result<Option<serde_json::Value>, Box<dyn Error>> {
    if !backend.device().is_cuda() {
        return Ok(None);
    }
    use candle_core::cuda_backend::cudarc::driver::sys;
    let device = backend.device().as_cuda_device()?;
    device.cuda_stream().context().bind_to_thread()?;
    let mut handle = std::ptr::null_mut();
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut handle, device.cuda_stream().context().cu_device())
            .result()?;
    }
    use sys::CUmemPool_attribute::*;
    let mut values = std::collections::BTreeMap::new();
    for (name, attribute) in [
        ("used_current", CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
        ("used_peak", CU_MEMPOOL_ATTR_USED_MEM_HIGH),
        ("reserved_current", CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
        ("reserved_peak", CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH),
    ] {
        let mut value = 0u64;
        unsafe {
            sys::cuMemPoolGetAttribute(handle, attribute, (&mut value as *mut u64).cast())
                .result()?;
        }
        values.insert(name, value);
    }
    Ok(Some(serde_json::json!(values)))
}
#[cfg(not(feature = "cuda"))]
fn pool(_: &CandleBackend) -> Result<Option<serde_json::Value>, Box<dyn Error>> {
    Ok(None)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: generator_bench ARTIFACT INPUT_JSON BATCH_SIZE cpu|cuda".into());
    }
    let batch: usize = args[3].parse()?;
    if !(1..=128).contains(&batch) {
        return Err("batch must be 1..=128".into());
    }
    let input: Vec<Vec<u32>> = serde_json::from_slice(&fs::read(&args[2])?)?;
    if input.len() < batch || input[..batch].iter().any(|x| x.len() != 91) {
        return Err("invalid input rows".into());
    }
    let prefix: Vec<u32> = input[..batch].iter().flatten().copied().collect();
    let backend = match args[4].as_str() {
        "cuda" => CandleBackend::cuda(0)?,
        "cpu" => CandleBackend::cpu(),
        _ => return Err("explicit cpu|cuda required".into()),
    };
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
    let pool = pool(&backend)?;
    println!(
        "{}",
        serde_json::json!({"batch":batch,"warmups":5,"trials":samples,"outputs_sha256_i64le":expected,"scope":"generate plus code and final-ID host readback; excludes model load","cuda_pool":pool})
    );
    Ok(())
}
