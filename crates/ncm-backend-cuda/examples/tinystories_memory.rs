//! Synchronized strict TinyStories CUDA pool observation; not a timing gate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::TinyStoriesTeacher;
use std::time::Instant;
// Safety: the context is bound to this thread, the CUDA-owned pool handle is
// not destroyed, and each attribute writes exactly one live u64 output value.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn pool(backend: &CandleBackend) -> Result<Option<serde_json::Value>, Box<dyn std::error::Error>> {
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
fn pool(_: &CandleBackend) -> Result<Option<serde_json::Value>, Box<dyn std::error::Error>> {
    Ok(None)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: tinystories_sample_bench ARTIFACT BATCH".into());
    }
    let batch: usize = args[2].parse()?;
    if !(1..=128).contains(&batch) {
        return Err("batch must be in 1..=128".into());
    }
    #[cfg(feature = "cuda-cudnn")]
    let backend = CandleBackend::cuda_cudnn(0)?;
    #[cfg(not(feature = "cuda-cudnn"))]
    let backend = CandleBackend::cuda(0)?;
    let artifact = Artifact::load(&args[1])?;
    let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    backend.device().synchronize()?;
    let after_load = pool(&backend)?;
    let prefix = vec![0; batch];
    let mut samples = Vec::with_capacity(1);
    for trial in 0..2 {
        backend.device().synchronize()?;
        let start = Instant::now();
        let result = model.sample_seeded(&prefix, batch, 42)?;
        let codes = backend.read_indices(&result.codes)?;
        let ids = backend.read_indices(&result.ids)?;
        let elapsed = start.elapsed().as_secs_f64();
        if codes.len() != batch * 641 || ids.len() != batch * 128 {
            return Err("wrong output shape".into());
        }
        if trial >= 1 {
            samples.push(elapsed);
        }
    }
    backend.device().synchronize()?;
    let after_requests = pool(&backend)?;
    println!(
        "{}",
        serde_json::json!({"after_load":after_load,"after_requests":after_requests,"pool_scope":"CUDA default async pool only; high water includes load; excludes other allocations","batch":batch,"warmups":1,"trials_seconds":samples,"scope":"generation plus code and final-ID host readback; model loading excluded","rng":"SplitMix64 and inverse CDF, seed 42; not PyTorch's RNG","status":"exploratory","backend":if cfg!(feature = "cuda-cudnn") { "strict-cudnn" } else { "cuda" }})
    );
    Ok(())
}
