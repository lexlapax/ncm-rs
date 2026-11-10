#![cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::sys;
use ncm_backend_cuda::CandleBackend;

// This integration-test process owns its CUDA policy; no other test binary shares it.
#[test]
#[allow(unsafe_code)]
fn explicit_pool_retention_and_opt_out() -> Result<(), Box<dyn std::error::Error>> {
    let read = |b: &CandleBackend| -> Result<u64, Box<dyn std::error::Error>> {
        let stream = b.device().as_cuda_device()?.cuda_stream();
        let context = stream.context();
        context.bind_to_thread()?;
        let mut pool = std::ptr::null_mut();
        let mut bytes = 0u64;
        // Safety: the initialized context is current, the pool is CUDA-owned,
        // and this attribute is a u64 written through a live local pointer.
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, context.cu_device()).result()?;
            sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut bytes as *mut u64).cast(),
            )
            .result()?;
        }
        Ok(bytes)
    };
    let default = CandleBackend::cuda(0)?;
    assert_eq!(read(&default)?, 128 * 1024 * 1024);
    assert_eq!(read(&default.clone())?, 128 * 1024 * 1024);
    let eager = CandleBackend::cuda_with_pool_retention(0, 0)?;
    assert_eq!(read(&eager)?, 0);
    let custom = CandleBackend::cuda_with_pool_retention(0, 32 * 1024 * 1024)?;
    assert_eq!(read(&custom)?, 32 * 1024 * 1024);
    Ok(())
}
