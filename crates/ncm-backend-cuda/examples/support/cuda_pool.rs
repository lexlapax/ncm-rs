use ncm_backend_cuda::CandleBackend;

// SAFETY: the CUDA context is bound to this thread, the pool is CUDA-owned,
// and each synchronous attribute query writes one live u64.
#[allow(unsafe_code)]
pub(crate) fn snapshot(
    backend: &CandleBackend,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    use sys::CUmemPool_attribute::*;
    let device = backend.device().as_cuda_device()?;
    device.cuda_stream().context().bind_to_thread()?;
    let mut handle = std::ptr::null_mut();
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut handle, device.cuda_stream().context().cu_device())
            .result()?;
    }
    let mut values = std::collections::BTreeMap::new();
    for (name, attribute) in [
        ("used_current", CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
        ("used_peak", CU_MEMPOOL_ATTR_USED_MEM_HIGH),
        ("reserved_current", CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
        ("reserved_peak", CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH),
    ] {
        let mut value = 0_u64;
        unsafe {
            sys::cuMemPoolGetAttribute(handle, attribute, (&mut value as *mut u64).cast())
                .result()?;
        }
        values.insert(name, value);
    }
    Ok(serde_json::json!(values))
}
