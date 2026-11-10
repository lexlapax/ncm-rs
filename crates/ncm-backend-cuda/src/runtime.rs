use candle_core::{Device, Tensor};
use ncm_core::{Error, Result, WeightBackend};

/// Explicit native device selection. CPU is a diagnostic path, never a fallback.
#[derive(Clone)]
pub struct CandleBackend {
    device: Device,
}
impl CandleBackend {
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
        }
    }
    /// Set a 128 MiB CUDA pool release threshold to reuse memory across synchronization.
    /// This is not a preallocation, hard allocation limit, or guaranteed cache bound.
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Self::cuda_with_pool_retention(ordinal, 128 * 1024 * 1024)
    }
    /// Configure the current process's per-device default CUDA pool. Other CUDA
    /// libraries sharing that pool see this setting; callers must coordinate policy.
    /// Use zero for CUDA's eager-release default. No system-wide setting is changed.
    pub fn cuda_with_pool_retention(ordinal: usize, retained_bytes: u64) -> Result<Self> {
        let device = Device::new_cuda(ordinal).map_err(|e| Error::Backend(e.to_string()))?;
        #[cfg(feature = "cuda")]
        configure_pool(&device, retained_bytes)?;
        #[cfg(not(feature = "cuda"))]
        let _ = retained_bytes; // Device::new_cuda above fails explicitly without CUDA.
        Ok(Self { device })
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
}
/// Safety: initialized CUDA context is bound on this thread; CUDA documents the
/// release-threshold attribute as u64. The pointer lives for the synchronous call.
/// The default pool handle remains CUDA-owned and is never destroyed here.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn configure_pool(device: &Device, retained_bytes: u64) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let cuda = device
        .as_cuda_device()
        .map_err(|e| Error::Backend(e.to_string()))?;
    let stream = cuda.cuda_stream();
    let context = stream.context();
    context
        .bind_to_thread()
        .map_err(|e| Error::Backend(e.to_string()))?;
    let mut pool = std::ptr::null_mut();
    let mut bytes = retained_bytes;
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, context.cu_device())
            .result()
            .map_err(|e| Error::Backend(e.to_string()))?;
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut bytes as *mut u64).cast(),
        )
        .result()
        .map_err(|e| Error::Backend(e.to_string()))?;
    }
    Ok(())
}

impl WeightBackend for CandleBackend {
    type Tensor = Tensor;
    fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Tensor> {
        Tensor::from_slice(values, shape, &self.device).map_err(|e| Error::Backend(e.to_string()))
    }
}
