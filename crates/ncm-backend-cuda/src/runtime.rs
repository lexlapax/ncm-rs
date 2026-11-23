use candle_core::{DType, Device, Tensor, TensorId, quantized::QMatMul};
use ncm_core::{ComputePrecision, Error, Result, WeightBackend};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub(crate) struct QuantizedWeight {
    pub(crate) shape: [usize; 2],
    pub(crate) matmul: Arc<QMatMul>,
}

/// Explicit native device selection. CPU is a diagnostic path, never a fallback.
pub struct CandleBackend {
    device: Device,
    pub(crate) precision: ComputePrecision,
    pub(crate) quantized_weights: Arc<Mutex<HashMap<TensorId, QuantizedWeight>>>,
    pub(crate) xsum_reference: bool,
    #[cfg(feature = "cuda")]
    pub(crate) lm1b_qkv_reference: bool,
    #[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
    pub(crate) lm1b_efficient_attention: bool,
    #[cfg(feature = "cuda-cudnn")]
    pub(crate) wmt14_reference: bool,
    #[cfg(feature = "cuda-cudnn")]
    pub(crate) wmt14_fast_argmax: bool,
    #[cfg(feature = "cuda-cudnn")]
    pub(crate) strict_cudnn: bool,
    #[cfg(feature = "cuda-cudnn")]
    pub(crate) cudnn_algorithms: crate::cudnn_convolution::AlgorithmCache,
    #[cfg(feature = "cuda")]
    pub(crate) fused: std::sync::Mutex<
        Option<std::sync::Arc<std::sync::Mutex<crate::fused_projection::StrictLt>>>,
    >,
    resampling: std::sync::Mutex<crate::resample::Cache>,
    #[cfg(feature = "cuda")]
    layout_kernel_supported: std::sync::OnceLock<bool>,
    #[cfg(feature = "cuda")]
    layout_functions:
        std::sync::Mutex<Option<[candle_core::cuda_backend::cudarc::driver::CudaFunction; 4]>>,
}
impl Clone for CandleBackend {
    fn clone(&self) -> Self {
        // Share the native device, not a previous owner's prepared buffers.
        #[cfg(feature = "cuda-cudnn")]
        {
            let mut cloned = Self::from_device(self.device.clone());
            cloned.precision = self.precision;
            cloned.quantized_weights = Arc::clone(&self.quantized_weights);
            cloned.strict_cudnn = self.strict_cudnn;
            #[cfg(feature = "lm1b-experimental")]
            {
                cloned.lm1b_efficient_attention = self.lm1b_efficient_attention;
                cloned.cudnn_algorithms.allow_default_math =
                    self.cudnn_algorithms.allow_default_math;
                cloned.cudnn_algorithms.graph_forward = self
                    .cudnn_algorithms
                    .graph_forward
                    .as_ref()
                    .map(|_| Default::default());
            }
            cloned.xsum_reference = self.xsum_reference;
            #[cfg(feature = "cuda")]
            {
                cloned.lm1b_qkv_reference = self.lm1b_qkv_reference;
            }
            cloned.wmt14_reference = self.wmt14_reference;
            cloned.wmt14_fast_argmax = self.wmt14_fast_argmax;
            cloned
        }
        #[cfg(not(feature = "cuda-cudnn"))]
        {
            let mut cloned = Self::from_device(self.device.clone());
            cloned.precision = self.precision;
            cloned.quantized_weights = Arc::clone(&self.quantized_weights);
            cloned.xsum_reference = self.xsum_reference;
            #[cfg(feature = "cuda")]
            {
                cloned.lm1b_qkv_reference = self.lm1b_qkv_reference;
            }
            cloned
        }
    }
}
impl CandleBackend {
    fn from_device(device: Device) -> Self {
        Self {
            device,
            precision: ComputePrecision::Fp32,
            quantized_weights: Arc::new(Mutex::new(HashMap::new())),
            xsum_reference: false,
            #[cfg(feature = "cuda")]
            lm1b_qkv_reference: false,
            #[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
            lm1b_efficient_attention: false,
            #[cfg(feature = "cuda-cudnn")]
            wmt14_reference: false,
            #[cfg(feature = "cuda-cudnn")]
            wmt14_fast_argmax: false,
            #[cfg(feature = "cuda-cudnn")]
            strict_cudnn: false,
            #[cfg(feature = "cuda-cudnn")]
            cudnn_algorithms: Default::default(),
            #[cfg(feature = "cuda")]
            fused: Default::default(),
            resampling: Default::default(),
            #[cfg(feature = "cuda")]
            layout_kernel_supported: Default::default(),
            #[cfg(feature = "cuda")]
            layout_functions: Default::default(),
        }
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn generator_layout_function(
        &self,
        index: usize,
    ) -> Result<candle_core::cuda_backend::cudarc::driver::CudaFunction> {
        let mut functions = self
            .layout_functions
            .lock()
            .map_err(|_| Error::Backend("layout function lock poisoned".into()))?;
        if functions.is_none() {
            let device = self
                .device
                .as_cuda_device()
                .map_err(|e| Error::Backend(e.to_string()))?;
            let load = |name| {
                device
                    .get_or_load_custom_func(
                        name,
                        "ncm-generator-layout-v1",
                        crate::generator_layout::PTX,
                    )
                    .map(|f| f.into_cuda_function())
                    .map_err(|e| Error::Backend(e.to_string()))
            };
            *functions = Some([
                load("ncm_pack_qkv_8x64")?,
                load("ncm_merge_heads_8x64")?,
                load("ncm_pack_qkv_12x64")?,
                load("ncm_merge_heads_12x64")?,
            ]);
        }
        Ok(functions.as_ref().expect("initialized layout functions")[index].clone())
    }
    pub fn cpu() -> Self {
        Self::from_device(Device::Cpu)
    }
    /// Set a 128 MiB CUDA pool release threshold to reuse memory across synchronization.
    /// This is not a preallocation, hard allocation limit, or guaranteed cache bound.
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Self::cuda_with_pool_retention(ordinal, 128 * 1024 * 1024)
    }
    /// Configure the current process's per-device default CUDA pool. Other CUDA
    /// libraries sharing that pool see this setting; callers must coordinate policy.
    /// Use zero for CUDA's eager-release default. No system-wide setting is changed.
    /// Explicit strict-FP32 cuDNN convolution selection. Requires native cuDNN;
    /// loading or allocation failures are returned rather than falling back.
    #[cfg(feature = "cuda-cudnn")]
    pub fn cuda_cudnn(ordinal: usize) -> Result<Self> {
        let mut backend = Self::cuda(ordinal)?;
        backend.strict_cudnn = true;
        Ok(backend)
    }
    /// Explicit diagnostic for LM1B reference-order QKV projection.
    #[cfg(feature = "cuda-cudnn")]
    pub fn cuda_lm1b_qkv_reference_experimental(ordinal: usize) -> Result<Self> {
        let mut backend = Self::cuda_cudnn(ordinal)?;
        backend.xsum_reference = true;
        backend.lm1b_qkv_reference = true;
        Ok(backend)
    }
    /// Explicit LM1B diagnostic allowing cuDNN's default convolution math.
    #[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
    pub fn cuda_lm1b_cudnn_default_math_experimental(ordinal: usize) -> Result<Self> {
        let mut backend = Self::cuda_lm1b_qkv_reference_experimental(ordinal)?;
        backend.cudnn_algorithms.allow_default_math = true;
        Ok(backend)
    }
    /// LM1B development diagnostic for cuDNN graph forward convolution.
    #[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
    pub fn cuda_lm1b_cudnn_graph_experimental(ordinal: usize) -> Result<Self> {
        let mut backend = Self::cuda_lm1b_cudnn_default_math_experimental(ordinal)?;
        backend.cudnn_algorithms.graph_forward = Some(Default::default());
        Ok(backend)
    }
    /// Experimental LM1B-small graph and efficient attention arithmetic on Ampere.
    #[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
    pub fn cuda_lm1b_efficient_experimental(ordinal: usize) -> Result<Self> {
        let mut backend = Self::cuda_lm1b_cudnn_graph_experimental(ordinal)?;
        let context = backend
            .device
            .as_cuda_device()
            .map_err(|e| Error::Backend(e.to_string()))?
            .cuda_stream()
            .context()
            .clone();
        let (major, _) = context
            .compute_capability()
            .map_err(|e| Error::Backend(e.to_string()))?;
        if major != 8 {
            return Err(Error::Backend(
                "LM1B efficient diagnostic requires Ampere CUDA".into(),
            ));
        }
        backend.lm1b_efficient_attention = true;
        Ok(backend)
    }
    /// Explicit experimental WMT14 reference-order initial reconstruction.
    #[cfg(feature = "cuda-cudnn")]
    pub fn cuda_wmt14_reference(ordinal: usize) -> Result<Self> {
        let mut backend = Self::cuda(ordinal)?;
        backend.wmt14_reference = true;
        Ok(backend)
    }
    /// Explicit experimental WMT14 reference backend with bounded final selection.
    #[cfg(feature = "cuda-cudnn")]
    pub fn cuda_wmt14_fast_argmax_experimental(ordinal: usize) -> Result<Self> {
        let mut backend = Self::cuda_wmt14_reference(ordinal)?;
        backend.wmt14_fast_argmax = true;
        Ok(backend)
    }
    /// Whether the explicitly selected backend uses strict cuDNN convolution.
    #[cfg(feature = "cuda-cudnn")]
    pub fn uses_strict_cudnn(&self) -> bool {
        #[cfg(feature = "lm1b-experimental")]
        if self.cudnn_algorithms.allow_default_math {
            return false;
        }
        self.strict_cudnn
    }

    pub fn cuda_with_pool_retention(ordinal: usize, retained_bytes: u64) -> Result<Self> {
        let device = Device::new_cuda(ordinal).map_err(|e| Error::Backend(e.to_string()))?;
        #[cfg(feature = "cuda")]
        configure_pool(&device, retained_bytes)?;
        #[cfg(not(feature = "cuda"))]
        let _ = retained_bytes; // Device::new_cuda above fails explicitly without CUDA.
        Ok(Self::from_device(device))
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// Select explicit compute and weight-storage precision. `Fp32` is the default
    /// and preserves exact qualified parity; `Bf16`, `Fp16`, `Q8`, and `Q4` require CUDA.
    pub fn with_precision(mut self, precision: ComputePrecision) -> Result<Self> {
        if precision != ComputePrecision::Fp32 && !self.device.is_cuda() {
            return Err(Error::Backend(format!(
                "{precision} precision requires a CUDA device; CPU supports fp32 only"
            )));
        }
        #[cfg(feature = "cuda")]
        if matches!(
            precision,
            ComputePrecision::Bf16 | ComputePrecision::Q8 | ComputePrecision::Q4
        ) {
            let (major, _) = self
                .device
                .as_cuda_device()
                .map_err(|e| Error::Backend(e.to_string()))?
                .cuda_stream()
                .context()
                .compute_capability()
                .map_err(|e| Error::Backend(e.to_string()))?;
            if major < 8 {
                return Err(Error::Backend(format!(
                    "{precision} precision requires Ampere (SM 8.0+) CUDA hardware"
                )));
            }
        }
        self.precision = precision;
        Ok(self)
    }
    pub(crate) fn float_dtype(&self) -> DType {
        match self.precision {
            ComputePrecision::Fp32 => DType::F32,
            ComputePrecision::Bf16 | ComputePrecision::Q8 | ComputePrecision::Q4 => DType::BF16,
            ComputePrecision::Fp16 => DType::F16,
        }
    }
    pub(crate) fn lookup_quantized(&self, weight: &Tensor) -> Option<([usize; 2], Arc<QMatMul>)> {
        if !self.precision.is_quantized() {
            return None;
        }
        let guard = self.quantized_weights.lock().ok()?;
        guard
            .get(&weight.id())
            .map(|entry| (entry.shape, Arc::clone(&entry.matmul)))
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn supports_generator_layout_kernel(&self) -> Result<bool> {
        if let Some(&supported) = self.layout_kernel_supported.get() {
            return Ok(supported);
        }
        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
        let device = self
            .device
            .as_cuda_device()
            .map_err(|e| Error::Backend(e.to_string()))?;
        let stream = device.cuda_stream();
        let context = stream.context();
        context
            .bind_to_thread()
            .map_err(|e| Error::Backend(e.to_string()))?;
        let major = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let minor = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let supported = major > 7 || (major == 7 && minor >= 5);
        let _ = self.layout_kernel_supported.set(supported);
        Ok(supported)
    }
    pub(crate) fn resample(
        &self,
        x: &Tensor,
        output: usize,
        area: bool,
    ) -> candle_core::Result<Tensor> {
        let plan = self
            .resampling
            .lock()
            .map_err(|_| candle_core::Error::Msg("resampling cache lock poisoned".into()))?
            .prepare(x, output, area)?;
        // Cache lock is released before executing the operator. Plans retain only
        // immutable geometry; activations and outputs remain request-local.
        match plan {
            #[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
            Some(plan) if area && self.cudnn_algorithms.graph_forward.is_some() => {
                plan.apply_lm1b_diagnostic(x)
            }
            Some(plan) => plan.apply(x),
            None => x.contiguous(),
        }
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
    fn precision(&self) -> ComputePrecision {
        self.precision
    }
    fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Tensor> {
        if self.precision == ComputePrecision::Fp16
            && values.iter().any(|&v| v.is_finite() && v.abs() > 65_504.0)
        {
            return Err(Error::Weights(
                "tensor contains values exceeding IEEE 754 FP16 finite range (65504.0); use --precision bf16 or --precision fp32".into(),
            ));
        }
        let tensor = Tensor::from_slice(values, shape, &self.device)
            .map_err(|e| Error::Backend(e.to_string()))?;
        if self.precision == ComputePrecision::Fp32 {
            Ok(tensor)
        } else {
            tensor
                .to_dtype(self.float_dtype())
                .map_err(|e| Error::Backend(e.to_string()))
        }
    }
    fn upload_linear_weight_f32(&self, values: &[f32], shape: &[usize]) -> Result<Tensor> {
        let ggml_dtype = match self.precision {
            ComputePrecision::Q8 => Some(candle_core::quantized::GgmlDType::Q8_0),
            ComputePrecision::Q4 => Some(candle_core::quantized::GgmlDType::Q4_0),
            _ => None,
        };
        if let (Some(qdtype), &[out_dim, in_dim]) = (ggml_dtype, shape)
            && out_dim > 0
            && in_dim > 0
            && in_dim.is_multiple_of(32)
            && !values.iter().any(|&v| !v.is_finite() || v.abs() > 65_504.0)
        {
            let cpu_t = Tensor::from_slice(values, shape, &Device::Cpu)
                .map_err(|e| Error::Backend(e.to_string()))?;
            let qtensor =
                candle_core::quantized::QTensor::quantize_onto(&cpu_t, qdtype, &self.device)
                    .map_err(|e| Error::Backend(e.to_string()))?;
            let matmul = Arc::new(
                QMatMul::from_qtensor(qtensor).map_err(|e| Error::Backend(e.to_string()))?,
            );
            // Tiny 1-element device sentinel carries the TensorId; actual weights live in QMatMul.
            let sentinel = Tensor::zeros((1,), self.float_dtype(), &self.device)
                .map_err(|e| Error::Backend(e.to_string()))?;
            self.quantized_weights
                .lock()
                .map_err(|_| Error::Backend("quantized weights lock poisoned".into()))?
                .insert(
                    sentinel.id(),
                    QuantizedWeight {
                        shape: [out_dim, in_dim],
                        matmul,
                    },
                );
            return Ok(sentinel);
        }
        self.upload_f32(values, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ncm_core::GeneratorBackend;

    #[test]
    fn xsum_reference_rejects_cpu_without_fallback() {
        let backend = CandleBackend::cpu();
        assert!(backend.xsum_reference_backend().is_err());
    }

    #[test]
    fn backend_clone_does_not_retain_prepared_geometry() {
        let backend = CandleBackend::cpu();
        let x = Tensor::ones((1, 2, 1), candle_core::DType::F32, backend.device()).unwrap();
        backend.resample(&x, 3, false).unwrap();
        assert!(!backend.resampling.lock().unwrap().is_empty());
        let clone = backend.clone();
        assert!(clone.resampling.lock().unwrap().is_empty());
        assert_eq!(clone.resample(&x, 3, false).unwrap().dims(), &[1, 3, 1]);
    }
}
