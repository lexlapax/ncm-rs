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
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Ok(Self {
            device: Device::new_cuda(ordinal).map_err(|e| Error::Backend(e.to_string()))?,
        })
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
}
impl WeightBackend for CandleBackend {
    type Tensor = Tensor;
    fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Tensor> {
        Tensor::from_slice(values, shape, &self.device).map_err(|e| Error::Backend(e.to_string()))
    }
}
