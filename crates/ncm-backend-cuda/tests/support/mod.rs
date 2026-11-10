#[path = "../../../../tests/support/operators.rs"]
mod contract;
use candle_core::{DType, Device, Tensor};
pub use contract::*;
use ncm_backend_cuda::ops;
use std::collections::BTreeMap;

pub struct Prepared<'a> {
    case: &'a Case,
    tensors: BTreeMap<String, Tensor>,
}
impl<'a> Prepared<'a> {
    pub fn new(case: &'a Case, device: &Device) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for (name, input) in &case.inputs {
            let data = if name == "mask" {
                if input.data.iter().any(|x| *x != 0.0 && *x != 1.0) {
                    return Err("attention mask must contain only zero/one".into());
                }
                let width = *input.shape.last().ok_or("empty mask")?;
                if input.data.chunks(width).any(|row| !row.contains(&1.0)) {
                    return Err("all-masked attention row".into());
                }
                input
                    .data
                    .iter()
                    .map(|x| if *x == 1.0 { 0.0 } else { f32::NEG_INFINITY })
                    .collect()
            } else {
                input.data.clone()
            };
            tensors.insert(
                name.clone(),
                Tensor::from_vec(data, input.shape.as_slice(), device)?,
            );
        }
        Ok(Self { case, tensors })
    }
    fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}").into())
    }
    pub fn execute(&self) -> Result<Tensor> {
        Ok(match self.case.op {
            Operation::Matmul => self.get("x")?.matmul(self.get("w")?)?,
            Operation::Conv1d { stride, padding } => self
                .get("x")?
                .transpose(1, 2)?
                .contiguous()?
                .conv1d(self.get("w")?, padding, stride, 1, 1)?
                .transpose(1, 2)?
                .contiguous()?,
            Operation::ConvTranspose1d { stride, padding } => self
                .get("x")?
                .transpose(1, 2)?
                .contiguous()?
                .conv_transpose1d(self.get("w")?, padding, 0, stride, 1, 1)?
                .transpose(1, 2)?
                .contiguous()?,
            Operation::GroupNorm { groups, epsilon } => {
                ops::group_norm(self.get("x")?, groups, epsilon)?
            }
            Operation::LayerNorm { epsilon } => ops::layer_norm(self.get("x")?, epsilon)?,
            Operation::RmsNorm { epsilon } => ops::rms_norm(self.get("x")?, epsilon)?,
            Operation::Area1d { length } => ops::area1d(self.get("x")?, length)?,
            Operation::Linear1d { length } => ops::linear1d(self.get("x")?, length)?,
            Operation::Attention => ops::attention(
                self.get("q")?,
                self.get("k")?,
                self.get("v")?,
                self.get("mask")?,
            )?,
            Operation::Rope => ops::rope(self.get("x")?, self.get("cos")?, self.get("sin")?)?,
            Operation::F16Roundtrip => self.get("x")?.to_dtype(DType::F16)?.to_dtype(DType::F32)?,
        })
    }
}
pub fn check_output(actual: &Tensor, expected: &TensorData) -> Result<ErrorMetrics> {
    if actual.dims() != expected.shape {
        return Err("output shape differs from reference".into());
    }
    compare(&actual.flatten_all()?.to_vec1::<f32>()?, &expected.data)
}
