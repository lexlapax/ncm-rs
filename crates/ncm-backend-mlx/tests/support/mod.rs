#[path = "../../../../tests/support/operators.rs"]
mod contract;
pub use contract::*;
use mlx_rs::{Array, Dtype};
use ncm_backend_mlx::ops;
use std::collections::BTreeMap;
pub struct Prepared<'a> {
    case: &'a Case,
    tensors: BTreeMap<String, Array>,
}
impl<'a> Prepared<'a> {
    pub fn new(case: &'a Case) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for (name, input) in &case.inputs {
            let shape = input
                .shape
                .iter()
                .map(|&d| i32::try_from(d))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let data = if name == "mask" {
                if input.data.iter().any(|&v| v != 0. && v != 1.) {
                    return Err("invalid mask value".into());
                }
                let width = *input.shape.last().ok_or("empty mask")?;
                if input.data.chunks(width).any(|row| !row.contains(&1.)) {
                    return Err("all-masked row".into());
                }
                input
                    .data
                    .iter()
                    .map(|&v| if v == 1. { 0. } else { f32::NEG_INFINITY })
                    .collect()
            } else {
                input.data.clone()
            };
            let array = Array::from_slice(&data, &shape);
            array.eval()?;
            tensors.insert(name.clone(), array);
        }
        Ok(Self { case, tensors })
    }
    fn get(&self, name: &str) -> Result<&Array> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("missing {name}").into())
    }
    pub fn execute(&self) -> Result<Array> {
        Ok(match self.case.op {
            Operation::Matmul => self.get("x")?.matmul(self.get("w")?)?,
            Operation::Conv1d { stride, padding } => {
                ops::conv1d(self.get("x")?, self.get("w")?, stride, padding)?
            }
            Operation::ConvTranspose1d { stride, padding } => {
                ops::conv_transpose1d(self.get("x")?, self.get("w")?, stride, padding)?
            }
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
            Operation::F16Roundtrip => {
                let half = self.get("x")?.as_dtype(Dtype::Float16)?;
                half.eval()?;
                half.as_dtype(Dtype::Float32)?
            }
        })
    }
}
pub fn check_output(actual: &Array, expected: &TensorData) -> Result<ErrorMetrics> {
    if actual
        .shape()
        .iter()
        .map(|&v| v as usize)
        .collect::<Vec<_>>()
        != expected.shape
    {
        return Err("output shape mismatch".into());
    }
    let flat = actual.reshape(&[-1])?.contiguous()?;
    flat.eval()?;
    compare(flat.try_as_slice::<f32>()?, &expected.data)
}
