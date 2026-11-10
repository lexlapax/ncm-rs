use std::collections::{BTreeMap, BTreeSet};

use candle_core::{DType, Device, Tensor};
use ncm_backend_cuda::ops;
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const ATOL: f64 = 1e-5;
pub const RTOL: f64 = 1e-4;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub schema_version: u32,
    pub upstream_commit: String,
    pub torch_version: String,
    pub dtype: String,
    pub cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub name: String,
    pub op: Operation,
    pub inputs: BTreeMap<String, TensorData>,
    pub expected: TensorData,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Matmul,
    Conv1d { stride: usize, padding: usize },
    ConvTranspose1d { stride: usize, padding: usize },
    GroupNorm { groups: usize, epsilon: f64 },
    LayerNorm { epsilon: f64 },
    RmsNorm { epsilon: f64 },
    Area1d { length: usize },
    Linear1d { length: usize },
    Attention,
    Rope,
    F16Roundtrip,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl TensorData {
    pub fn validate(&self) -> Result<()> {
        if self.shape.is_empty() || self.shape.len() > 4 || self.shape.contains(&0) {
            return Err("fixture tensor rank/shape is invalid".into());
        }
        let size = self.shape.iter().try_fold(1usize, |size, dimension| {
            size.checked_mul(*dimension).ok_or("shape overflow")
        })?;
        if size > 1_000_000 || size != self.data.len() || self.data.iter().any(|x| !x.is_finite()) {
            return Err("fixture tensor size or values are invalid".into());
        }
        Ok(())
    }
}

impl Fixture {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 16 * 1024 * 1024 {
            return Err("fixture exceeds 16 MiB limit".into());
        }
        let fixture: Self = serde_json::from_slice(bytes)?;
        if fixture.schema_version != 1 || fixture.dtype != "float32" || fixture.cases.is_empty() {
            return Err("unsupported fixture schema/dtype or empty suite".into());
        }
        if fixture.upstream_commit != "c70274273f489abc73832b2b98584a8483b6b203" {
            return Err("fixture upstream pin changed without qualification".into());
        }
        let mut names = BTreeSet::new();
        for case in &fixture.cases {
            if !names.insert(&case.name) {
                return Err("duplicate fixture case".into());
            }
            case.expected.validate()?;
            for input in case.inputs.values() {
                input.validate()?;
            }
        }
        Ok(fixture)
    }
}

pub struct Prepared<'a> {
    case: &'a Case,
    tensors: BTreeMap<String, Tensor>,
}

impl<'a> Prepared<'a> {
    /// Upload all inputs before timing. Mask fixtures encode allowed keys as 1.
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
        let tensor = match self.case.op {
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
        };
        Ok(tensor)
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorMetrics {
    pub passed: bool,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub mismatches: usize,
    pub elements: usize,
}

/// Compare the complete tensor; never let NaNs pass a tolerance comparison.
pub fn compare(actual: &[f32], expected: &[f32]) -> Result<ErrorMetrics> {
    if actual.is_empty() || actual.len() != expected.len() {
        return Err("empty or mismatched output length".into());
    }
    let mut max: f64 = 0.0;
    let mut total = 0.0;
    let mut mismatches = 0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("nonfinite operator output".into());
        }
        let error = (f64::from(actual) - f64::from(expected)).abs();
        max = max.max(error);
        total += error;
        mismatches += usize::from(error > ATOL + RTOL * f64::from(expected).abs());
    }
    Ok(ErrorMetrics {
        passed: mismatches == 0,
        max_abs_error: max,
        mean_abs_error: total / actual.len() as f64,
        mismatches,
        elements: actual.len(),
    })
}

pub fn check_output(actual: &Tensor, expected: &TensorData) -> Result<ErrorMetrics> {
    if actual.dims() != expected.shape {
        return Err("output shape differs from reference".into());
    }
    compare(&actual.flatten_all()?.to_vec1::<f32>()?, &expected.data)
}
