//! Shared bounded operator fixture contract; no runtime dependency.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
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
        let size = self.shape.iter().try_fold(1usize, |size, dim| {
            size.checked_mul(*dim).ok_or("shape overflow")
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
#[derive(Debug, Serialize)]
pub struct ErrorMetrics {
    pub passed: bool,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub mismatches: usize,
    pub elements: usize,
}
pub fn compare(actual: &[f32], expected: &[f32]) -> Result<ErrorMetrics> {
    if actual.is_empty() || actual.len() != expected.len() {
        return Err("empty or mismatched output length".into());
    }
    let mut max: f64 = 0.;
    let mut total = 0.;
    let mut mismatches = 0;
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() || !e.is_finite() {
            return Err("nonfinite operator output".into());
        }
        let error = (f64::from(a) - f64::from(e)).abs();
        max = max.max(error);
        total += error;
        mismatches += usize::from(error > ATOL + RTOL * f64::from(e).abs());
    }
    Ok(ErrorMetrics {
        passed: mismatches == 0,
        max_abs_error: max,
        mean_abs_error: total / actual.len() as f64,
        mismatches,
        elements: actual.len(),
    })
}
