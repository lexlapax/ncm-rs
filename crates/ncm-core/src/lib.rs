//! Validated FP32 codec configuration and native backend contracts.

mod config;
pub use config::{
    Auxiliary, CodecConfig, FineDrop, Levels, QuantizerConfig, RawCodecConfig, Residual,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid or unsupported codec configuration: {0}")]
    Config(String),
    #[error("invalid codec weights: {0}")]
    Weights(String),
    #[error("invalid codec input: {0}")]
    Input(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

/// Checked model loading only needs FP32 allocation. No implicit device fallback.
/// Arrays must have immutable value semantics; Clone may share immutable storage.
/// Neither arrays nor adapters are required to be Send or Sync.
pub trait WeightBackend {
    type Tensor: Clone;
    fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor>;
}

/// Minimal codec operations. Logical sequence layout is BLC, attention is BHLD;
/// convolution weights retain PyTorch OIK (transposed: IOK) layout at this boundary.
/// All floating operations are FP32. Operations must not mutate their inputs.
/// Selection returns the FIRST index on exact ties. No approximate GELU or SDPA
/// algorithm may be substituted without passing the declared numerical gates.
pub trait CodecBackend: WeightBackend {
    type Indices: Clone;
    fn upload_indices(&self, values: &[u32], shape: &[usize]) -> Result<Self::Indices>;
    fn read_f32(&self, x: &Self::Tensor) -> Result<Vec<f32>>;
    fn read_indices(&self, x: &Self::Indices) -> Result<Vec<u32>>;
    fn shape(&self, x: &Self::Tensor) -> Vec<usize>;
    fn indices_shape(&self, x: &Self::Indices) -> Vec<usize>;
    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor>;
    fn permute(&self, x: &Self::Tensor, axes: &[usize]) -> Result<Self::Tensor>;
    fn narrow(
        &self,
        x: &Self::Tensor,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<Self::Tensor>;
    fn add(&self, x: &Self::Tensor, y: &Self::Tensor) -> Result<Self::Tensor>;
    fn sub(&self, x: &Self::Tensor, y: &Self::Tensor) -> Result<Self::Tensor>;
    fn scale(&self, x: &Self::Tensor, scale: f64) -> Result<Self::Tensor>;
    fn square(&self, x: &Self::Tensor) -> Result<Self::Tensor>;
    fn sum_last(&self, x: &Self::Tensor) -> Result<Self::Tensor>;
    /// Last-axis matrix product x @ weight.T; weight is rank two.
    fn project(&self, x: &Self::Tensor, weight: &Self::Tensor) -> Result<Self::Tensor>;
    fn embedding(&self, weight: &Self::Tensor, ids: &Self::Indices) -> Result<Self::Tensor>;
    fn select_last(&self, x: &Self::Tensor, largest: bool) -> Result<Self::Indices>;
    fn reshape_indices(&self, x: &Self::Indices, shape: &[usize]) -> Result<Self::Indices>;
    fn concat_indices(&self, xs: &[Self::Indices], axis: usize) -> Result<Self::Indices>;
    fn narrow_indices(
        &self,
        x: &Self::Indices,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<Self::Indices>;
    fn layer_norm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        eps: f64,
    ) -> Result<Self::Tensor>;
    /// Equivalent layer normalization with an optional backend-specific implementation.
    /// The default preserves the ordinary layer normalization behavior.
    fn layer_norm_accelerated(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        eps: f64,
    ) -> Result<Self::Tensor> {
        self.layer_norm(x, weight, bias, eps)
    }
    fn rms_norm(&self, x: &Self::Tensor, weight: &Self::Tensor, eps: f64) -> Result<Self::Tensor>;
    fn group_norm(&self, x: &Self::Tensor, groups: usize, eps: f64) -> Result<Self::Tensor>;
    fn gelu(&self, x: &Self::Tensor) -> Result<Self::Tensor>;
    /// Additive broadcast mask; None means bidirectional, never implicit causal.
    fn attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        mask: Option<&Self::Tensor>,
    ) -> Result<Self::Tensor>;
    fn conv1d(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
        stride: usize,
        padding: usize,
        transposed: bool,
    ) -> Result<Self::Tensor>;
    fn area1d(&self, x: &Self::Tensor, length: usize) -> Result<Self::Tensor>;
    fn linear1d(&self, x: &Self::Tensor, length: usize) -> Result<Self::Tensor>;
    /// Materialize lazy results and wait for device completion for measurements.
    fn evaluate(&self, xs: &[Self::Tensor]) -> Result<()>;
}
