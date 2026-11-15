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
    fn project_bias(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
    ) -> Result<Self::Tensor> {
        self.add(&self.project(x, weight)?, bias)
    }
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

/// Additional operations needed by the eager generator. Implementations retain
/// the same FP32, device and immutable-value guarantees as `CodecBackend`.
pub trait GeneratorBackend: CodecBackend {
    /// Experimental 4096-way categorical draw with one supplied random word
    /// per row. Each full 32-bit word represents a uniform in [0,1). Invalid rows must
    /// return u32::MAX, which the caller must reject before returning results.
    /// Backends without a native implementation fail explicitly.
    fn categorical_4096(
        &self,
        _logits: &Self::Tensor,
        _uniforms: &Self::Indices,
    ) -> Result<Self::Indices> {
        Err(Error::Backend(
            "native categorical sampling is unsupported".into(),
        ))
    }
    /// Embed sampled 4096-way codes without indexing out of bounds when a
    /// sampler has returned its invalid-row sentinel. The original indices
    /// must remain unchanged so the caller can reject the sentinel before
    /// returning a result. Backends must implement this explicitly.
    fn sampled_embedding_4096(
        &self,
        _weight: &Self::Tensor,
        _indices: &Self::Indices,
    ) -> Result<Self::Tensor> {
        Err(Error::Backend(
            "safe sampled embedding is unsupported".into(),
        ))
    }
    /// Copy B×S×(3HD) projected Q/K/V to contiguous 3×B×H×S×D.
    fn pack_qkv(
        &self,
        x: &Self::Tensor,
        batch: usize,
        seq: usize,
        heads: usize,
        head: usize,
    ) -> Result<Self::Tensor> {
        let width = heads
            .checked_mul(head)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| Error::Input("QKV dimensions overflow".into()))?;
        if batch == 0 || seq == 0 || heads == 0 || head == 0 || self.shape(x) != [batch, seq, width]
        {
            return Err(Error::Input("invalid QKV shape".into()));
        }
        self.reshape(
            &self.permute(
                &self.reshape(x, &[batch, seq, 3, heads, head])?,
                &[2, 0, 3, 1, 4],
            )?,
            &[3, batch, heads, seq, head],
        )
    }
    /// Split projected Q/K/V into three B×H×S×D arrays. Backends may use
    /// equivalent views when that avoids a packed temporary.
    fn split_qkv(
        &self,
        x: &Self::Tensor,
        batch: usize,
        seq: usize,
        heads: usize,
        head: usize,
    ) -> Result<[Self::Tensor; 3]> {
        let packed = self.pack_qkv(x, batch, seq, heads, head)?;
        let shape = [batch, heads, seq, head];
        Ok([
            self.reshape(&self.narrow(&packed, 0, 0, 1)?, &shape)?,
            self.reshape(&self.narrow(&packed, 0, 1, 1)?, &shape)?,
            self.reshape(&self.narrow(&packed, 0, 2, 1)?, &shape)?,
        ])
    }
    /// Copy B×H×S×D attention output to contiguous B×S×(HD).
    fn merge_heads(
        &self,
        x: &Self::Tensor,
        batch: usize,
        seq: usize,
        heads: usize,
        head: usize,
    ) -> Result<Self::Tensor> {
        let width = heads
            .checked_mul(head)
            .ok_or_else(|| Error::Input("attention dimensions overflow".into()))?;
        if batch == 0
            || seq == 0
            || heads == 0
            || head == 0
            || self.shape(x) != [batch, heads, seq, head]
        {
            return Err(Error::Input("invalid attention shape".into()));
        }
        self.reshape(&self.permute(x, &[0, 2, 1, 3])?, &[batch, seq, width])
    }
    fn concat_f32(&self, xs: &[Self::Tensor], axis: usize) -> Result<Self::Tensor>;
    /// Prepare immutable model-owned rotary tables for repeated attention blocks.
    /// The default retains the ordinary half-width cosine and sine arrays.
    fn prepare_rope_tables(
        &self,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
    ) -> Result<[Self::Tensor; 2]> {
        Ok([cos.clone(), sin.clone()])
    }
    /// Apply tables returned by this backend's `prepare_rope_tables`.
    /// The default uses the ordinary split-half operation.
    fn rope_prepared(
        &self,
        x: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
    ) -> Result<Self::Tensor> {
        self.rope(x, cos, sin)
    }
    /// Split-half rotary embedding with explicit [sequence, head_width/2]
    /// cosine/sine arrays; positions are supplied by the model.
    fn rope(
        &self,
        x: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
    ) -> Result<Self::Tensor>;
}
