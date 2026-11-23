//! Validated FP32 codec configuration and native backend contracts.

mod config;
mod owt;
pub use config::{
    Auxiliary, CodecConfig, FineDrop, Levels, LocalGeneratorConfig, LocalHeadKind, LocalNcmConfig,
    QuantizerConfig, RawCodecConfig, Residual,
};
pub use owt::OwtConfigOnly;

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

/// Explicit compute and weight-storage precision. `Fp32` is the default and
/// preserves exact qualified parity; `Bf16` and `Fp16` are opt-in mixed-precision
/// modes; `Q8` and `Q4` store 2D linear projection weights in compact 8-bit/4-bit
/// group-quantized device buffers (norms, reductions, and sampling remain FP32/FP64).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputePrecision {
    #[default]
    Fp32,
    Bf16,
    Fp16,
    Q8,
    Q4,
}

impl ComputePrecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Q8 | Self::Q4)
    }
}

impl std::fmt::Display for ComputePrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ComputePrecision {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "fp32" | "f32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            "fp16" | "f16" => Ok(Self::Fp16),
            "q8" | "q8_0" => Ok(Self::Q8),
            "q4" | "q4_0" => Ok(Self::Q4),
            _ => Err(Error::Input(format!(
                "unsupported precision {s:?}; expected fp32, bf16, fp16, q8, or q4"
            ))),
        }
    }
}

/// Checked model loading only needs FP32 allocation. No implicit device fallback.
/// Arrays must have immutable value semantics; Clone may share immutable storage.
/// Neither arrays nor adapters are required to be Send or Sync.
pub trait WeightBackend {
    type Tensor: Clone;
    fn precision(&self) -> ComputePrecision {
        ComputePrecision::Fp32
    }
    fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor>;
    /// Upload a 2D linear projection weight `[out_dim, in_dim]`. Quantized backends (`q8`, `q4`)
    /// pack qualifying linear weights directly into compact device storage; dense backends
    /// forward to [`Self::upload_f32`].
    fn upload_linear_weight_f32(&self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor> {
        self.upload_f32(values, shape)
    }
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
    /// Calls `f` with the tensor's FP32 values in row-major order. Backends whose evaluated
    /// memory is host-visible may lend it without copying; the default copies via `read_f32`.
    fn visit_f32<R>(&self, x: &Self::Tensor, f: impl FnOnce(&[f32]) -> R) -> Result<R> {
        Ok(f(&self.read_f32(x)?))
    }
    /// Calls `f` on consecutive blocks of at most `rows` leading-axis rows, in order, with each
    /// block's FP32 values in row-major order. Backends may materialize each block separately
    /// to bound host-visible residency; values are identical to `read_f32`.
    fn visit_f32_rows<R>(
        &self,
        x: &Self::Tensor,
        rows: usize,
        mut f: impl FnMut(&[f32]) -> R,
    ) -> Result<Vec<R>> {
        let shape = self.shape(x);
        let (Some(&leading), true) = (shape.first(), rows > 0) else {
            return Err(Error::Input(
                "row visits need a leading axis and rows".into(),
            ));
        };
        let values = self.read_f32(x)?;
        let width = values.len() / leading.max(1);
        Ok(values.chunks(rows * width.max(1)).map(&mut f).collect())
    }
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
    /// WMT14's first residual convolution. Backends may provide an explicitly
    /// selected reference-order implementation; the ordinary path is unchanged.
    fn wmt14_initial_refinement(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
    ) -> Result<Self::Tensor> {
        self.conv1d(x, weight, bias, 1, 1, false)
    }
    fn area1d(&self, x: &Self::Tensor, length: usize) -> Result<Self::Tensor>;
    fn linear1d(&self, x: &Self::Tensor, length: usize) -> Result<Self::Tensor>;
    /// Materialize lazy results and wait for device completion for measurements.
    fn evaluate(&self, xs: &[Self::Tensor]) -> Result<()>;
}

/// Additional operations needed by the eager generator. Implementations retain
/// the same FP32, device and immutable-value guarantees as `CodecBackend`.
pub trait GeneratorBackend: CodecBackend {
    /// Construct the explicitly supported backend variant for the XSum rotary
    /// reference profile. Backends must opt in; there is no device fallback.
    fn xsum_reference_backend(&self) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Backend(
            "XSum rotary reference backend is unsupported".into(),
        ))
    }
    /// Apply the pinned XSum FP64 penalties on the selected device. The output
    /// is `[batch,time]`; invalid rows use u32::MAX and must be rejected.
    /// Backends must implement this explicitly and may not silently fall back.
    fn xsum_penalty_fp64(
        &self,
        _logits: &Self::Tensor,
        _allowance: &Self::Tensor,
    ) -> Result<Self::Indices> {
        Err(Error::Backend(
            "device XSum FP64 selection is unsupported".into(),
        ))
    }
    /// Explicit LM1B FP64 penalties; invalid rows return u32::MAX.
    fn lm1b_penalty_fp64(
        &self,
        _logits: &Self::Tensor,
        _allowance: &Self::Tensor,
    ) -> Result<Self::Indices> {
        Err(Error::Backend(
            "device LM1B FP64 selection is unsupported".into(),
        ))
    }
    /// Select unique finite maxima from 4096-way rows. Ties and nonfinite rows
    /// produce u32::MAX, which must be checked before returning a result.
    fn unique_max_4096(&self, _logits: &Self::Tensor) -> Result<Self::Indices> {
        Err(Error::Backend(
            "native unique top-1 selection is unsupported".into(),
        ))
    }
    /// Select the first finite maximum using the pinned XSum CUDA reference
    /// ordering. This is an opt-in implementation detail of that reference
    /// profile, not a general guarantee about PyTorch or other backends.
    fn xsum_reference_top1_4096(&self, _logits: &Self::Tensor) -> Result<Self::Indices> {
        Err(Error::Backend(
            "XSum reference tie selection is unsupported".into(),
        ))
    }
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
    /// Experimental LM1B 8192-way FP64 inverse-CDF draw using midpoint u32
    /// uniforms. Invalid rows return u32::MAX and must be rejected by callers.
    fn categorical_8192(
        &self,
        _logits: &Self::Tensor,
        _uniforms: &Self::Indices,
    ) -> Result<Self::Indices> {
        Err(Error::Backend(
            "8192-way categorical sampling is unsupported".into(),
        ))
    }
    /// Experimental LM1B 8192-way FP32 inverse-CDF draw. `uniforms` holds one FP32 value
    /// in (0, 1) per `[batch, length]` row of `[batch, length, 8192]` logits. Returns choices
    /// that are always valid indices, plus a one-element count of rows with nonfinite logits.
    /// Those rows' choices are meaningless; callers must reject any nonzero count before
    /// using the request's outputs. The count lets callers defer that check.
    fn lm1b_categorical_8192_fp32(
        &self,
        _logits: &Self::Tensor,
        _uniforms: &Self::Tensor,
    ) -> Result<(Self::Indices, Self::Indices)> {
        Err(Error::Backend(
            "LM1B FP32 categorical sampling is unsupported".into(),
        ))
    }
    /// Experimental 8192-way FP32 top-p=1 probability calculation.
    /// Unsupported backends fail explicitly; numerical details are backend-specific.
    fn lm1b_top_p_one_probabilities(&self, _logits: &Self::Tensor) -> Result<Self::Tensor> {
        Err(Error::Backend(
            "LM1B FP32 top-p probabilities are unsupported".into(),
        ))
    }
    /// XSum-specific 4096-way categorical draw. Backends explicitly opt in;
    /// unsupported devices fail rather than silently selecting another path.
    fn xsum_categorical_4096(
        &self,
        _logits: &Self::Tensor,
        _uniforms: &Self::Indices,
    ) -> Result<Self::Indices> {
        Err(Error::Backend(
            "XSum categorical sampling is unsupported".into(),
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
    /// Embed sampled 8192-way codes without indexing out of bounds when a
    /// sampler has returned its invalid-row sentinel. Original indices remain
    /// unchanged so the caller can reject any sentinel before returning.
    fn sampled_embedding_8192(
        &self,
        _weight: &Self::Tensor,
        _indices: &Self::Indices,
    ) -> Result<Self::Tensor> {
        Err(Error::Backend(
            "safe 8192-way sampled embedding is unsupported".into(),
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
    /// Split-half rotary embedding with explicit [sequence, head_width/2]
    /// cosine/sine arrays; positions are supplied by the model.
    fn rope(
        &self,
        x: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
    ) -> Result<Self::Tensor>;
    /// Apply RMS normalization followed by split-half rotary embedding.
    fn rms_rope(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
        eps: f64,
    ) -> Result<Self::Tensor> {
        self.rope(&self.rms_norm(x, weight, eps)?, cos, sin)
    }
}
