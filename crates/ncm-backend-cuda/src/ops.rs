//! Correctness-first compositions. No host tensor fallback or optimized-kernel claim.

use candle_core::{D, Result, Tensor};

fn valid_epsilon(epsilon: f64) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        candle_core::bail!("normalization epsilon must be positive and finite")
    }
    Ok(())
}

/// Non-affine LayerNorm over the last axis, with population variance.
pub fn layer_norm(x: &Tensor, epsilon: f64) -> Result<Tensor> {
    valid_epsilon(epsilon)?;
    let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
    let denominator = (centered.sqr()?.mean_keepdim(D::Minus1)? + epsilon)?.sqrt()?;
    centered.broadcast_div(&denominator)
}

/// Non-affine RMSNorm. Phase 1 fixtures use FP32 inputs only.
pub fn rms_norm(x: &Tensor, epsilon: f64) -> Result<Tensor> {
    valid_epsilon(epsilon)?;
    let denominator = (x.sqr()?.mean_keepdim(D::Minus1)? + epsilon)?.sqrt()?;
    x.broadcast_div(&denominator)
}

/// Non-affine GroupNorm on BLC inputs, reducing channels AND positions per group.
pub fn group_norm(x: &Tensor, groups: usize, epsilon: f64) -> Result<Tensor> {
    let (batch, length, channels) = x.dims3()?;
    if groups == 0 || channels % groups != 0 || length == 0 {
        candle_core::bail!("invalid GroupNorm groups or sequence length")
    }
    let grouped =
        x.transpose(1, 2)?
            .contiguous()?
            .reshape((batch, groups, channels / groups * length))?;
    layer_norm(&grouped, epsilon)?
        .reshape((batch, channels, length))?
        .transpose(1, 2)?
        .contiguous()
}

/// PyTorch adaptive-area semantics, including overlapping bins for uneven ratios.
pub fn area1d(x: &Tensor, output_length: usize) -> Result<Tensor> {
    let (_, input_length, _) = x.dims3()?;
    if input_length == 0 || output_length == 0 {
        candle_core::bail!("resampling requires nonempty input and output")
    }
    input_length
        .checked_mul(output_length)
        .ok_or_else(|| candle_core::Error::Msg("resampling size overflow".into()))?;
    let mut bins = Vec::with_capacity(output_length);
    for i in 0..output_length {
        let start = i * input_length / output_length;
        let end = ((i + 1) * input_length).div_ceil(output_length);
        bins.push(x.narrow(1, start, end - start)?.mean_keepdim(1)?);
    }
    Tensor::cat(&bins, 1)
}

/// PyTorch linear interpolation with `align_corners=false`, BLC inputs/outputs.
pub fn linear1d(x: &Tensor, output_length: usize) -> Result<Tensor> {
    let (_, input_length, _) = x.dims3()?;
    if input_length == 0 || output_length == 0 {
        candle_core::bail!("resampling requires nonempty input and output")
    }
    let mut bins = Vec::with_capacity(output_length);
    for i in 0..output_length {
        let source = ((i as f64 + 0.5) * input_length as f64 / output_length as f64 - 0.5).max(0.0);
        let left = (source.floor() as usize).min(input_length - 1);
        let right = (left + 1).min(input_length - 1);
        let weight = source - left as f64;
        bins.push(
            ((x.narrow(1, left, 1)? * (1.0 - weight))? + (x.narrow(1, right, 1)? * weight)?)?,
        );
    }
    Tensor::cat(&bins, 1)
}

/// Dense noncausal attention with an additive broadcastable mask, BHLD tensors.
/// Caller must ensure each query has at least one unmasked key.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (_, _, _, head_dim) = q.dims4()?;
    if head_dim == 0 {
        candle_core::bail!("attention head dimension must be nonzero")
    }
    let scores =
        (q.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?.broadcast_add(mask)?;
    let exps = scores
        .broadcast_sub(&scores.max_keepdim(D::Minus1)?)?
        .exp()?;
    let probabilities = exps.broadcast_div(&exps.sum_keepdim(D::Minus1)?)?;
    probabilities.matmul(v)
}

/// Split-half RoPE with explicit cosine/sine tensors; no implicit position offsets.
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let width = x.dim(D::Minus1)?;
    if width == 0 || width % 2 != 0 {
        candle_core::bail!("RoPE requires a nonzero even head dimension")
    }
    let first = x.narrow(D::Minus1, 0, width / 2)?;
    let second = x.narrow(D::Minus1, width / 2, width / 2)?.neg()?;
    let rotated = Tensor::cat(&[second, first], D::Minus1)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
}
