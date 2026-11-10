//! Vectorized FP32 resampling. Shape-derived metadata never depends on tensor values.
//! Unlike per-position compositions, device dispatch count is independent of output length.
use candle_core::{Result, Tensor};

fn dimensions(x: &Tensor, output: usize) -> Result<(usize, usize, usize)> {
    let (batch, input, channels) = x.dims3()?;
    if input == 0 || output == 0 || x.dtype() != candle_core::DType::F32 {
        candle_core::bail!("resampling requires nonempty FP32 input and output")
    }
    u32::try_from(input)
        .map_err(|_| candle_core::Error::Msg("resampling index exceeds u32".into()))?;
    input
        .checked_mul(output)
        .ok_or_else(|| candle_core::Error::Msg("resampling size overflow".into()))?;
    Ok((batch, input, channels))
}

/// Half-pixel linear interpolation, with FP32 scale/coordinates as in Torch FP32.
/// No sample is evaluated on the host; only fixed indices and weights are uploaded.
pub fn linear1d(x: &Tensor, output: usize) -> Result<Tensor> {
    let (_, input, _) = dimensions(x, output)?;
    if input == output {
        return x.contiguous();
    }
    let scale = input as f32 / output as f32;
    let mut left = Vec::with_capacity(output);
    let mut right = Vec::with_capacity(output);
    let mut weight = Vec::with_capacity(output);
    let mut complement = Vec::with_capacity(output);
    for i in 0..output {
        let source = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
        let l = (source.floor() as usize).min(input - 1);
        let w = (source - l as f32).clamp(0.0, 1.0);
        left.push(l as u32);
        right.push((l + 1).min(input - 1) as u32);
        weight.push(w);
        complement.push(1.0 - w);
    }
    let device = x.device();
    let left = Tensor::from_vec(left, output, device)?;
    let right = Tensor::from_vec(right, output, device)?;
    let weight = Tensor::from_vec(weight, (1, output, 1), device)?;
    let complement = Tensor::from_vec(complement, (1, output, 1), device)?;
    let x = x.contiguous()?;
    x.index_select(&left, 1)?.broadcast_mul(&complement)?
        + x.index_select(&right, 1)?.broadcast_mul(&weight)?
}

/// Adaptive-area bins may overlap. Padding is masked before the FP32 reduction.
pub fn area1d(x: &Tensor, output: usize) -> Result<Tensor> {
    let (batch, input, channels) = dimensions(x, output)?;
    if input == output {
        return x.contiguous();
    }
    let bins: Vec<_> = (0..output)
        .map(|i| (i * input / output, ((i + 1) * input).div_ceil(output)))
        .collect();
    let width = bins.iter().map(|(a, b)| b - a).max().unwrap();
    let count = output
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::Msg("resampling metadata size overflow".into()))?;
    let mut indices = Vec::with_capacity(count);
    let mut mask = Vec::with_capacity(count);
    let mut divisor = Vec::with_capacity(output);
    for (start, end) in bins {
        divisor.push((end - start) as f32);
        for i in 0..width {
            indices.push((start + i).min(end - 1) as u32);
            mask.push(if start + i < end { 1f32 } else { 0f32 });
        }
    }
    let device = x.device();
    let indices = Tensor::from_vec(indices, count, device)?;
    let mask = Tensor::from_vec(mask, (1, output, width, 1), device)?;
    let divisor = Tensor::from_vec(divisor, (1, output, 1), device)?;
    x.contiguous()?
        .index_select(&indices, 1)?
        .reshape((batch, output, width, channels))?
        .broadcast_mul(&mask)?
        .sum(2)?
        .broadcast_div(&divisor)
}
