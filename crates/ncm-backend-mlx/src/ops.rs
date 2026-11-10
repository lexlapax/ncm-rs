//! FP32 compositions, BLC activations, explicit upstream weight-layout conversion.
use mlx_rs::{
    Array,
    ops::{
        self,
        indexing::{Ellipsis, IndexOp},
    },
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn epsilon(epsilon: f64) -> Result<Array> {
    if !epsilon.is_finite() || epsilon <= 0. || epsilon > f32::MAX as f64 {
        return Err("invalid epsilon".into());
    }
    Ok(Array::from_f32(epsilon as f32))
}
pub fn layer_norm(x: &Array, eps: f64) -> Result<Array> {
    let centered = x.subtract(x.mean_axis(-1, true)?)?;
    Ok(centered.multiply(
        centered
            .square()?
            .mean_axis(-1, true)?
            .add(epsilon(eps)?)?
            .rsqrt()?,
    )?)
}
pub fn rms_norm(x: &Array, eps: f64) -> Result<Array> {
    Ok(x.multiply(
        x.square()?
            .mean_axis(-1, true)?
            .add(epsilon(eps)?)?
            .rsqrt()?,
    )?)
}
pub fn group_norm(x: &Array, groups: usize, eps: f64) -> Result<Array> {
    let &[b, l, c] = x.shape() else {
        return Err("GroupNorm requires BLC".into());
    };
    let g = i32::try_from(groups)?;
    if g <= 0 || c % g != 0 {
        return Err("invalid normalization groups".into());
    }
    let flat =
        x.swap_axes(1, 2)?
            .reshape(&[b, g, (c / g).checked_mul(l).ok_or("shape overflow")?])?;
    Ok(layer_norm(&flat, eps)?
        .reshape(&[b, c, l])?
        .swap_axes(1, 2)?)
}
pub fn conv1d(x: &Array, w: &Array, stride: usize, padding: usize) -> Result<Array> {
    if stride == 0 {
        return Err("zero stride".into());
    }
    Ok(ops::conv1d(
        x,
        w.transpose_axes(&[0, 2, 1])?,
        i32::try_from(stride)?,
        i32::try_from(padding)?,
        1,
        1,
    )?)
}
pub fn conv_transpose1d(x: &Array, w: &Array, stride: usize, padding: usize) -> Result<Array> {
    if stride == 0 {
        return Err("zero stride".into());
    }
    // Upstream [in,out,k] -> MLX [out,k,in].
    Ok(ops::conv_transpose1d(
        x,
        w.transpose_axes(&[1, 2, 0])?,
        i32::try_from(stride)?,
        i32::try_from(padding)?,
        1,
        0,
        1,
    )?)
}
fn lengths(x: &Array, length: usize) -> Result<(usize, i32)> {
    if x.shape().len() != 3 || x.shape()[1] <= 0 || length == 0 || length > 1_000_000 {
        return Err("invalid BLC resampling shape/length".into());
    }
    Ok((length, x.shape()[1]))
}
pub fn area1d(x: &Array, length: usize) -> Result<Array> {
    let (length, source) = lengths(x, length)?;
    let source = source as usize;
    let mut bins = Vec::with_capacity(length);
    for i in 0..length {
        let start = i * source / length;
        let end = ((i + 1) * source).div_ceil(length);
        bins.push(
            x.try_index((.., start as i32..end as i32, ..))?
                .mean_axis(1, true)?,
        );
    }
    Ok(ops::concatenate(&bins, 1)?)
}
pub fn linear1d(x: &Array, length: usize) -> Result<Array> {
    let (length, source) = lengths(x, length)?;
    let mut bins = Vec::with_capacity(length);
    for i in 0..length {
        let position = ((i as f64 + 0.5) * f64::from(source) / length as f64 - 0.5).max(0.);
        let lo = (position.floor() as i32).min(source - 1);
        let hi = (lo + 1).min(source - 1);
        let weight = position - lo as f64;
        let a = x
            .try_index((.., lo..lo + 1, ..))?
            .multiply(Array::from_f32((1. - weight) as f32))?;
        let b = x
            .try_index((.., hi..hi + 1, ..))?
            .multiply(Array::from_f32(weight as f32))?;
        bins.push(a.add(b)?);
    }
    Ok(ops::concatenate(&bins, 1)?)
}
/// Additive mask; caller must reject rows with no allowed keys before dispatch.
pub fn attention(q: &Array, k: &Array, v: &Array, mask: &Array) -> Result<Array> {
    if q.shape().len() != 4 || q.shape()[3] <= 0 {
        return Err("attention requires BHLD".into());
    }
    let scores = q
        .matmul(k.swap_axes(-1, -2)?)?
        .multiply(Array::from_f32(1. / (q.shape()[3] as f32).sqrt()))?
        .add(mask)?;
    Ok(ops::softmax_axis(&scores, -1, true)?.matmul(v)?)
}
pub fn rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let width = *x.shape().last().ok_or("empty RoPE shape")?;
    if width <= 0 || width % 2 != 0 {
        return Err("RoPE needs even positive head width".into());
    }
    let first = x.try_index((Ellipsis, ..width / 2))?;
    let second = x.try_index((Ellipsis, width / 2..))?.negative()?;
    Ok(x.multiply(cos)?
        .add(ops::concatenate(&[second, first], -1)?.multiply(sin)?)?)
}
