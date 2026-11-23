//! Vectorized FP32 resampling. Shape-derived metadata never depends on tensor values.
//! Unlike per-position compositions, device dispatch count is independent of output length.
use candle_core::{Device, Result, Tensor};
use std::collections::BTreeMap;

fn dimensions(x: &Tensor, output: usize) -> Result<(usize, usize, usize)> {
    let (batch, input, channels) = x.dims3()?;
    if input == 0 || output == 0 || x.dtype() != candle_core::DType::F32 {
        candle_core::bail!("resampling requires nonempty FP32 input and output")
    }
    u32::try_from(input)
        .map_err(|_| candle_core::Error::Msg("resampling index exceeds u32".into()))?;
    input
        .checked_mul(output)
        .and_then(|_| output.checked_mul(16))
        .ok_or_else(|| candle_core::Error::Msg("resampling size overflow".into()))?;
    Ok((batch, input, channels))
}

#[derive(Clone)]
enum Geometry {
    Linear {
        left: Tensor,
        right: Tensor,
        weight: Tensor,
        complement: Tensor,
    },
    Area {
        indices: Tensor,
        mask: Tensor,
        divisor: Tensor,
        output: usize,
        width: usize,
    },
}
#[derive(Clone)]
pub(crate) struct Prepared {
    geometry: Geometry,
    bytes: usize,
}
impl Prepared {
    fn new(input: usize, output: usize, area: bool, device: &Device) -> Result<Self> {
        if !area {
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
            return Ok(Self {
                geometry: Geometry::Linear {
                    left: Tensor::from_vec(left, output, device)?,
                    right: Tensor::from_vec(right, output, device)?,
                    weight: Tensor::from_vec(weight, (1, output, 1), device)?,
                    complement: Tensor::from_vec(complement, (1, output, 1), device)?,
                },
                bytes: output * 16,
            });
        }
        let bins: Vec<_> = (0..output)
            .map(|i| (i * input / output, ((i + 1) * input).div_ceil(output)))
            .collect();
        let width = bins.iter().map(|(a, b)| b - a).max().unwrap();
        let count = output
            .checked_mul(width)
            .ok_or_else(|| candle_core::Error::Msg("resampling metadata size overflow".into()))?;
        let bytes = count
            .checked_mul(8)
            .and_then(|n| n.checked_add(output * 4))
            .ok_or_else(|| candle_core::Error::Msg("resampling metadata bytes overflow".into()))?;
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
        Ok(Self {
            geometry: Geometry::Area {
                indices: Tensor::from_vec(indices, count, device)?,
                mask: Tensor::from_vec(mask, (1, output, width, 1), device)?,
                divisor: Tensor::from_vec(divisor, (1, output, 1), device)?,
                output,
                width,
            },
            bytes,
        })
    }
    pub(crate) fn apply(&self, x: &Tensor) -> Result<Tensor> {
        self.apply_with_reciprocal(x, false)
    }

    #[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
    pub(crate) fn apply_lm1b_diagnostic(&self, x: &Tensor) -> Result<Tensor> {
        self.apply_with_reciprocal(x, true)
    }

    fn apply_with_reciprocal(&self, x: &Tensor, reciprocal: bool) -> Result<Tensor> {
        let x = x.contiguous()?;
        match &self.geometry {
            Geometry::Linear {
                left,
                right,
                weight,
                complement,
            } => {
                x.index_select(left, 1)?.broadcast_mul(complement)?
                    + x.index_select(right, 1)?.broadcast_mul(weight)?
            }
            Geometry::Area {
                indices,
                mask,
                divisor,
                output,
                width,
            } => {
                let (batch, _, channels) = x.dims3()?;
                let sum = x
                    .index_select(indices, 1)?
                    .reshape((batch, *output, *width, channels))?
                    .broadcast_mul(mask)?
                    .sum(2)?;
                if reciprocal {
                    sum.broadcast_mul(&divisor.recip()?)
                } else {
                    sum.broadcast_div(divisor)
                }
            }
        }
    }
}

/// Backend-owned geometry only; no activations or model weights. A backend clone
/// starts empty, so a caller retaining its original backend does not retain a
/// model's prepared buffers. Device identity is fixed by that backend's checks.
#[derive(Default)]
pub(crate) struct Cache {
    entries: BTreeMap<(bool, usize, usize), Prepared>,
    bytes: usize,
    matrix_entries: BTreeMap<(bool, usize, usize, u8), Tensor>,
    matrix_bytes: usize,
}
pub(crate) enum MatrixPlan {
    Identity,
    Gemm(Tensor),
    Oversized,
}
fn matrix_dtype_tag(dtype: candle_core::DType) -> Result<u8> {
    match dtype {
        candle_core::DType::F32 => Ok(0),
        candle_core::DType::BF16 => Ok(1),
        candle_core::DType::F16 => Ok(2),
        other => candle_core::bail!("unsupported dtype {other:?} for fixed-matrix resampling"),
    }
}
fn build_resample_matrix(
    input: usize,
    output: usize,
    area: bool,
    dtype: candle_core::DType,
    device: &Device,
) -> Result<Tensor> {
    let mut weights = vec![0.0f32; output * input];
    if !area {
        let scale = input as f32 / output as f32;
        for i in 0..output {
            let source = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
            let left = (source.floor() as usize).min(input - 1);
            let right = (left + 1).min(input - 1);
            let w = (source - left as f32).clamp(0.0, 1.0);
            weights[i * input + left] += 1.0 - w;
            weights[i * input + right] += w;
        }
    } else {
        for i in 0..output {
            let start = i * input / output;
            let end = ((i + 1) * input).div_ceil(output);
            let inv = 1.0f32 / (end - start) as f32;
            for j in start..end {
                weights[i * input + j] = inv;
            }
        }
    }
    let mat = Tensor::from_vec(weights, (output, input), device)?;
    if dtype == candle_core::DType::F32 {
        Ok(mat)
    } else {
        mat.to_dtype(dtype)
    }
}
pub(crate) fn apply_resample_matrix(matrix: &Tensor, x: &Tensor) -> Result<Tensor> {
    let (batch, input, channels) = x.dims3()?;
    let (output, mat_in) = matrix.dims2()?;
    if mat_in != input {
        candle_core::bail!("resampling matrix input mismatch: {mat_in} != {input}")
    }
    let x = x.contiguous()?;
    if batch == 1 {
        let x2d = x.reshape((input, channels))?;
        matrix.matmul(&x2d)?.reshape((1, output, channels))
    } else if x.device().is_cuda() {
        // cuBLAS gemm_strided_batched accepts batch stride 0 on LHS without copying `matrix`.
        matrix.broadcast_as((batch, output, input))?.matmul(&x)
    } else {
        matrix
            .broadcast_as((batch, output, input))?
            .contiguous()?
            .matmul(&x)
    }
}
impl Cache {
    const MAX_ENTRIES: usize = 64;
    const MAX_BYTES: usize = 1024 * 1024;
    const MAX_MATRIX_ENTRIES: usize = 64;
    const MAX_MATRIX_BYTES: usize = 4 * 1024 * 1024;
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.matrix_entries.is_empty()
    }
    pub(crate) fn prepare(
        &mut self,
        x: &Tensor,
        output: usize,
        area: bool,
    ) -> Result<Option<Prepared>> {
        let (_, input, _) = dimensions(x, output)?;
        if input == output {
            return Ok(None);
        }
        let key = (area, input, output);
        if let Some(value) = self.entries.get(&key) {
            return Ok(Some(value.clone()));
        }
        let value = Prepared::new(input, output, area, x.device())?;
        if self.entries.len() < Self::MAX_ENTRIES && value.bytes <= Self::MAX_BYTES - self.bytes {
            self.bytes += value.bytes;
            self.entries.insert(key, value.clone());
        }
        // Oversized/new shapes still execute the same native operator, uncached.
        // The budget bounds logical metadata, not allocator rounding or process RSS.
        Ok(Some(value))
    }
    pub(crate) fn prepare_matrix(
        &mut self,
        x: &Tensor,
        output: usize,
        area: bool,
    ) -> Result<MatrixPlan> {
        let (_, input, _) = x.dims3()?;
        if input == 0 || output == 0 {
            candle_core::bail!("resampling requires nonempty input and output")
        }
        if input == output {
            return Ok(MatrixPlan::Identity);
        }
        let dtype = x.dtype();
        let dtype_tag = matrix_dtype_tag(dtype)?;
        let Some(bytes) = input
            .checked_mul(output)
            .and_then(|n| n.checked_mul(dtype.size_in_bytes()))
        else {
            return Ok(MatrixPlan::Oversized);
        };
        if bytes > Self::MAX_MATRIX_BYTES {
            return Ok(MatrixPlan::Oversized);
        }
        let key = (area, input, output, dtype_tag);
        if let Some(matrix) = self.matrix_entries.get(&key) {
            return Ok(MatrixPlan::Gemm(matrix.clone()));
        }
        let matrix = build_resample_matrix(input, output, area, dtype, x.device())?;
        if self.matrix_entries.len() < Self::MAX_MATRIX_ENTRIES
            && bytes <= Self::MAX_MATRIX_BYTES - self.matrix_bytes
        {
            self.matrix_bytes += bytes;
            self.matrix_entries.insert(key, matrix.clone());
        }
        Ok(MatrixPlan::Gemm(matrix))
    }
}

/// Half-pixel linear interpolation, with FP32 scale/coordinates as in Torch FP32.
/// This standalone function is uncached; codec backends retain bounded geometry.
pub fn linear1d(x: &Tensor, output: usize) -> Result<Tensor> {
    let (_, input, _) = dimensions(x, output)?;
    if input == output {
        return x.contiguous();
    }
    Prepared::new(input, output, false, x.device())?.apply(x)
}
/// Adaptive-area bins may overlap. Padding is masked before the FP32 reduction.
pub fn area1d(x: &Tensor, output: usize) -> Result<Tensor> {
    let (_, input, _) = dimensions(x, output)?;
    if input == output {
        return x.contiguous();
    }
    Prepared::new(input, output, true, x.device())?.apply(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_is_reused_across_batches_but_separates_algorithms() {
        let mut cache = Cache::default();
        for batch in [1, 2, 4] {
            let x = Tensor::arange(0f32, (batch * 7 * 3) as f32, &Device::Cpu)
                .unwrap()
                .reshape((batch, 7, 3))
                .unwrap();
            for area in [false, true] {
                let result = cache
                    .prepare(&x, 5, area)
                    .unwrap()
                    .unwrap()
                    .apply(&x)
                    .unwrap();
                let expected = if area { area1d(&x, 5) } else { linear1d(&x, 5) }.unwrap();
                assert_eq!(
                    result.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                );
            }
            assert_eq!(cache.entries.len(), 2);
        }
    }
    #[test]
    fn admission_is_bounded_and_oversized_shapes_are_not_retained() {
        let x = Tensor::ones((1, 2, 1), candle_core::DType::F32, &Device::Cpu).unwrap();
        let mut cache = Cache::default();
        for output in 3..80 {
            cache.prepare(&x, output, false).unwrap();
        }
        assert_eq!(cache.entries.len(), Cache::MAX_ENTRIES);
        assert!(cache.bytes <= Cache::MAX_BYTES);
        let mut empty = Cache::default();
        let large = empty
            .prepare(&x, Cache::MAX_BYTES / 16 + 1, false)
            .unwrap()
            .unwrap();
        assert!(empty.entries.is_empty());
        assert_eq!(empty.bytes, 0);
        assert_eq!(
            large.apply(&x).unwrap().dims(),
            &[1, Cache::MAX_BYTES / 16 + 1, 1]
        );
        assert!(empty.prepare(&x, 0, false).is_err());
        assert!(empty.prepare(&x, usize::MAX, false).is_err());
        assert!(empty.entries.is_empty());
    }
    #[test]
    fn fixed_matrix_resampling_matches_reference_linear_and_area() {
        let mut cache = Cache::default();
        for (input, output) in [(7, 5), (5, 11), (16, 4), (4, 16), (8, 8)] {
            for batch in [1, 3] {
                let x = Tensor::arange(0f32, (batch * input * 4) as f32, &Device::Cpu)
                    .unwrap()
                    .reshape((batch, input, 4))
                    .unwrap();
                for area in [false, true] {
                    let result = match cache.prepare_matrix(&x, output, area).unwrap() {
                        MatrixPlan::Identity => x.contiguous().unwrap(),
                        MatrixPlan::Gemm(m) => apply_resample_matrix(&m, &x).unwrap(),
                        MatrixPlan::Oversized => panic!("unexpected oversized"),
                    };
                    let expected = if area {
                        area1d(&x, output)
                    } else {
                        linear1d(&x, output)
                    }
                    .unwrap();
                    let a = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let b = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let max_diff = a
                        .iter()
                        .zip(b.iter())
                        .map(|(u, v)| (u - v).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_diff <= 1e-4,
                        "area={area} {input}->{output} batch={batch}: max_diff={max_diff}"
                    );
                }
            }
        }
        assert_eq!(cache.matrix_entries.len(), 8);
    }
}
