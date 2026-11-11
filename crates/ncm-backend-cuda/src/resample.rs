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
                x.index_select(indices, 1)?
                    .reshape((batch, *output, *width, channels))?
                    .broadcast_mul(mask)?
                    .sum(2)?
                    .broadcast_div(divisor)
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
}
impl Cache {
    const MAX_ENTRIES: usize = 64;
    const MAX_BYTES: usize = 1024 * 1024;
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
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
}
