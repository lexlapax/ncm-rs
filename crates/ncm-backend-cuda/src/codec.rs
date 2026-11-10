//! FP32 native operations for the shared codec; no model-specific scheduling.
use crate::{CandleBackend, ops};
use candle_core::{D, DType, Tensor};
use ncm_core::{CodecBackend, Error, Result};

macro_rules! op {
    ($backend:expr, [$($input:expr),*], $body:block) => {{
        (|| -> candle_core::Result<_> {
            $( $backend.check_tensor($input, DType::F32)?; )*
            $body
        })().map_err(|e| Error::Backend(e.to_string()))
    }};
}
impl CandleBackend {
    fn check_tensor(&self, x: &Tensor, dtype: DType) -> candle_core::Result<()> {
        if x.dtype() != dtype || !x.device().same_device(self.device()) {
            candle_core::bail!("tensor dtype/device does not match the explicit backend")
        }
        Ok(())
    }
}
impl CodecBackend for CandleBackend {
    type Indices = Tensor;
    fn upload_indices(&self, values: &[u32], shape: &[usize]) -> Result<Tensor> {
        Tensor::from_slice(values, shape, self.device()).map_err(|e| Error::Backend(e.to_string()))
    }
    fn read_f32(&self, x: &Tensor) -> Result<Vec<f32>> {
        op!(self, [x], { x.flatten_all()?.to_vec1() })
    }
    fn read_indices(&self, x: &Tensor) -> Result<Vec<u32>> {
        op!(self, [], {
            self.check_tensor(x, DType::U32)?;
            x.flatten_all()?.to_vec1()
        })
    }
    fn shape(&self, x: &Tensor) -> Vec<usize> {
        x.dims().to_vec()
    }
    fn indices_shape(&self, x: &Tensor) -> Vec<usize> {
        x.dims().to_vec()
    }
    fn reshape(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        op!(self, [x], { x.reshape(shape) })
    }
    fn permute(&self, x: &Tensor, axes: &[usize]) -> Result<Tensor> {
        op!(self, [x], { x.permute(axes)?.contiguous() })
    }
    fn narrow(&self, x: &Tensor, axis: usize, start: usize, len: usize) -> Result<Tensor> {
        op!(self, [x], { x.narrow(axis, start, len)?.contiguous() })
    }
    fn add(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        op!(self, [x, y], { x.broadcast_add(y) })
    }
    fn sub(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        op!(self, [x, y], { x.broadcast_sub(y) })
    }
    fn scale(&self, x: &Tensor, scale: f64) -> Result<Tensor> {
        op!(self, [x], { x * scale })
    }
    fn square(&self, x: &Tensor) -> Result<Tensor> {
        op!(self, [x], { x.sqr() })
    }
    fn sum_last(&self, x: &Tensor) -> Result<Tensor> {
        op!(self, [x], { x.sum_keepdim(D::Minus1) })
    }
    fn project(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        op!(self, [x, weight], {
            let (output, input) = weight.dims2()?;
            if input == 0 || x.dim(D::Minus1)? != input {
                candle_core::bail!("invalid projection shape")
            }
            let mut shape = x.dims().to_vec();
            *shape
                .last_mut()
                .ok_or_else(|| candle_core::Error::Msg("scalar projection".into()))? = output;
            x.contiguous()?
                .reshape((x.elem_count() / input, input))?
                .matmul(&weight.t()?)?
                .reshape(shape)
        })
    }
    fn embedding(&self, weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
        op!(self, [weight], {
            self.check_tensor(ids, DType::U32)?;
            let (_, width) = weight.dims2()?;
            let mut shape = ids.dims().to_vec();
            shape.push(width);
            weight.index_select(&ids.flatten_all()?, 0)?.reshape(shape)
        })
    }
    fn select_last(&self, x: &Tensor, largest: bool) -> Result<Tensor> {
        op!(self, [x], {
            // Native CUDA argmin/argmax does not guarantee the first tied index.
            // Reduce values, then reduce masked indices, entirely on-device.
            let count = u32::try_from(x.dim(D::Minus1)?)
                .map_err(|_| candle_core::Error::Msg("selection axis exceeds u32".into()))?;
            if count == 0 {
                candle_core::bail!("empty selection axis")
            }
            let best = if largest {
                x.max_keepdim(D::Minus1)?
            } else {
                x.min_keepdim(D::Minus1)?
            };
            let matches = x.broadcast_eq(&best)?;
            let indices = Tensor::arange(0u32, count, self.device())?.broadcast_as(x.shape())?;
            let sentinel = Tensor::new(u32::MAX, self.device())?.broadcast_as(x.shape())?;
            matches.where_cond(&indices, &sentinel)?.min(D::Minus1)
        })
    }
    fn reshape_indices(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        op!(self, [], {
            self.check_tensor(x, DType::U32)?;
            x.reshape(shape)
        })
    }
    fn concat_indices(&self, xs: &[Tensor], axis: usize) -> Result<Tensor> {
        op!(self, [], {
            for x in xs {
                self.check_tensor(x, DType::U32)?;
            }
            Tensor::cat(xs, axis)
        })
    }
    fn narrow_indices(&self, x: &Tensor, axis: usize, start: usize, len: usize) -> Result<Tensor> {
        op!(self, [], {
            self.check_tensor(x, DType::U32)?;
            x.narrow(axis, start, len)?.contiguous()
        })
    }
    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Result<Tensor> {
        op!(self, [x, weight], {
            let value = ops::layer_norm(x, eps)?.broadcast_mul(weight)?;
            match bias {
                Some(b) => {
                    self.check_tensor(b, DType::F32)?;
                    value.broadcast_add(b)
                }
                None => Ok(value),
            }
        })
    }
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        op!(self, [x, weight], {
            ops::rms_norm(x, eps)?.broadcast_mul(weight)
        })
    }
    fn group_norm(&self, x: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
        op!(self, [x], { ops::group_norm(x, groups, eps) })
    }
    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        op!(self, [x], { x.gelu_erf() })
    }
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        op!(self, [q, k, v], {
            let zero;
            let mask = match mask {
                Some(m) => {
                    self.check_tensor(m, DType::F32)?;
                    m
                }
                None => {
                    zero = Tensor::zeros((1, 1, 1, 1), DType::F32, self.device())?;
                    &zero
                }
            };
            ops::attention(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, mask)
        })
    }
    fn conv1d(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        stride: usize,
        padding: usize,
        transposed: bool,
    ) -> Result<Tensor> {
        op!(self, [x, weight, bias], {
            let (batch, length, input) = x.dims3()?;
            let (a, c, kernel) = weight.dims3()?;
            let (expected_input, output) = if transposed { (a, c) } else { (c, a) };
            if stride == 0
                || batch == 0
                || length == 0
                || kernel == 0
                || output == 0
                || input != expected_input
                || bias.dims() != [output]
            {
                candle_core::bail!("invalid convolution shape or stride")
            }
            let twice_padding = padding
                .checked_mul(2)
                .ok_or_else(|| candle_core::Error::Msg("convolution padding overflow".into()))?;
            let span = if transposed {
                (length - 1)
                    .checked_mul(stride)
                    .and_then(|n| n.checked_add(kernel))
                    .and_then(|n| n.checked_sub(twice_padding))
                    .filter(|&n| n > 0)
            } else {
                length
                    .checked_add(twice_padding)
                    .and_then(|n| n.checked_sub(kernel))
            };
            if span.is_none() {
                candle_core::bail!("convolution output is empty or overflows")
            }
            let x = x.transpose(1, 2)?.contiguous()?;
            let value = if transposed && self.device().is_cpu() && x.dim(0)? > 1 {
                // Candle 0.11.1's CPU batched transpose path misindexes the
                // second batch for length > 1. Keep native CPU execution but
                // dispatch each batch separately; never move CUDA data to host.
                let batches = (0..x.dim(0)?)
                    .map(|i| {
                        x.narrow(0, i, 1)?
                            .conv_transpose1d(weight, padding, 0, stride, 1, 1)
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                Tensor::cat(&batches, 0)?
            } else if transposed {
                x.conv_transpose1d(weight, padding, 0, stride, 1, 1)?
            } else {
                x.conv1d(weight, padding, stride, 1, 1)?
            };
            value.transpose(1, 2)?.contiguous()?.broadcast_add(bias)
        })
    }
    fn area1d(&self, x: &Tensor, length: usize) -> Result<Tensor> {
        op!(self, [x], { ops::area1d(x, length) })
    }
    fn linear1d(&self, x: &Tensor, length: usize) -> Result<Tensor> {
        op!(self, [x], { ops::linear1d(x, length) })
    }
    fn evaluate(&self, xs: &[Tensor]) -> Result<()> {
        op!(self, [], {
            for x in xs {
                self.check_tensor(x, DType::F32)?;
            }
            self.device().synchronize()
        })
    }
}
