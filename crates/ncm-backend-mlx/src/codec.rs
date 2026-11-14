//! Explicit Metal operations for the shared codec. No model-specific scheduling.
use crate::ops;
use mlx_rs::{
    Array, Device, Dtype, Stream,
    ops::indexing::{ArrayIndex, TryIndexOp},
    with_stream,
};
use ncm_core::{CodecBackend, Error, Result, WeightBackend};

/// Opaque immutable array: callers cannot inject foreign-device graphs or mutate storage.
#[derive(Clone)]
pub struct MlxTensor(Array);

/// One explicit GPU stream, independent of the caller's default device/stream.
/// Neither this wrapper nor its arrays impose unsafe Send/Sync implementations.
#[derive(Clone)]
pub struct MlxBackend {
    stream: Stream,
}
fn error(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}
fn dims(shape: &[usize]) -> Result<Vec<i32>> {
    shape
        .iter()
        .map(|&d| i32::try_from(d).map_err(error))
        .collect()
}
fn upload_shape(shape: &[usize], count: usize) -> Result<Vec<i32>> {
    let elements = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if shape.contains(&0) || elements != Some(count) {
        return Err(Error::Input("invalid upload shape".into()));
    }
    dims(shape)
}
fn check(x: &MlxTensor, dtype: Dtype) -> Result<()> {
    if x.0.dtype() != dtype {
        return Err(Error::Input(
            "tensor dtype does not match backend operation".into(),
        ));
    }
    Ok(())
}
impl MlxBackend {
    pub fn gpu() -> Result<Self> {
        let stream = Stream::try_default_on_device(&Device::gpu()).map_err(error)?;
        let backend = Self { stream };
        // Evaluate a GPU operation now: unavailable Metal must fail, never fall back.
        let one = backend.upload_f32(&[1.], &[1])?;
        backend.evaluate(&[backend.add(&one, &one)?])?;
        Ok(backend)
    }
    fn array_op(
        &self,
        inputs: &[&MlxTensor],
        f: impl FnOnce() -> ops::Result<Array>,
    ) -> Result<MlxTensor> {
        for x in inputs {
            check(x, Dtype::Float32)?;
        }
        with_stream(&self.stream, f).map(MlxTensor).map_err(error)
    }
    fn slice(&self, x: &MlxTensor, axis: usize, start: usize, len: usize) -> Result<MlxTensor> {
        let shape = x.0.shape();
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::Input("slice overflow".into()))?;
        if len == 0 || axis >= shape.len() || end > shape[axis] as usize {
            return Err(Error::Input("slice out of bounds".into()));
        }
        let mut indices: Vec<_> = shape.iter().map(|_| (..).index_op()).collect();
        indices[axis] =
            (i32::try_from(start).map_err(error)?..i32::try_from(end).map_err(error)?).index_op();
        self.array_op(&[], || Ok(x.0.try_index(indices.as_slice())?))
    }
}
impl WeightBackend for MlxBackend {
    type Tensor = MlxTensor;
    fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<MlxTensor> {
        let shape = upload_shape(shape, values.len())?;
        self.array_op(&[], || Ok(Array::from_slice(values, &shape)))
    }
}
impl CodecBackend for MlxBackend {
    type Indices = MlxTensor;
    fn upload_indices(&self, values: &[u32], shape: &[usize]) -> Result<MlxTensor> {
        let shape = upload_shape(shape, values.len())?;
        self.array_op(&[], || Ok(Array::from_slice(values, &shape)))
    }
    fn read_f32(&self, x: &MlxTensor) -> Result<Vec<f32>> {
        check(x, Dtype::Float32)?;
        with_stream(&self.stream, || -> ops::Result<_> {
            let flat = x.0.reshape(&[-1])?.contiguous()?;
            flat.eval()?;
            Ok(flat.try_as_slice::<f32>()?.to_vec())
        })
        .map_err(error)
    }
    fn read_indices(&self, x: &MlxTensor) -> Result<Vec<u32>> {
        check(x, Dtype::Uint32)?;
        with_stream(&self.stream, || -> ops::Result<_> {
            let flat = x.0.reshape(&[-1])?.contiguous()?;
            flat.eval()?;
            Ok(flat.try_as_slice::<u32>()?.to_vec())
        })
        .map_err(error)
    }
    fn shape(&self, x: &MlxTensor) -> Vec<usize> {
        x.0.shape().iter().map(|&d| d as usize).collect()
    }
    fn indices_shape(&self, x: &MlxTensor) -> Vec<usize> {
        self.shape(x)
    }
    fn reshape(&self, x: &MlxTensor, shape: &[usize]) -> Result<MlxTensor> {
        let shape = dims(shape)?;
        self.array_op(&[x], || Ok(x.0.reshape(&shape)?))
    }
    fn permute(&self, x: &MlxTensor, axes: &[usize]) -> Result<MlxTensor> {
        let axes = dims(axes)?;
        self.array_op(&[x], || Ok(x.0.transpose_axes(&axes)?))
    }
    fn narrow(&self, x: &MlxTensor, axis: usize, start: usize, len: usize) -> Result<MlxTensor> {
        check(x, Dtype::Float32)?;
        self.slice(x, axis, start, len)
    }
    fn add(&self, x: &MlxTensor, y: &MlxTensor) -> Result<MlxTensor> {
        self.array_op(&[x, y], || Ok(x.0.add(&y.0)?))
    }
    fn sub(&self, x: &MlxTensor, y: &MlxTensor) -> Result<MlxTensor> {
        self.array_op(&[x, y], || Ok(x.0.subtract(&y.0)?))
    }
    fn scale(&self, x: &MlxTensor, scale: f64) -> Result<MlxTensor> {
        if !scale.is_finite() || scale.abs() > f32::MAX as f64 {
            return Err(Error::Input("invalid FP32 scale".into()));
        }
        self.array_op(&[x], || Ok(x.0.multiply(Array::from_f32(scale as f32))?))
    }
    fn square(&self, x: &MlxTensor) -> Result<MlxTensor> {
        self.array_op(&[x], || Ok(x.0.square()?))
    }
    fn sum_last(&self, x: &MlxTensor) -> Result<MlxTensor> {
        self.array_op(&[x], || Ok(x.0.sum_axis(-1, true)?))
    }
    fn project(&self, x: &MlxTensor, weight: &MlxTensor) -> Result<MlxTensor> {
        self.array_op(&[x, weight], || {
            let &[output, input] = weight.0.shape() else {
                return Err("projection weight must be rank two".into());
            };
            if input <= 0 || x.0.shape().last() != Some(&input) {
                return Err("invalid projection shape".into());
            }
            let mut shape = x.0.shape().to_vec();
            *shape.last_mut().ok_or("scalar projection")? = output;
            Ok(x.0
                .reshape(&[-1, input])?
                .matmul(weight.0.transpose_axes(&[1, 0])?)?
                .reshape(&shape)?)
        })
    }
    fn embedding(&self, weight: &MlxTensor, ids: &MlxTensor) -> Result<MlxTensor> {
        check(ids, Dtype::Uint32)?;
        self.array_op(&[weight], || {
            if weight.0.shape().len() != 2 {
                return Err("embedding weight must be rank two".into());
            }
            Ok(weight.0.take_axis(&ids.0, 0)?)
        })
    }
    fn select_last(&self, x: &MlxTensor, largest: bool) -> Result<MlxTensor> {
        self.array_op(&[x], || {
            Ok(if largest {
                mlx_rs::ops::indexing::argmax_axis(&x.0, -1, false)?
            } else {
                mlx_rs::ops::indexing::argmin_axis(&x.0, -1, false)?
            })
        })
    }
    fn reshape_indices(&self, x: &MlxTensor, shape: &[usize]) -> Result<MlxTensor> {
        check(x, Dtype::Uint32)?;
        let shape = dims(shape)?;
        self.array_op(&[], || Ok(x.0.reshape(&shape)?))
    }
    fn concat_indices(&self, xs: &[MlxTensor], axis: usize) -> Result<MlxTensor> {
        for x in xs {
            check(x, Dtype::Uint32)?;
        }
        let axis = i32::try_from(axis).map_err(error)?;
        let arrays: Vec<_> = xs.iter().map(|x| &x.0).collect();
        self.array_op(&[], || Ok(mlx_rs::ops::concatenate(&arrays, axis)?))
    }
    fn narrow_indices(
        &self,
        x: &MlxTensor,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<MlxTensor> {
        check(x, Dtype::Uint32)?;
        self.slice(x, axis, start, len)
    }
    fn layer_norm(
        &self,
        x: &MlxTensor,
        weight: &MlxTensor,
        bias: Option<&MlxTensor>,
        eps: f64,
    ) -> Result<MlxTensor> {
        if let Some(b) = bias {
            check(b, Dtype::Float32)?;
        }
        self.array_op(&[x, weight], || {
            let y = ops::layer_norm(&x.0, eps)?.multiply(&weight.0)?;
            Ok(match bias {
                Some(b) => y.add(&b.0)?,
                None => y,
            })
        })
    }
    fn layer_norm_accelerated(
        &self,
        x: &MlxTensor,
        weight: &MlxTensor,
        bias: Option<&MlxTensor>,
        eps: f64,
    ) -> Result<MlxTensor> {
        if let Some(b) = bias {
            check(b, Dtype::Float32)?;
        }
        self.array_op(&[x, weight], || {
            let width = x.0.shape().last().copied();
            let affine_shape = weight.0.shape().len() == 1
                && weight.0.shape().first().copied() == width
                && bias.is_none_or(|b| b.0.shape() == weight.0.shape());
            if affine_shape {
                if !eps.is_finite() || eps <= 0. || eps > f32::MAX as f64 {
                    return Err("invalid epsilon".into());
                }
                return Ok(mlx_rs::fast::layer_norm(
                    &x.0,
                    Some(&weight.0),
                    bias.map(|b| &b.0),
                    eps as f32,
                )?);
            }
            let y = ops::layer_norm_accelerated(&x.0, eps)?.multiply(&weight.0)?;
            Ok(match bias {
                Some(b) => y.add(&b.0)?,
                None => y,
            })
        })
    }
    fn rms_norm(&self, x: &MlxTensor, weight: &MlxTensor, eps: f64) -> Result<MlxTensor> {
        self.array_op(&[x, weight], || {
            Ok(ops::rms_norm(&x.0, eps)?.multiply(&weight.0)?)
        })
    }
    fn group_norm(&self, x: &MlxTensor, groups: usize, eps: f64) -> Result<MlxTensor> {
        self.array_op(&[x], || ops::group_norm(&x.0, groups, eps))
    }
    fn gelu(&self, x: &MlxTensor) -> Result<MlxTensor> {
        self.array_op(&[x], || {
            Ok(x.0.multiply(Array::from_f32(0.5))?.multiply(
                mlx_rs::ops::erf(x.0.divide(Array::from_f32(std::f32::consts::SQRT_2))?)?
                    .add(Array::from_f32(1.))?,
            )?)
        })
    }
    fn attention(
        &self,
        q: &MlxTensor,
        k: &MlxTensor,
        v: &MlxTensor,
        mask: Option<&MlxTensor>,
    ) -> Result<MlxTensor> {
        if let Some(m) = mask {
            check(m, Dtype::Float32)?;
        }
        self.array_op(&[q, k, v], || {
            let zero = Array::from_f32(0.);
            ops::attention(&q.0, &k.0, &v.0, mask.map(|m| &m.0).unwrap_or(&zero))
        })
    }
    fn conv1d(
        &self,
        x: &MlxTensor,
        weight: &MlxTensor,
        bias: &MlxTensor,
        stride: usize,
        padding: usize,
        transposed: bool,
    ) -> Result<MlxTensor> {
        self.array_op(&[x, weight, bias], || {
            let &[batch, length, input] = x.0.shape() else {
                return Err("convolution requires BLC".into());
            };
            let &[a, c, kernel] = weight.0.shape() else {
                return Err("convolution requires rank-three weight".into());
            };
            let (expected, output) = if transposed { (a, c) } else { (c, a) };
            if batch <= 0
                || length <= 0
                || kernel <= 0
                || output <= 0
                || input != expected
                || stride == 0
                || bias.0.shape() != [output]
            {
                return Err("invalid convolution shape/stride".into());
            }
            let pad = padding.checked_mul(2).ok_or("padding overflow")?;
            let span = if transposed {
                (length as usize - 1)
                    .checked_mul(stride)
                    .and_then(|n| n.checked_add(kernel as usize))
                    .and_then(|n| n.checked_sub(pad))
                    .filter(|&n| n > 0)
            } else {
                (length as usize)
                    .checked_add(pad)
                    .and_then(|n| n.checked_sub(kernel as usize))
            };
            if span.is_none() {
                return Err("invalid convolution output size".into());
            }
            let y = if transposed {
                ops::conv_transpose1d(&x.0, &weight.0, stride, padding)?
            } else if padding == 0 && length == kernel {
                // A full-span convolution is exactly a projection. The native
                // convolution's long FP32 reduction exceeded the trained oracle
                // budget; use native FP32 matmul, not a precision/oracle change.
                let width = input
                    .checked_mul(kernel)
                    .ok_or("projection width overflow")?;
                x.0.transpose_axes(&[0, 2, 1])?
                    .reshape(&[batch, width])?
                    .matmul(
                        weight
                            .0
                            .reshape(&[output, width])?
                            .transpose_axes(&[1, 0])?,
                    )?
                    .reshape(&[batch, 1, output])?
            } else {
                ops::conv1d(&x.0, &weight.0, stride, padding)?
            };
            Ok(y.add(&bias.0)?)
        })
    }
    fn area1d(&self, x: &MlxTensor, length: usize) -> Result<MlxTensor> {
        self.array_op(&[x], || ops::area1d(&x.0, length))
    }
    fn linear1d(&self, x: &MlxTensor, length: usize) -> Result<MlxTensor> {
        self.array_op(&[x], || ops::linear1d(&x.0, length))
    }
    fn evaluate(&self, xs: &[MlxTensor]) -> Result<()> {
        for x in xs {
            check(x, Dtype::Float32)?;
        }
        // eval is synchronous; all supplied lazy roots are materialized together.
        with_stream(&self.stream, || {
            mlx_rs::transforms::eval(xs.iter().map(|x| &x.0))
        })
        .map_err(error)
    }
}
