//! Explicit Metal operations for the shared codec. No model-specific scheduling.
use crate::ops;
use mlx_rs::{
    Array, Device, Dtype, Stream,
    ops::indexing::{ArrayIndex, TryIndexOp},
    with_stream,
};
use ncm_core::{CodecBackend, Error, Result, WeightBackend};
use std::{cell::RefCell, rc::Rc};
// PyTorch uses an inner-dimension Sklansky scan for multiple rows. Its
// single-row CUDA path uses CUB instead; retain MLX's scan for that case.
#[cfg(feature = "lm1b-experimental")]
fn lm1b_grouped_scan(probabilities: &Array, rows: usize) -> ops::Result<Array> {
    if rows == 1 {
        return Ok(probabilities.cumsum(-1, false, true)?);
    }
    let log_rows = usize::BITS - (rows - 1).leading_zeros();
    let log_threads = ((22 - log_rows) / 2).clamp(4, 9);
    let chunk = 2_i32 << log_threads;
    let flat = probabilities.reshape(&[rows as i32, 8192])?;
    let first = flat.try_index((.., 0..1))?;
    let mut carry = mlx_rs::ops::zeros_like(&first)?;
    let mut output = Vec::new();
    for base in (0..8192).step_by(chunk as usize) {
        let first = flat.try_index((.., base..base + 1))?.add(&carry)?;
        let rest = flat.try_index((.., base + 1..base + chunk))?;
        let mut values = mlx_rs::ops::concatenate(&[&first, &rest], 1)?;
        let mut width = 1_i32;
        while width < chunk {
            let blocks = values.reshape(&[rows as i32, chunk / (2 * width), 2, width])?;
            let left = blocks.try_index((.., .., 0, ..))?;
            let right = blocks.try_index((.., .., 1, ..))?;
            let end = left.try_index((.., .., width - 1..width))?;
            let right = right.add(&end)?;
            values =
                mlx_rs::ops::concatenate(&[&left, &right], -1)?.reshape(&[rows as i32, chunk])?;
            width *= 2;
        }
        carry = values.try_index((.., chunk - 1..chunk))?;
        output.push(values);
    }
    Ok(mlx_rs::ops::concatenate(&output, 1)?.reshape(probabilities.shape())?)
}

type CompiledGelu =
    Box<dyn FnMut(&[Array]) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception>>;
#[cfg(panic = "unwind")]
fn gelu_graph(inputs: &[Array]) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
    let (x, half, sqrt_two, one) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
    Ok(vec![x.multiply(half)?.multiply(
        mlx_rs::ops::erf(x.divide(sqrt_two)?)?.add(one)?,
    )?])
}

#[cfg(panic = "unwind")]
fn gelu_multiply_graph(
    inputs: &[Array],
) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
    let (x, half, inv_sqrt_two, one) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
    Ok(vec![x.multiply(half)?.multiply(
        mlx_rs::ops::erf(x.multiply(inv_sqrt_two)?)?.add(one)?,
    )?])
}

#[cfg(panic = "unwind")]
fn guarded_compiler_init<T>(init: impl FnOnce() -> T + std::panic::UnwindSafe) -> ops::Result<T> {
    std::panic::catch_unwind(init).map_err(|_| "GELU compiler initialization failed".into())
}

fn new_gelu_compiler() -> ops::Result<CompiledGelu> {
    #[cfg(panic = "unwind")]
    {
        guarded_compiler_init(|| {
            Box::new(mlx_rs::transforms::compile::compile(gelu_graph, Some(true))) as CompiledGelu
        })
    }
    #[cfg(not(panic = "unwind"))]
    {
        Err("compiled GELU requires panic unwinding".into())
    }
}

fn new_gelu_multiply_compiler() -> ops::Result<CompiledGelu> {
    #[cfg(panic = "unwind")]
    {
        guarded_compiler_init(|| {
            Box::new(mlx_rs::transforms::compile::compile(
                gelu_multiply_graph,
                Some(true),
            )) as CompiledGelu
        })
    }
    #[cfg(not(panic = "unwind"))]
    {
        Err("compiled GELU requires panic unwinding".into())
    }
}

#[cfg(all(test, panic = "unwind"))]
mod compiler_init_tests {
    use super::guarded_compiler_init;

    #[test]
    fn compiler_initialization_panic_becomes_an_error() {
        let result = guarded_compiler_init(|| -> () { panic!("simulated compiler failure") });
        assert!(result.is_err());
        assert_eq!(guarded_compiler_init(|| 7).unwrap(), 7);
    }
}

/// Opaque immutable array: callers cannot inject foreign-device graphs or mutate storage.
#[derive(Clone)]
pub struct MlxTensor(Array);

/// One explicit GPU stream, independent of the caller's default device/stream.
/// Neither this wrapper nor its arrays impose unsafe Send/Sync implementations.
#[derive(Clone)]
pub struct MlxBackend {
    stream: Stream,
    compiled_gelu: Rc<RefCell<Option<CompiledGelu>>>,
    xsum_refined_norm_experimental: bool,
    xsum_reference_projection_experimental: bool,
    xsum_target_projection_experimental: bool,
    xsum_refinement_experimental: bool,
    xsum_norm_experimental: bool,
    xsum_upsample_experimental: bool,
    wmt14_upsample_experimental: bool,
    xsum_groupnorm_experimental: bool,
    xsum_gelu_experimental: bool,
    xsum_compiled_gelu_experimental: bool,
    xsum_head_split_experimental: bool,
    xsum_head_projection_split_experimental: bool,
    xsum_head_projection_six_experimental: bool,
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
        let backend = Self {
            stream,
            compiled_gelu: Rc::new(RefCell::new(None)),
            xsum_refined_norm_experimental: false,
            xsum_reference_projection_experimental: false,
            xsum_target_projection_experimental: false,
            xsum_refinement_experimental: false,
            xsum_norm_experimental: false,
            xsum_upsample_experimental: false,
            wmt14_upsample_experimental: false,
            xsum_groupnorm_experimental: false,
            xsum_gelu_experimental: false,
            xsum_compiled_gelu_experimental: false,
            xsum_head_split_experimental: false,
            xsum_head_projection_split_experimental: false,
            xsum_head_projection_six_experimental: false,
        };
        // Evaluate a GPU operation now: unavailable Metal must fail, never fall back.
        let one = backend.upload_f32(&[1.], &[1])?;
        backend.evaluate(&[backend.add(&one, &one)?])?;
        Ok(backend)
    }
    /// Use nonoverlapping projection layouts for the three WMT14 cascade strides.
    pub fn gpu_wmt14_upsample_experimental() -> Result<Self> {
        let mut backend = Self::gpu()?;
        backend.wmt14_upsample_experimental = true;
        Ok(backend)
    }
    /// Explicit diagnostic backend; this is not a qualified inference profile.
    pub fn gpu_xsum_norm_experimental() -> Result<Self> {
        let mut backend = Self::gpu()?;
        backend.xsum_norm_experimental = true;
        Ok(backend)
    }
    /// Explicit diagnostic backend; this is not a qualified inference profile.
    pub fn gpu_xsum_upsample_experimental() -> Result<Self> {
        let mut backend = Self::gpu()?;
        backend.xsum_upsample_experimental = true;
        Ok(backend)
    }
    /// Explicit combined diagnostic; this is not a qualified inference profile.
    pub fn gpu_xsum_combined_experimental() -> Result<Self> {
        let mut backend = Self::gpu()?;
        backend.xsum_norm_experimental = true;
        backend.xsum_upsample_experimental = true;
        Ok(backend)
    }
    /// Explicit upsampling and group-normalization diagnostic backend.
    pub fn gpu_xsum_groupnorm_experimental() -> Result<Self> {
        let mut backend = Self::gpu()?;
        backend.xsum_upsample_experimental = true;
        backend.xsum_groupnorm_experimental = true;
        Ok(backend)
    }
    /// Explicit CUDA-order GELU diagnostic combined with the XSum cascade probes.
    pub fn gpu_xsum_gelu_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_groupnorm_experimental()?;
        backend.xsum_gelu_experimental = true;
        Ok(backend)
    }
    /// Explicit compiled GELU diagnostic combined with the XSum cascade probes.
    pub fn gpu_xsum_compiled_gelu_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_groupnorm_experimental()?;
        backend.xsum_compiled_gelu_experimental = true;
        Ok(backend)
    }
    /// Explicit diagnostic combining the experimental XSum operations.
    pub fn gpu_xsum_all_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_compiled_gelu_experimental()?;
        backend.xsum_norm_experimental = true;
        Ok(backend)
    }
    /// Explicit diagnostic for two-part XSum head projections and upsampling.
    pub fn gpu_xsum_head_split_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_upsample_experimental()?;
        backend.xsum_head_split_experimental = true;
        Ok(backend)
    }
    /// Explicit diagnostic splitting both XSum head projections.
    pub fn gpu_xsum_double_head_split_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_head_split_experimental()?;
        backend.xsum_head_projection_split_experimental = true;
        Ok(backend)
    }
    /// Explicit diagnostic with six-part hidden and two-part codebook projections.
    pub fn gpu_xsum_six_head_split_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_head_split_experimental()?;
        backend.xsum_head_projection_six_experimental = true;
        Ok(backend)
    }
    /// Explicit XSum refinement layout diagnostic with the split-head candidate.
    pub fn gpu_xsum_refinement_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_head_split_experimental()?;
        backend.xsum_refinement_experimental = true;
        Ok(backend)
    }
    /// Explicit XSum target projection diagnostic with the refinement candidate.
    pub fn gpu_xsum_target_projection_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_refinement_experimental()?;
        backend.xsum_target_projection_experimental = true;
        Ok(backend)
    }
    /// Explicit XSum two-part projection with reference bias order.
    pub fn gpu_xsum_reference_projection_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_refinement_experimental()?;
        backend.xsum_reference_projection_experimental = true;
        Ok(backend)
    }
    /// Explicit refined normalization with the reference-order projection candidate.
    pub fn gpu_xsum_refined_norm_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_reference_projection_experimental()?;
        backend.xsum_norm_experimental = true;
        backend.xsum_refined_norm_experimental = true;
        Ok(backend)
    }
    /// Explicit refined normalization plus the CUDA-order GELU diagnostic.
    pub fn gpu_xsum_refined_norm_gelu_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_refined_norm_experimental()?;
        backend.xsum_gelu_experimental = true;
        Ok(backend)
    }
    /// Explicit refined normalization plus compiled CUDA-order GELU.
    pub fn gpu_xsum_refined_norm_compiled_gelu_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_refined_norm_experimental()?;
        backend.xsum_compiled_gelu_experimental = true;
        Ok(backend)
    }
    /// Explicit reference-order projection with the six-part head diagnostic.
    pub fn gpu_xsum_reference_six_head_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_reference_projection_experimental()?;
        backend.xsum_head_projection_six_experimental = true;
        Ok(backend)
    }
    /// Explicit refined normalization with the six-part head diagnostic.
    pub fn gpu_xsum_refined_six_head_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_refined_norm_experimental()?;
        backend.xsum_head_projection_six_experimental = true;
        Ok(backend)
    }
    /// Explicit refined six-part head with CUDA-order GELU diagnostic.
    pub fn gpu_xsum_refined_six_head_gelu_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_refined_six_head_experimental()?;
        backend.xsum_gelu_experimental = true;
        Ok(backend)
    }
    /// Explicit split-head diagnostic with experimental group normalization.
    pub fn gpu_xsum_head_split_groupnorm_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_groupnorm_experimental()?;
        backend.xsum_head_split_experimental = true;
        Ok(backend)
    }
    /// Explicit split-head diagnostic with experimental compiled GELU.
    pub fn gpu_xsum_head_split_gelu_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_compiled_gelu_experimental()?;
        backend.xsum_head_split_experimental = true;
        Ok(backend)
    }
    /// Explicit split-head diagnostic with all experimental operations.
    pub fn gpu_xsum_head_split_all_experimental() -> Result<Self> {
        let mut backend = Self::gpu_xsum_all_experimental()?;
        backend.xsum_head_split_experimental = true;
        Ok(backend)
    }
    /// Explicit FP32 top-p=1 probability diagnostic for finite LM1B logits.
    ///
    /// Multirow inputs use a CUDA-shaped scan over MLX FP32 probabilities;
    /// single-row inputs use the MLX scan. Neither is qualified as bit-identical
    /// to PyTorch CUDA. It does not draw
    /// a sample or alter the existing full-distribution sampler.
    #[cfg(feature = "lm1b-experimental")]
    pub fn lm1b_top_p_one_probabilities_experimental(
        &self,
        logits: &MlxTensor,
    ) -> Result<MlxTensor> {
        let &[batch, length, width] = logits.0.shape() else {
            return Err(Error::Input("expected rank-three LM1B logits".into()));
        };
        if !(1..=32).contains(&batch) || !(1..=128).contains(&length) || width != 8192 {
            return Err(Error::Input("invalid LM1B top-p logits shape".into()));
        }
        if self
            .read_f32(logits)?
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(Error::Input("nonfinite LM1B top-p logits".into()));
        }
        self.array_op(&[logits], || {
            let order = mlx_rs::ops::argsort_axis(logits.0.negative()?, -1)?;
            let sorted = logits.0.take_along_axis(&order, -1)?;
            let probabilities = mlx_rs::ops::softmax_axis(&sorted, -1, true)?;
            let cumulative = lm1b_grouped_scan(&probabilities, batch as usize * length as usize)?;
            let remove = cumulative
                .subtract(&probabilities)?
                .gt(Array::from_f32(1.0))?;
            let masked = mlx_rs::ops::select(&remove, Array::from_f32(f32::NEG_INFINITY), &sorted)?;
            let inverse = mlx_rs::ops::argsort_axis(&order, -1)?;
            let restored = masked.take_along_axis(&inverse, -1)?;
            Ok(mlx_rs::ops::softmax_axis(&restored, -1, true)?)
        })
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
    fn visit_f32<R>(&self, x: &MlxTensor, f: impl FnOnce(&[f32]) -> R) -> Result<R> {
        check(x, Dtype::Float32)?;
        let flat = with_stream(&self.stream, || -> ops::Result<_> {
            let flat = x.0.reshape(&[-1])?.contiguous()?;
            flat.eval()?;
            Ok(flat)
        })
        .map_err(error)?;
        Ok(f(flat.try_as_slice::<f32>().map_err(error)?))
    }
    fn visit_f32_rows<R>(
        &self,
        x: &MlxTensor,
        rows: usize,
        mut f: impl FnMut(&[f32]) -> R,
    ) -> Result<Vec<R>> {
        check(x, Dtype::Float32)?;
        let Some(&leading) = x.0.shape().first() else {
            return Err(Error::Input("row visits need a leading axis".into()));
        };
        if rows == 0 {
            return Err(Error::Input("row visits need rows".into()));
        }
        let leading = usize::try_from(leading).map_err(error)?;
        let mut results = Vec::with_capacity(leading.div_ceil(rows));
        for start in (0..leading).step_by(rows) {
            let block = self.slice(x, 0, start, rows.min(leading - start))?;
            // Multiplying by one materializes a separate, exactly equal buffer, so the host
            // never touches (and makes resident) the full source tensor.
            let copy = with_stream(&self.stream, || -> ops::Result<_> {
                let copy = block.0.multiply(Array::from_f32(1.0))?.reshape(&[-1])?;
                copy.eval()?;
                Ok(copy)
            })
            .map_err(error)?;
            results.push(f(copy.try_as_slice::<f32>().map_err(error)?));
        }
        Ok(results)
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
        if self.xsum_head_projection_six_experimental
            && weight.0.shape() == [384, 768]
            && matches!(x.0.shape(), [_, _, 768])
        {
            let mut value =
                self.project(&self.slice(x, 2, 0, 128)?, &self.slice(weight, 1, 0, 128)?)?;
            for start in (128..768).step_by(128) {
                let part = self.project(
                    &self.slice(x, 2, start, 128)?,
                    &self.slice(weight, 1, start, 128)?,
                )?;
                value = self.add(&value, &part)?;
            }
            return Ok(value);
        }
        if self.xsum_head_projection_split_experimental
            && weight.0.shape() == [384, 768]
            && matches!(x.0.shape(), [_, _, 768])
        {
            let left = self.project(&self.slice(x, 2, 0, 384)?, &self.slice(weight, 1, 0, 384)?)?;
            let right = self.project(
                &self.slice(x, 2, 384, 384)?,
                &self.slice(weight, 1, 384, 384)?,
            )?;
            return self.add(&left, &right);
        }
        if self.xsum_head_split_experimental
            && weight.0.shape() == [4096, 384]
            && matches!(x.0.shape(), [_, _, 384])
        {
            let left = self.project(&self.slice(x, 2, 0, 192)?, &self.slice(weight, 1, 0, 192)?)?;
            let right = self.project(
                &self.slice(x, 2, 192, 192)?,
                &self.slice(weight, 1, 192, 192)?,
            )?;
            return self.add(&left, &right);
        }
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
    fn project_bias(
        &self,
        x: &MlxTensor,
        weight: &MlxTensor,
        bias: &MlxTensor,
    ) -> Result<MlxTensor> {
        if self.xsum_reference_projection_experimental
            && weight.0.shape() == [768, 384]
            && matches!(x.0.shape(), [_, 6, 384])
        {
            let first = self.project_bias(
                &self.slice(x, 2, 0, 192)?,
                &self.slice(weight, 1, 0, 192)?,
                bias,
            )?;
            let second = self.project(
                &self.slice(x, 2, 192, 192)?,
                &self.slice(weight, 1, 192, 192)?,
            )?;
            return self.add(&first, &second);
        }
        if self.xsum_target_projection_experimental
            && weight.0.shape() == [768, 384]
            && matches!(x.0.shape(), [_, 6, 384])
        {
            let mut value =
                self.project(&self.slice(x, 2, 0, 128)?, &self.slice(weight, 1, 0, 128)?)?;
            for start in [128, 256] {
                let part = self.project(
                    &self.slice(x, 2, start, 128)?,
                    &self.slice(weight, 1, start, 128)?,
                )?;
                value = self.add(&value, &part)?;
            }
            return self.add(&value, bias);
        }
        if weight.0.shape().len() != 2 || bias.0.shape() != [weight.0.shape()[0]] {
            return self.add(&self.project(x, weight)?, bias);
        }
        self.array_op(&[x, weight, bias], || {
            let &[output, input] = weight.0.shape() else {
                return Err("projection weight must be rank two".into());
            };
            if input <= 0 || x.0.shape().last() != Some(&input) {
                return Err("invalid projection shape".into());
            }
            let mut shape = x.0.shape().to_vec();
            *shape.last_mut().ok_or("scalar projection")? = output;
            Ok(mlx_rs::ops::addmm(
                &bias.0,
                x.0.reshape(&[-1, input])?,
                weight.0.transpose_axes(&[1, 0])?,
                None,
                None,
            )?
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
            if self.xsum_norm_experimental && width == Some(768) && eps == 1e-6 {
                let bias = bias.ok_or("experimental XSum normalization requires bias")?;
                return crate::xsum_norm_experimental::apply(
                    &x.0,
                    &weight.0,
                    &bias.0,
                    self.xsum_refined_norm_experimental,
                );
            }
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
        self.array_op(&[x], || {
            if self.xsum_groupnorm_experimental
                && matches!(x.0.shape(), [_, 12, 384])
                && groups == 8
                && eps == 1e-5
            {
                return crate::xsum_groupnorm_experimental::apply(&x.0);
            }
            ops::group_norm(&x.0, groups, eps)
        })
    }
    fn gelu(&self, x: &MlxTensor) -> Result<MlxTensor> {
        self.array_op(&[x], || {
            if self.xsum_compiled_gelu_experimental {
                let mut slot = self
                    .compiled_gelu
                    .try_borrow_mut()
                    .map_err(|_| "GELU compiler reentered")?;
                if slot.is_none() {
                    *slot = Some(new_gelu_multiply_compiler()?);
                }
                let compiled = slot.as_mut().ok_or("GELU compiler unavailable")?;
                let inputs = [
                    x.0.clone(),
                    Array::from_f32(0.5),
                    Array::from_f32(std::f32::consts::FRAC_1_SQRT_2),
                    Array::from_f32(1.),
                ];
                let mut result = compiled(&inputs)?;
                if result.len() != 1 {
                    return Err("invalid compiled GELU result".into());
                }
                return Ok(result.remove(0));
            }
            if self.xsum_gelu_experimental {
                return Ok(x.0.multiply(Array::from_f32(0.5))?.multiply(
                    mlx_rs::ops::erf(
                        x.0.multiply(Array::from_f32(std::f32::consts::FRAC_1_SQRT_2))?,
                    )?
                    .add(Array::from_f32(1.))?,
                )?);
            }
            if matches!(x.0.shape(), [batch, _, 3072] if *batch >= 16) {
                let mut slot = self
                    .compiled_gelu
                    .try_borrow_mut()
                    .map_err(|_| "GELU compiler reentered")?;
                if slot.is_none() {
                    *slot = Some(new_gelu_compiler()?);
                }
                let compiled = slot.as_mut().ok_or("GELU compiler unavailable")?;
                let inputs = [
                    x.0.clone(),
                    Array::from_f32(0.5),
                    Array::from_f32(std::f32::consts::SQRT_2),
                    Array::from_f32(1.),
                ];
                let mut result = compiled(&inputs)?;
                if result.len() != 1 {
                    return Err("invalid compiled GELU result".into());
                }
                return Ok(result.remove(0));
            }
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
            let projected_upsample =
                (self.xsum_upsample_experimental && length == 4 && kernel == 3)
                    || (self.wmt14_upsample_experimental
                        && [1, 4, 16].contains(&length)
                        && kernel == 4);
            let y = if projected_upsample
                && transposed
                && input == 384
                && output == 384
                && stride == kernel as usize
                && padding == 0
            {
                x.0.matmul(&weight.0.reshape(&[input, output * kernel])?)?
                    .reshape(&[batch, length, output, kernel])?
                    .transpose_axes(&[0, 1, 3, 2])?
                    .reshape(&[batch, length * kernel, output])?
            } else if self.xsum_refinement_experimental
                && !transposed
                && length == 48
                && input == 384
                && output == 384
                && kernel == 3
                && stride == 1
                && padding == 1
            {
                let zeros = Array::zeros::<f32>(&[batch, 1, input])?;
                let padded = mlx_rs::ops::concatenate(&[&zeros, &x.0, &zeros], 1)?;
                let windows = [
                    padded.try_index((.., 0..48, ..))?,
                    padded.try_index((.., 1..49, ..))?,
                    padded.try_index((.., 2..50, ..))?,
                ];
                let rows = batch.checked_mul(48).ok_or("refinement rows overflow")?;
                mlx_rs::ops::stack(&windows, -1)?
                    .reshape(&[rows, 1152])?
                    .matmul(&weight.0.reshape(&[384, 1152])?.transpose_axes(&[1, 0])?)?
                    .reshape(&[batch, 48, 384])?
            } else if transposed {
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

impl ncm_core::GeneratorBackend for MlxBackend {
    #[cfg(feature = "lm1b-experimental")]
    fn lm1b_top_p_one_probabilities(&self, logits: &Self::Tensor) -> Result<Self::Tensor> {
        self.lm1b_top_p_one_probabilities_experimental(logits)
    }

    fn lm1b_categorical_8192_fp32(
        &self,
        logits: &MlxTensor,
        uniforms: &MlxTensor,
    ) -> Result<(MlxTensor, MlxTensor)> {
        check(logits, Dtype::Float32)?;
        check(uniforms, Dtype::Float32)?;
        let &[batch, length, width] = logits.0.shape() else {
            return Err(Error::Input("expected rank-three LM1B logits".into()));
        };
        if !(1..=32).contains(&batch)
            || !(1..=128).contains(&length)
            || width != 8192
            || uniforms.0.shape() != [batch, length]
        {
            return Err(Error::Input("invalid LM1B categorical shapes".into()));
        }
        with_stream(&self.stream, || -> ops::Result<(Array, Array)> {
            let probabilities = mlx_rs::ops::softmax_axis(&logits.0, -1, true)?;
            let cumulative = probabilities.cumsum(-1, false, true)?;
            let total = cumulative.try_index((.., .., 8191..8192))?;
            let threshold = uniforms.0.expand_dims(-1)?.multiply(&total)?;
            // First index whose cumulative probability exceeds the threshold.
            let index = cumulative
                .le(&threshold)?
                .as_type::<u32>()?
                .sum_axis(-1, false)?;
            let index = mlx_rs::ops::minimum(&index, Array::from_slice(&[8191_u32], &[]))?;
            let invalid = logits.0.is_finite()?.logical_not()?.any_axis(-1, false)?;
            let choices = mlx_rs::ops::select(&invalid, Array::from_slice(&[0_u32], &[]), &index)?;
            let count = invalid.as_type::<u32>()?.sum(false)?.reshape(&[1])?;
            Ok((choices, count))
        })
        .map(|(choices, count)| (MlxTensor(choices), MlxTensor(count)))
        .map_err(error)
    }

    fn xsum_reference_backend(&self) -> Result<Self> {
        Ok(self.clone())
    }

    fn unique_max_4096(&self, logits: &MlxTensor) -> Result<MlxTensor> {
        check(logits, Dtype::Float32)?;
        with_stream(&self.stream, || crate::xsum_top1::select(&logits.0, true))
            .map(MlxTensor)
            .map_err(error)
    }

    fn xsum_reference_top1_4096(&self, logits: &MlxTensor) -> Result<MlxTensor> {
        check(logits, Dtype::Float32)?;
        with_stream(&self.stream, || crate::xsum_top1::select(&logits.0, false))
            .map(MlxTensor)
            .map_err(error)
    }

    fn rms_rope(
        &self,
        x: &MlxTensor,
        weight: &MlxTensor,
        cos: &MlxTensor,
        sin: &MlxTensor,
        eps: f64,
    ) -> Result<MlxTensor> {
        if matches!(x.0.shape(), [batch, _, _, 64] if *batch >= 16) {
            return self.array_op(&[x, weight, cos, sin], || {
                crate::qk_rope::apply(&x.0, &weight.0, &cos.0, &sin.0, eps)
            });
        }
        self.rope(&self.rms_norm(x, weight, eps)?, cos, sin)
    }
    fn sampled_embedding_4096(&self, weight: &MlxTensor, indices: &MlxTensor) -> Result<MlxTensor> {
        check(weight, Dtype::Float32)?;
        check(indices, Dtype::Uint32)?;
        if weight.0.shape().len() != 2 || weight.0.shape().first() != Some(&4096) {
            return Err(Error::Input("sampled codebook must have 4096 rows".into()));
        }
        with_stream(&self.stream, || -> ops::Result<Array> {
            let safe = mlx_rs::ops::minimum(&indices.0, Array::from_slice(&[4095u32], &[1]))?;
            Ok(weight.0.take_axis(&safe, 0)?)
        })
        .map(MlxTensor)
        .map_err(error)
    }
    fn xsum_categorical_4096(&self, logits: &MlxTensor, uniforms: &MlxTensor) -> Result<MlxTensor> {
        check(logits, Dtype::Float32)?;
        check(uniforms, Dtype::Uint32)?;
        with_stream(&self.stream, || {
            crate::xsum_categorical::draw(&logits.0, &uniforms.0)
        })
        .map(MlxTensor)
        .map_err(error)
    }

    fn categorical_4096(&self, logits: &MlxTensor, uniforms: &MlxTensor) -> Result<MlxTensor> {
        check(logits, Dtype::Float32)?;
        check(uniforms, Dtype::Uint32)?;
        with_stream(&self.stream, || {
            crate::categorical::draw(&logits.0, &uniforms.0)
        })
        .map(MlxTensor)
        .map_err(error)
    }

    fn split_qkv(
        &self,
        x: &MlxTensor,
        batch: usize,
        seq: usize,
        heads: usize,
        head: usize,
    ) -> Result<[MlxTensor; 3]> {
        let width = heads
            .checked_mul(head)
            .ok_or_else(|| Error::Input("QKV width overflow".into()))?;
        let total = width
            .checked_mul(3)
            .ok_or_else(|| Error::Input("QKV width overflow".into()))?;
        if batch == 0 || seq == 0 || heads == 0 || head == 0 || self.shape(x) != [batch, seq, total]
        {
            return Err(Error::Input("invalid QKV shape".into()));
        }
        // Large batches can retain substantially more MLX scratch when each
        // view is materialized separately; retain the packed path there.
        if batch >= 16 {
            let packed = self.pack_qkv(x, batch, seq, heads, head)?;
            let shape = [batch, heads, seq, head];
            return Ok([
                self.reshape(&self.narrow(&packed, 0, 0, 1)?, &shape)?,
                self.reshape(&self.narrow(&packed, 0, 1, 1)?, &shape)?,
                self.reshape(&self.narrow(&packed, 0, 2, 1)?, &shape)?,
            ]);
        }
        let part = |i| {
            let values = self.narrow(x, 2, i * width, width)?;
            let values = self.reshape(&values, &[batch, seq, heads, head])?;
            self.permute(&values, &[0, 2, 1, 3])
        };
        Ok([part(0)?, part(1)?, part(2)?])
    }
    fn concat_f32(&self, xs: &[MlxTensor], axis: usize) -> Result<MlxTensor> {
        let axis = i32::try_from(axis).map_err(error)?;
        let inputs: Vec<_> = xs.iter().collect();
        let arrays: Vec<_> = xs.iter().map(|x| &x.0).collect();
        self.array_op(&inputs, || Ok(mlx_rs::ops::concatenate(&arrays, axis)?))
    }
    fn rope(&self, x: &MlxTensor, cos: &MlxTensor, sin: &MlxTensor) -> Result<MlxTensor> {
        self.array_op(&[x, cos, sin], || {
            let cosine = mlx_rs::ops::concatenate(&[&cos.0, &cos.0], 1)?;
            let sine = mlx_rs::ops::concatenate(&[&sin.0, &sin.0], 1)?;
            ops::rope(&x.0, &cosine, &sine)
        })
    }
}
