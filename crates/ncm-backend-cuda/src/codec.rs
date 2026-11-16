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
        // Preserve immutable views; consumers that require dense storage materialize it.
        op!(self, [x], { x.permute(axes) })
    }
    fn narrow(&self, x: &Tensor, axis: usize, start: usize, len: usize) -> Result<Tensor> {
        op!(self, [x], { x.narrow(axis, start, len) })
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
    fn project_bias(&self, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.device().is_cuda()
            && weight.dims().len() == 2
            && weight.dims()[0] > 0
            && x.elem_count() > 0
            && bias.dims() == [weight.dims()[0]]
            && !candle_core::cuda_backend::gemm_reduced_precision_f32()
        {
            let fused = (|| -> candle_core::Result<Tensor> {
                self.check_tensor(x, DType::F32)?;
                self.check_tensor(weight, DType::F32)?;
                self.check_tensor(bias, DType::F32)?;
                let (output, input) = weight.dims2()?;
                if input == 0 || x.dim(D::Minus1)? != input || bias.dims() != [output] {
                    candle_core::bail!("invalid projection shape")
                }
                let mut shape = x.dims().to_vec();
                *shape
                    .last_mut()
                    .ok_or_else(|| candle_core::Error::Msg("scalar projection".into()))? = output;
                let handle = {
                    let mut lock = self.fused.lock().map_err(|_| {
                        candle_core::Error::Msg("projection initialization lock poisoned".into())
                    })?;
                    if lock.is_none() {
                        let stream = self.device().as_cuda_device()?.cuda_stream();
                        let h =
                            crate::fused_projection::StrictLt::new(stream, self.xsum_reference)?;
                        *lock = Some(std::sync::Arc::new(std::sync::Mutex::new(h)));
                    }
                    lock.as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("projection handle missing".into()))?
                        .clone()
                };
                x.contiguous()?
                    .reshape((x.elem_count() / input, input))?
                    .apply_op3_no_bwd(
                        &weight.contiguous()?,
                        &bias.contiguous()?,
                        &crate::fused_projection::Fused(handle),
                    )?
                    .reshape(shape)
            })();
            match fused {
                Ok(value) => return Ok(value),
                Err(ref error) if crate::fused_projection::unavailable(error) => {}
                Err(error) => return Err(Error::Backend(error.to_string())),
            }
        }
        self.add(&self.project(x, weight)?, bias)
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
            #[cfg(feature = "cuda")]
            if self.device().is_cuda() {
                if !eps.is_finite() || eps <= 0.0 {
                    candle_core::bail!("normalization epsilon must be positive and finite")
                }
                let zero;
                let bias = match bias {
                    Some(b) => {
                        self.check_tensor(b, DType::F32)?;
                        b
                    }
                    None => {
                        zero = Tensor::zeros(weight.shape(), DType::F32, self.device())?;
                        &zero
                    }
                };
                if self.xsum_reference && x.dims().last() == Some(&768) && eps == 1e-6 {
                    return crate::xsum_layernorm::apply(self, x, weight, bias);
                }
                return candle_nn::ops::layer_norm(
                    &x.contiguous()?,
                    &weight.contiguous()?,
                    &bias.contiguous()?,
                    eps as f32,
                );
            }
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
            #[cfg(feature = "cuda")]
            if self.device().is_cuda() {
                if !eps.is_finite() || eps <= 0.0 {
                    candle_core::bail!("normalization epsilon must be positive and finite")
                }
                if self.xsum_reference && x.dims().last() == Some(&64) && eps == 1e-6 {
                    return crate::xsum_rmsnorm::apply(self, x, weight);
                }
                return candle_nn::ops::rms_norm(
                    &x.contiguous()?,
                    &weight.contiguous()?,
                    eps as f32,
                );
            }
            ops::rms_norm(x, eps)?.broadcast_mul(weight)
        })
    }
    fn group_norm(&self, x: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
        op!(self, [x], {
            #[cfg(feature = "cuda")]
            if self.xsum_reference
                && x.dims().len() == 3
                && (1..=128).contains(&x.dims()[0])
                && x.dims()[1..] == [12, 384]
                && groups == 8
                && eps == 1e-5
            {
                return crate::xsum_groupnorm::apply(self, x);
            }
            ops::group_norm(x, groups, eps)
        })
    }
    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        op!(self, [x], {
            #[cfg(feature = "cuda")]
            if self.xsum_reference {
                return crate::xsum_gelu::apply(self, x);
            }
            x.gelu_erf()
        })
    }
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        op!(self, [q, k, v], {
            #[cfg(feature = "cuda")]
            if self.device().is_cuda() {
                let (batch, heads, seq, width) = q.dims4()?;
                if width == 0 {
                    candle_core::bail!("attention width is zero")
                }
                if width == 64
                    && (1..=128).contains(&batch)
                    && (512..=768).contains(&seq)
                    && (1..=16).contains(&heads)
                    && let Some(m) = mask
                    && matches!(m.dims(), [mb,1,mq,mk] if (*mb==1||*mb==batch) && *mq==seq && *mk==seq)
                    && k.dims() == [batch, heads, seq, width]
                    && v.dims() == [batch, heads, seq, width]
                {
                    self.check_tensor(m, DType::F32)?;
                    let scores = q.contiguous()?.matmul(&k.contiguous()?.transpose(2, 3)?)?;
                    return crate::masked_softmax::apply(self, &scores, m)?
                        .matmul(&v.contiguous()?);
                }
                let scores = (q.contiguous()?.matmul(&k.contiguous()?.transpose(2, 3)?)?
                    / (width as f64).sqrt())?;
                let scores = match mask {
                    Some(m) => {
                        self.check_tensor(m, DType::F32)?;
                        scores.broadcast_add(m)?
                    }
                    None => scores,
                };
                return candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?
                    .matmul(&v.contiguous()?);
            }
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
            #[cfg(feature = "cuda-cudnn")]
            if self.strict_cudnn {
                return crate::cudnn_convolution::convolve(
                    &self.cudnn_algorithms,
                    x,
                    weight,
                    bias,
                    stride,
                    padding,
                    transposed,
                );
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
            value.transpose(1, 2)?.broadcast_add(bias)
        })
    }
    fn area1d(&self, x: &Tensor, length: usize) -> Result<Tensor> {
        op!(self, [x], { self.resample(x, length, true) })
    }
    fn linear1d(&self, x: &Tensor, length: usize) -> Result<Tensor> {
        op!(self, [x], { self.resample(x, length, false) })
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

impl ncm_core::GeneratorBackend for CandleBackend {
    fn xsum_reference_backend(&self) -> Result<Self> {
        #[cfg(feature = "cuda-cudnn")]
        if self.device().is_cuda() && self.strict_cudnn {
            let mut backend = self.clone();
            backend.xsum_reference = true;
            return Ok(backend);
        }
        Err(Error::Backend(
            "XSum rotary reference requires strict CUDA/cuDNN".into(),
        ))
    }
    fn sampled_embedding_4096(&self, weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.check_tensor(indices, DType::U32)
            .map_err(|e| Error::Backend(e.to_string()))?;
        if weight.dims().first() != Some(&4096) {
            return Err(Error::Input("sampled codebook must have 4096 rows".into()));
        }
        let safe = indices
            .minimum(4095u32)
            .map_err(|e| Error::Backend(e.to_string()))?;
        self.embedding(weight, &safe)
    }
    fn categorical_4096(&self, logits: &Tensor, uniforms: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            crate::categorical::categorical_from_uniforms(self, logits, uniforms)
                .map_err(|e| Error::Backend(e.to_string()))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (logits, uniforms);
            Err(Error::Backend(
                "native categorical sampling is unsupported".into(),
            ))
        }
    }
    fn unique_max_4096(&self, logits: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            crate::unique_top1::unique_max(self, logits).map_err(|e| Error::Backend(e.to_string()))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = logits;
            Err(Error::Backend(
                "native unique top-1 selection is unsupported".into(),
            ))
        }
    }
    fn xsum_penalty_fp64(&self, logits: &Tensor, allowance: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            crate::xsum_penalty::select(self, logits, allowance)
                .map_err(|e| Error::Backend(e.to_string()))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (logits, allowance);
            Err(Error::Backend(
                "device XSum FP64 selection is unsupported".into(),
            ))
        }
    }
    fn pack_qkv(
        &self,
        x: &Tensor,
        batch: usize,
        seq: usize,
        heads: usize,
        head: usize,
    ) -> Result<Tensor> {
        let width = heads
            .checked_mul(head)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| Error::Input("QKV dimensions overflow".into()))?;
        if batch == 0 || seq == 0 || heads == 0 || head == 0 || x.dims() != [batch, seq, width] {
            return Err(Error::Input("invalid QKV shape".into()));
        }
        #[cfg(feature = "cuda")]
        if self.device().is_cuda()
            && matches!(heads, 8 | 12)
            && head == 64
            && batch <= 128
            && seq <= 768
            && self.supports_generator_layout_kernel()?
        {
            let function = self.generator_layout_function(if heads == 8 { 0 } else { 2 })?;
            return op!(self, [x], {
                x.contiguous()?
                    .apply_op1_no_bwd(&crate::generator_layout::LayoutCopy {
                        kind: crate::generator_layout::LayoutKind::Pack,
                        heads,
                        function,
                    })
            });
        }
        op!(self, [x], {
            x.reshape((batch, seq, 3, heads, head))?
                .permute((2, 0, 3, 1, 4))?
                .reshape((3, batch, heads, seq, head))
        })
    }
    fn merge_heads(
        &self,
        x: &Tensor,
        batch: usize,
        seq: usize,
        heads: usize,
        head: usize,
    ) -> Result<Tensor> {
        let width = heads
            .checked_mul(head)
            .ok_or_else(|| Error::Input("attention dimensions overflow".into()))?;
        if batch == 0
            || seq == 0
            || heads == 0
            || head == 0
            || x.dims() != [batch, heads, seq, head]
        {
            return Err(Error::Input("invalid attention shape".into()));
        }
        #[cfg(feature = "cuda")]
        if self.device().is_cuda()
            && matches!(heads, 8 | 12)
            && head == 64
            && batch <= 128
            && seq <= 768
            && batch.checked_mul(seq).is_some_and(|n| n <= 65535)
            && self.supports_generator_layout_kernel()?
        {
            let function = self.generator_layout_function(if heads == 8 { 1 } else { 3 })?;
            return op!(self, [x], {
                x.contiguous()?
                    .apply_op1_no_bwd(&crate::generator_layout::LayoutCopy {
                        kind: crate::generator_layout::LayoutKind::Merge,
                        heads,
                        function,
                    })
            });
        }
        op!(self, [x], {
            x.permute((0, 2, 1, 3))?.reshape((batch, seq, width))
        })
    }
    fn concat_f32(&self, xs: &[Tensor], axis: usize) -> Result<Tensor> {
        op!(self, [], {
            for x in xs {
                self.check_tensor(x, DType::F32)?;
            }
            Tensor::cat(xs, axis)
        })
    }
    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        op!(self, [x, cos, sin], {
            #[cfg(feature = "cuda")]
            if self.device().is_cuda() {
                if self.xsum_reference {
                    return crate::xsum_rope::apply(self, x, cos, sin);
                }
                return candle_nn::rotary_emb::rope(
                    &x.contiguous()?,
                    &cos.contiguous()?,
                    &sin.contiguous()?,
                );
            }
            let cos = Tensor::cat(&[cos, cos], 1)?;
            let sin = Tensor::cat(&[sin, sin], 1)?;
            ops::rope(x, &cos, &sin)
        })
    }
}
