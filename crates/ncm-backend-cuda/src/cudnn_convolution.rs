//! Explicit upstream-layout cuDNN convolution with strict FP32 by default.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::{
        cudnn::{
            safe::{ConvBackwardData, ConvForward, Cudnn},
            sys,
        },
        driver::{CudaSlice, CudaStream},
    },
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Shape, Tensor};

// A model request may hold a pool reservation in addition to this workspace.
// Reject oversize heuristic selections explicitly rather than changing algorithms.
const MAX_WORKSPACE_BYTES: usize = 128 * 1024 * 1024;
const MAX_CACHED_SHAPES: usize = 16;
type Key = (usize, usize, usize, usize, usize, usize, usize, bool);
#[derive(Clone, Copy)]
enum Algorithm {
    Forward(sys::cudnnConvolutionFwdAlgo_t, usize),
    Backward(sys::cudnnConvolutionBwdDataAlgo_t, usize),
}
/// Serialized handle ownership. cuDNN permits a handle to move between host
/// threads when calls are not simultaneous. The mutex enclosing this value is
/// held throughout descriptor creation and launch, and the CUDA context is
/// rebound on every call. See NVIDIA's cuDNN developer guide, "Thread Safety".
struct SerializedHandle {
    handle: Option<std::sync::Arc<Cudnn>>,
    stream: std::sync::Arc<CudaStream>,
}
// SAFETY: access to the non-Send cuDNN handle is exclusively guarded by its
// owning mutex; no handle or descriptors escape that lock, each call rebinds
// the CUDA context, and Drop rebinds it before destroying the last handle.
#[allow(unsafe_code)]
unsafe impl Send for SerializedHandle {}
impl Drop for SerializedHandle {
    fn drop(&mut self) {
        self.stream
            .context()
            .bind_to_thread()
            .expect("bind CUDA context before cuDNN handle destruction");
        drop(self.handle.take());
    }
}
/// Model-owned, bounded host metadata and one serialized native handle.
#[derive(Default)]
pub(crate) struct AlgorithmCache {
    #[cfg(feature = "lm1b-experimental")]
    pub(crate) allow_default_math: bool,
    #[cfg(feature = "lm1b-experimental")]
    pub(crate) graph_forward: Option<crate::cudnn_graph::ForwardCache>,
    entries: std::sync::Mutex<std::collections::VecDeque<(Key, Algorithm)>>,
    handle: std::sync::Mutex<Option<SerializedHandle>>,
}
impl AlgorithmCache {
    fn get_or_create(
        &self,
        key: Key,
        create: impl FnOnce() -> candle_core::Result<Algorithm>,
    ) -> candle_core::Result<Algorithm> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| candle_core::Error::Msg("cuDNN algorithm cache poisoned".into()))?;
        if let Some(i) = entries.iter().position(|(existing, _)| *existing == key) {
            let entry = entries.remove(i).expect("located cache entry");
            let result = entry.1;
            entries.push_back(entry);
            return Ok(result);
        }
        let result = create()?;
        if entries.len() == MAX_CACHED_SHAPES {
            entries.pop_front();
        }
        entries.push_back((key, result));
        Ok(result)
    }
}

struct Conv<'a> {
    cache: &'a AlgorithmCache,
    stride: usize,
    padding: usize,
    transposed: bool,
}
impl CustomOp2 for Conv<'_> {
    fn name(&self) -> &'static str {
        "ncm-cudnn-fp32-convolution"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("strict cuDNN convolution requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let [batch, input_channels, input_len] = xl.dims() else {
            candle_core::bail!("invalid convolution input rank")
        };
        let [a, b, kernel] = wl.dims() else {
            candle_core::bail!("invalid convolution weight rank")
        };
        let (weight_input, output_channels) = if self.transposed { (*a, *b) } else { (*b, *a) };
        if !(1..=128).contains(batch)
            || !(1..=1024).contains(input_channels)
            || !(1..=1024).contains(&output_channels)
            || !(1..=1024).contains(input_len)
            || !((1..=16).contains(kernel) && (1..=16).contains(&self.stride)
                || (*kernel == 128 && self.stride == 128 && self.padding == 0))
            || self.padding > 16
            || *input_channels != weight_input
        {
            candle_core::bail!("unsupported strict cuDNN convolution dimensions")
        }
        let twice_pad = self
            .padding
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::Msg("padding overflow".into()))?;
        let output_len = if self.transposed {
            (input_len - 1)
                .checked_mul(self.stride)
                .and_then(|n| n.checked_add(*kernel))
                .and_then(|n| n.checked_sub(twice_pad))
        } else {
            input_len
                .checked_add(twice_pad)
                .and_then(|n| n.checked_sub(*kernel))
                .map(|n| n / self.stride + 1)
        }
        .filter(|&n| (1..=4096).contains(&n))
        .ok_or_else(|| candle_core::Error::Msg("invalid output length".into()))?;
        let input_count = batch
            .checked_mul(*input_channels)
            .and_then(|n| n.checked_mul(*input_len))
            .ok_or_else(|| candle_core::Error::Msg("input overflow".into()))?;
        let weight_count = a
            .checked_mul(*b)
            .and_then(|n| n.checked_mul(*kernel))
            .ok_or_else(|| candle_core::Error::Msg("weight overflow".into()))?;
        let output_count = batch
            .checked_mul(output_channels)
            .and_then(|n| n.checked_mul(output_len))
            .filter(|&n| n <= 128 * 1024 * 1024)
            .ok_or_else(|| candle_core::Error::Msg("output overflow or budget exceeded".into()))?;
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("input must be contiguous".into()))?;
        let (w0, w1) = wl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("weights must be contiguous".into()))?;
        let xs = x.as_cuda_slice::<f32>()?;
        let ws = w.as_cuda_slice::<f32>()?;
        if x1.checked_sub(x0) != Some(input_count)
            || w1.checked_sub(w0) != Some(weight_count)
            || x1 > xs.len()
            || w1 > ws.len()
        {
            candle_core::bail!("convolution storage extent mismatch")
        }
        let xs = xs.slice(x0..x1);
        let ws = ws.slice(w0..w1);
        let device = &x.device;
        let i32v = |n: usize| i32::try_from(n).map_err(candle_core::Error::wrap);
        let stream = device.cuda_stream();
        #[cfg(feature = "lm1b-experimental")]
        if let Some(cache) = &self.cache.graph_forward
            && !self.transposed
            && [
                *input_channels,
                output_channels,
                *kernel,
                self.stride,
                self.padding,
            ] == [384, 384, 3, 1, 1]
        {
            // SAFETY: the validated device and size above match the descriptor.
            let mut out =
                unsafe { device.alloc::<f32>(output_count) }.map_err(candle_core::Error::wrap)?;
            cache.execute(
                &stream,
                &xs,
                &ws,
                &mut out,
                *batch as i64,
                *input_len as i64,
            )?;
            return Ok((
                CudaStorage {
                    slice: CudaStorageSlice::F32(out),
                    device: device.clone(),
                },
                vec![*batch, output_channels, output_len].into(),
            ));
        }
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        let mut handle_guard = self
            .cache
            .handle
            .lock()
            .map_err(|_| candle_core::Error::Msg("cuDNN handle lock poisoned".into()))?;
        if handle_guard.is_none() {
            *handle_guard = Some(SerializedHandle {
                handle: Some(Cudnn::new(stream.clone()).map_err(candle_core::Error::wrap)?),
                stream,
            });
        }
        let handle = handle_guard
            .as_ref()
            .and_then(|h| h.handle.as_ref())
            .ok_or_else(|| candle_core::Error::Msg("missing cuDNN handle".into()))?;
        let mut desc = handle
            .create_conv2d::<f32>(
                [0, i32v(self.padding)?],
                [1, i32v(self.stride)?],
                [1, 1],
                sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .map_err(candle_core::Error::wrap)?;
        let math_type = sys::cudnnMathType_t::CUDNN_FMA_MATH;
        #[cfg(feature = "lm1b-experimental")]
        let math_type = if self.cache.allow_default_math {
            sys::cudnnMathType_t::CUDNN_DEFAULT_MATH
        } else {
            math_type
        };
        desc.set_math_type(math_type)
            .map_err(candle_core::Error::wrap)?;
        let format = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        let src = handle
            .create_4d_tensor::<f32>(
                format,
                [i32v(*batch)?, i32v(*input_channels)?, 1, i32v(*input_len)?],
            )
            .map_err(candle_core::Error::wrap)?;
        let dst = handle
            .create_4d_tensor::<f32>(
                format,
                [i32v(*batch)?, i32v(output_channels)?, 1, i32v(output_len)?],
            )
            .map_err(candle_core::Error::wrap)?;
        let filter = handle
            .create_4d_filter::<f32>(format, [i32v(*a)?, i32v(*b)?, 1, i32v(*kernel)?])
            .map_err(candle_core::Error::wrap)?;
        // SAFETY: output length and dtype were bounded and validated above.
        let mut out =
            unsafe { device.alloc::<f32>(output_count) }.map_err(candle_core::Error::wrap)?;
        let key = (
            *batch,
            *input_channels,
            *input_len,
            output_channels,
            *kernel,
            self.stride,
            self.padding,
            self.transposed,
        );
        if self.transposed {
            let op = ConvBackwardData {
                conv: &desc,
                dx: &dst,
                w: &filter,
                dy: &src,
            };
            let choice = self.cache.get_or_create(key, || {
                let algo = op.pick_algorithm().map_err(candle_core::Error::wrap)?;
                let size = op
                    .get_workspace_size(algo)
                    .map_err(candle_core::Error::wrap)?;
                if size > MAX_WORKSPACE_BYTES {
                    candle_core::bail!("strict cuDNN convolution workspace exceeds limit")
                }
                Ok(Algorithm::Backward(algo, size))
            })?;
            let Algorithm::Backward(algo, size) = choice else {
                candle_core::bail!("incorrect cuDNN cache entry")
            };
            let mut workspace = device
                .cuda_stream()
                .alloc_zeros::<u8>(size)
                .map_err(candle_core::Error::wrap)?;
            // SAFETY: FP32 descriptors, checked storage extents and workspace match the selected operation.
            unsafe {
                op.launch::<CudaSlice<u8>, _, _, _>(
                    algo,
                    Some(&mut workspace),
                    (1.0, 0.0),
                    &mut out,
                    &ws,
                    &xs,
                )
            }
            .map_err(candle_core::Error::wrap)?;
        } else {
            let op = ConvForward {
                conv: &desc,
                x: &src,
                w: &filter,
                y: &dst,
            };
            let choice = self.cache.get_or_create(key, || {
                let algo = op.pick_algorithm().map_err(candle_core::Error::wrap)?;
                let size = op
                    .get_workspace_size(algo)
                    .map_err(candle_core::Error::wrap)?;
                if size > MAX_WORKSPACE_BYTES {
                    candle_core::bail!("strict cuDNN convolution workspace exceeds limit")
                }
                Ok(Algorithm::Forward(algo, size))
            })?;
            let Algorithm::Forward(algo, size) = choice else {
                candle_core::bail!("incorrect cuDNN cache entry")
            };
            let mut workspace = device
                .cuda_stream()
                .alloc_zeros::<u8>(size)
                .map_err(candle_core::Error::wrap)?;
            // SAFETY: FP32 descriptors, checked storage extents and workspace match the selected operation.
            unsafe {
                op.launch::<CudaSlice<u8>, _, _, _>(
                    algo,
                    Some(&mut workspace),
                    (1.0, 0.0),
                    &xs,
                    &ws,
                    &mut out,
                )
            }
            .map_err(candle_core::Error::wrap)?;
        }
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: device.clone(),
            },
            vec![*batch, output_channels, output_len].into(),
        ))
    }
}

pub(crate) fn convolve(
    cache: &AlgorithmCache,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
    transposed: bool,
) -> candle_core::Result<Tensor> {
    let input = input.transpose(1, 2)?.contiguous()?;
    let weight = weight.contiguous()?;
    input
        .apply_op2_no_bwd(
            &weight,
            &Conv {
                cache,
                stride,
                padding,
                transposed,
            },
        )?
        .transpose(1, 2)?
        .broadcast_add(bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    #[test]
    fn algorithm_cache_is_bounded_and_does_not_cache_errors() {
        let cache = AlgorithmCache::default();
        let calls = Cell::new(0);
        let key = |n| (1, 384, n, 384, 3, 1, 1, false);
        let make = || {
            calls.set(calls.get() + 1);
            Ok(Algorithm::Forward(
                sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                0,
            ))
        };
        cache.get_or_create(key(1), make).unwrap();
        cache.get_or_create(key(1), make).unwrap();
        assert_eq!(calls.get(), 1);
        for n in 2..=17 {
            cache.get_or_create(key(n), make).unwrap();
        }
        assert_eq!(cache.entries.lock().unwrap().len(), MAX_CACHED_SHAPES);
        cache.get_or_create(key(1), make).unwrap();
        assert_eq!(calls.get(), 18);
        for _ in 0..2 {
            assert!(
                cache
                    .get_or_create(key(100), || Err(candle_core::Error::Msg(
                        "intentional".into()
                    )))
                    .is_err()
            );
        }
        assert!(
            !cache
                .entries
                .lock()
                .unwrap()
                .iter()
                .any(|(k, _)| *k == key(100))
        );
    }
}
