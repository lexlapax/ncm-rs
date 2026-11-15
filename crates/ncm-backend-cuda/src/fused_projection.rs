//! FP32 projection with a fused bias epilogue on the backend-owned stream.
use candle_core::cuda_backend::{CudaStorage, cudarc::cublaslt::sys};
use candle_core::{CpuStorage, CustomOp3, Layout, Shape};
use std::sync::{Arc, Mutex};
mod bounded;
use bounded::BoundedLt;
pub(crate) type StrictLt = BoundedLt;
#[derive(Debug)]
pub(crate) struct NoFusedAlgorithm;
impl std::fmt::Display for NoFusedAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no fused projection algorithm")
    }
}
impl std::error::Error for NoFusedAlgorithm {}
pub(crate) fn unavailable(error: &candle_core::Error) -> bool {
    matches!(error, candle_core::Error::WrappedContext { wrapped, .. } if wrapped.is::<NoFusedAlgorithm>())
}
fn checked_output_elements(m: usize, n: usize) -> candle_core::Result<usize> {
    let count = m
        .checked_mul(n)
        .ok_or_else(|| candle_core::Error::Msg("output overflow".into()))?;
    count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| candle_core::Error::Msg("output byte size overflow".into()))?;
    Ok(count)
}
pub(crate) struct Fused(pub Arc<Mutex<StrictLt>>);
impl CustomOp3 for Fused {
    fn name(&self) -> &'static str {
        "fp32-projection-bias"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("projection epilogue requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        lx: &Layout,
        w: &CudaStorage,
        lw: &Layout,
        b: &CudaStorage,
        lb: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if !lx.is_contiguous() || !lw.is_contiguous() || !lb.is_contiguous() {
            candle_core::bail!("projection epilogue requires contiguous inputs")
        }
        let [n, k] = lx.dims() else {
            candle_core::bail!("invalid input")
        };
        let [m, wk] = lw.dims() else {
            candle_core::bail!("invalid weight")
        };
        if *k == 0 || *n == 0 || *m == 0 || wk != k || lb.dims() != [*m] {
            candle_core::bail!("invalid shapes")
        }
        let count = checked_output_elements(*m, *n)?;
        let xs = x.as_cuda_slice::<f32>()?;
        let ws = w.as_cuda_slice::<f32>()?;
        let bs = b.as_cuda_slice::<f32>()?;
        let xend = lx
            .start_offset()
            .checked_add(
                n.checked_mul(*k)
                    .ok_or_else(|| candle_core::Error::Msg("input overflow".into()))?,
            )
            .ok_or_else(|| candle_core::Error::Msg("input overflow".into()))?;
        let wend = lw
            .start_offset()
            .checked_add(
                m.checked_mul(*k)
                    .ok_or_else(|| candle_core::Error::Msg("weight overflow".into()))?,
            )
            .ok_or_else(|| candle_core::Error::Msg("weight overflow".into()))?;
        let bend = lb
            .start_offset()
            .checked_add(*m)
            .ok_or_else(|| candle_core::Error::Msg("bias overflow".into()))?;
        if xend > xs.len() || wend > ws.len() || bend > bs.len() {
            candle_core::bail!("storage bounds")
        }
        let xv = xs.slice(lx.start_offset()..xend);
        let wv = ws.slice(lw.start_offset()..wend);
        let bv = bs.slice(lb.start_offset()..bend);
        let lda = i64::try_from(*k).map_err(candle_core::Error::wrap)?;
        let ldc = i64::try_from(*m).map_err(candle_core::Error::wrap)?;
        let stream = x.device.cuda_stream();
        // SAFETY: checked dimensions give the exact output length; cuBLAS uses
        // beta=0 and writes all output elements before they can be observed.
        let mut out = unsafe {
            stream
                .alloc::<f32>(count)
                .map_err(candle_core::Error::wrap)?
        };
        // Shapes, lengths, offsets and FP32 types are checked above. This lock
        // serializes host use of the handle; queued workspace use is ordered on
        // its sole stream, with cudarc's event tracking retained for inputs.
        let handle = self
            .0
            .lock()
            .map_err(|_| candle_core::Error::Msg("projection lock poisoned".into()))?;
        // SAFETY: contiguous FP32 views and bounds, matrix dimensions, leading
        // dimensions and bias length were checked above; all use one stream.
        if let Err(error) = unsafe {
            handle.matmul(
                *m as u64, *n as u64, *k as u64, lda, ldc, &wv, &xv, &mut out, &bv,
            )
        } {
            if error.0 == sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED {
                return Err(candle_core::Error::WrappedContext {
                    wrapped: Box::new(NoFusedAlgorithm),
                    context: "fused projection unavailable".into(),
                });
            }
            return Err(candle_core::Error::wrap(error));
        }
        Ok((
            CudaStorage::wrap_cuda_slice(out, x.device.clone()),
            Shape::from_dims(&[*n, *m]),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_byte_size_is_bounded_before_allocation() {
        assert_eq!(checked_output_elements(3, 7).unwrap(), 21);
        assert!(checked_output_elements(usize::MAX, 2).is_err());
        assert!(checked_output_elements(usize::MAX / 2, 1).is_err());
    }
    #[test]
    fn fallback_only_for_unavailable_algorithm() {
        let unavailable_error = candle_core::Error::WrappedContext {
            wrapped: Box::new(NoFusedAlgorithm),
            context: "unavailable".into(),
        };
        assert!(unavailable(&unavailable_error));
        assert!(!unavailable(&candle_core::Error::Msg(
            "device allocation failed".into()
        )));
    }
}
