//! Narrow cuBLASLt projection wrapper with a bounded retained workspace.
//!
//! This follows cudarc 0.19.10's descriptor and stream-tracking sequence, but
//! keeps algorithm selection independent of the actual workspace allocation.
use candle_core::cuda_backend::cudarc::{
    cublaslt::{result, sys},
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
};
use std::sync::Arc;

const WORKSPACE: usize = 5 * 1024 * 1024 / 2;

// Ask for the original algorithm first, then constrain the search only when its
// returned workspace requirement cannot fit. Never launch an oversized result.
fn select_with_limit<T>(
    preference: usize,
    capacity: usize,
    mut select: impl FnMut(usize) -> Result<(usize, T), result::CublasError>,
) -> Result<T, result::CublasError> {
    let (needed, chosen) = select(preference)?;
    if needed <= capacity {
        return Ok(chosen);
    }
    let (needed, chosen) = select(capacity)?;
    if needed <= capacity {
        Ok(chosen)
    } else {
        Err(result::CublasError(
            sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED,
        ))
    }
}

fn projection_preference(default: usize, xsum: bool, lm1b: bool, m: u64, k: u64) -> usize {
    if (xsum && m == 768 && k == 768) || (lm1b && m == 2304 && k == 768) {
        1024 * 1024
    } else {
        default
    }
}

pub(crate) struct BoundedLt {
    handle: sys::cublasLtHandle_t,
    buffer: CudaSlice<u8>,
    preference: usize,
    xsum_reference: bool,
    lm1b_qkv_reference: bool,
    stream: Arc<CudaStream>,
}
// SAFETY: callers serialize handle and buffer access; the sole owning stream
// is retained, and its context is bound before every operation and destruction.
#[allow(unsafe_code)]
unsafe impl Send for BoundedLt {}

impl BoundedLt {
    #[allow(unsafe_code)]
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        xsum_reference: bool,
        lm1b_qkv_reference: bool,
    ) -> candle_core::Result<Self> {
        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        let major = stream
            .context()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(candle_core::Error::wrap)?;
        // Keep the pinned wrapper's original allocation on newer architectures,
        // whose algorithm workspace requirements have not been characterized.
        let preference = if major >= 9 {
            32 * 1024 * 1024
        } else {
            4 * 1024 * 1024
        };
        let size = if major == 8 { WORKSPACE } else { preference };
        // SAFETY: workspace is opaque storage and no uninitialized byte is read
        // by Rust. cuBLASLt owns any initialization required by its algorithms.
        let buffer = unsafe { stream.alloc::<u8>(size) }.map_err(candle_core::Error::wrap)?;
        let handle = result::create_handle().map_err(candle_core::Error::wrap)?;
        Ok(Self {
            handle,
            buffer,
            preference,
            xsum_reference,
            lm1b_qkv_reference,
            stream,
        })
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub(super) unsafe fn matmul<I: DevicePtr<f32>, O: DevicePtrMut<f32>>(
        &self,
        m: u64,
        n: u64,
        k: u64,
        ld: i64,
        out_ld: i64,
        a: &I,
        b: &I,
        out: &mut O,
        bias: &I,
    ) -> Result<(), result::CublasError> {
        let a_layout = Matrix::new(k, m, ld)?;
        let b_layout = Matrix::new(k, n, ld)?;
        let c_layout = Matrix::new(m, n, out_ld)?;
        let desc = Desc::new()?;
        desc.set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &1i32,
        )?;
        desc.set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &0i32,
        )?;
        desc.set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC,
            &0i32,
        )?;
        let (bias_ptr, _bias_guard) = bias.device_ptr(&self.stream);
        desc.set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr,
        )?;
        desc.set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            &sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        )?;
        let pref = Pref::new()?;
        let preference = projection_preference(
            self.preference,
            self.xsum_reference,
            self.lm1b_qkv_reference,
            m,
            k,
        );
        let heuristic = select_with_limit(preference, self.buffer.len(), |limit| {
            pref.set_size(limit)?;
            // SAFETY: all descriptors and the serialized handle are live, and
            // caller-checked layouts describe the provided allocations.
            let selected = unsafe {
                result::get_matmul_algo_heuristic(
                    self.handle,
                    desc.0,
                    a_layout.0,
                    b_layout.0,
                    c_layout.0,
                    c_layout.0,
                    pref.0,
                )?
            };
            Ok((selected.workspaceSize, selected))
        })?;
        let (ap, _ag) = a.device_ptr(&self.stream);
        let (bp, _bg) = b.device_ptr(&self.stream);
        let (cp, _cg) = out.device_ptr_mut(&self.stream);
        let (wp, _wg) = self.buffer.device_ptr(&self.stream);
        let alpha = 1f32;
        let beta = 0f32;
        // SAFETY: caller validates dimensions, strides, FP32 storage bounds and
        // output size. All resources and tracking guards remain live through
        // submission; the mutex and owning stream serialize workspace use.
        unsafe {
            result::matmul(
                self.handle,
                desc.0,
                (&alpha as *const f32).cast(),
                (&beta as *const f32).cast(),
                ap as *const _,
                a_layout.0,
                bp as *const _,
                b_layout.0,
                cp as *const _,
                c_layout.0,
                cp as *mut _,
                c_layout.0,
                &heuristic.algo,
                wp as *mut _,
                self.buffer.len(),
                self.stream.cu_stream() as *mut _,
            )
        }
    }
}
impl Drop for BoundedLt {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        // SAFETY: sole handle ownership; the stream remains alive through drop.
        let _ = unsafe { result::destroy_handle(self.handle) };
    }
}
struct Matrix(sys::cublasLtMatrixLayout_t);
impl Matrix {
    fn new(rows: u64, cols: u64, ld: i64) -> Result<Self, result::CublasError> {
        Ok(Self(result::create_matrix_layout(
            sys::cudaDataType_t::CUDA_R_32F,
            rows,
            cols,
            ld,
        )?))
    }
}
impl Drop for Matrix {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        let _ = unsafe { result::destroy_matrix_layout(self.0) };
    }
}
struct Desc(sys::cublasLtMatmulDesc_t);
impl Desc {
    fn new() -> Result<Self, result::CublasError> {
        Ok(Self(result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )?))
    }
    #[allow(unsafe_code)]
    fn set<T>(
        &self,
        attr: sys::cublasLtMatmulDescAttributes_t,
        value: &T,
    ) -> Result<(), result::CublasError> {
        // SAFETY: internal callers pair each attribute with its specified type.
        unsafe {
            result::set_matmul_desc_attribute(
                self.0,
                attr,
                (value as *const T).cast(),
                std::mem::size_of::<T>(),
            )
        }
    }
}
impl Drop for Desc {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        let _ = unsafe { result::destroy_matmul_desc(self.0) };
    }
}
struct Pref(sys::cublasLtMatmulPreference_t);
impl Pref {
    fn new() -> Result<Self, result::CublasError> {
        Ok(Self(result::create_matmul_pref()?))
    }
    #[allow(unsafe_code)]
    fn set_size(&self, size: usize) -> Result<(), result::CublasError> {
        // SAFETY: workspace preference is documented as a size_t.
        unsafe {
            result::set_matmul_pref_attribute(
                self.0,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&size as *const usize).cast(),
                std::mem::size_of::<usize>(),
            )
        }
    }
}
impl Drop for Pref {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        let _ = unsafe { result::destroy_matmul_pref(self.0) };
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn diagnostic_preference_is_explicit_and_shape_bounded() {
        let default = 4 * 1024 * 1024;
        assert_eq!(
            super::projection_preference(default, false, false, 2304, 768),
            default
        );
        assert_eq!(
            super::projection_preference(default, true, false, 2304, 768),
            default
        );
        assert_eq!(
            super::projection_preference(default, true, true, 2304, 768),
            1024 * 1024
        );
        assert_eq!(
            super::projection_preference(default, true, true, 2304, 769),
            default
        );
        assert_eq!(
            super::projection_preference(default, true, false, 768, 768),
            1024 * 1024
        );
    }
    use super::*;

    #[test]
    fn oversized_selection_retries_and_never_launches_oversized_result() {
        let mut calls = Vec::new();
        let selected = select_with_limit(100, 40, |limit| {
            calls.push(limit);
            Ok(if limit == 100 {
                (80, "large")
            } else {
                (40, "fits")
            })
        })
        .unwrap();
        assert_eq!(selected, "fits");
        assert_eq!(calls, [100, 40]);

        let mut calls = 0;
        let error = select_with_limit(100, 40, |_| {
            calls += 1;
            Ok((41, ()))
        })
        .unwrap_err();
        assert_eq!(calls, 2);
        assert_eq!(error.0, sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED);
    }

    #[test]
    fn fitting_selection_and_errors_are_not_retried() {
        let mut calls = 0;
        let selected = select_with_limit(100, 40, |_| {
            calls += 1;
            Ok((40, 7))
        })
        .unwrap();
        assert_eq!(selected, 7);
        assert_eq!(calls, 1);
        let mut calls = 0;
        let error = select_with_limit::<()>(100, 40, |_| {
            calls += 1;
            Err(result::CublasError(
                sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED,
            ))
        })
        .unwrap_err();
        assert_eq!(calls, 1);
        assert_eq!(error.0, sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED);
    }
}
