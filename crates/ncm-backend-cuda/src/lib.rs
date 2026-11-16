//! Candidate NCM tensor operations for Phase 1 capability testing.
//!
//! Enable `cpu` for the fixture reference path or `cuda` for Linux/NVIDIA.
//! These operations are not yet a complete model or a frozen backend contract.

#[cfg(all(feature = "cuda", not(target_os = "linux")))]
compile_error!("The ncm CUDA runtime currently supports Linux only");

#[cfg(feature = "cuda")]
pub mod categorical;
#[cfg(feature = "cpu")]
mod codec;
#[cfg(feature = "cuda-cudnn")]
mod cudnn_convolution;
#[cfg(feature = "cuda")]
mod fused_projection;
#[cfg(feature = "cuda")]
mod generator_layout;
#[cfg(feature = "cpu")]
pub mod ops;
#[cfg(feature = "cpu")]
pub mod resample;
#[cfg(feature = "cpu")]
mod runtime;
#[cfg(feature = "cuda")]
mod unique_top1;
#[cfg(feature = "cuda")]
mod xsum_penalty;
#[cfg(feature = "cpu")]
pub use runtime::CandleBackend;
