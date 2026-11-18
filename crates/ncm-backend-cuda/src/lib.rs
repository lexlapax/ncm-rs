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
#[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
mod cudnn_graph;
#[cfg(feature = "cuda")]
mod fused_projection;
#[cfg(feature = "cuda")]
mod generator_layout;
#[cfg(all(feature = "cuda-cudnn", feature = "lm1b-experimental"))]
mod lm1b_linear_fma;
#[cfg(feature = "cuda")]
pub mod masked_softmax;
#[cfg(feature = "cpu")]
pub mod ops;
#[cfg(feature = "cpu")]
pub mod resample;
#[cfg(feature = "cpu")]
mod runtime;
#[cfg(feature = "cuda")]
mod unique_top1;
#[cfg(feature = "cuda")]
pub mod wmt14_argmax;
#[cfg(feature = "cuda-cudnn")]
mod wmt14_groupnorm;
#[cfg(feature = "cuda")]
mod xsum_gelu;
#[cfg(feature = "cuda")]
mod xsum_groupnorm;
#[cfg(feature = "cuda")]
mod xsum_layernorm;
#[cfg(feature = "cuda")]
mod xsum_penalty;
#[cfg(feature = "cuda")]
mod xsum_rmsnorm;
#[cfg(feature = "cuda")]
mod xsum_rope;
#[cfg(feature = "cpu")]
pub use runtime::CandleBackend;
