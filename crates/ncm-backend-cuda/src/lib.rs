//! Candidate NCM tensor operations for Phase 1 capability testing.
//!
//! Enable `cpu` for the fixture reference path or `cuda` for Linux/NVIDIA.
//! These operations are not yet a complete model or a frozen backend contract.

#[cfg(all(feature = "cuda", not(target_os = "linux")))]
compile_error!("The ncm CUDA runtime currently supports Linux only");

#[cfg(feature = "cpu")]
pub mod ops;
#[cfg(feature = "cpu")]
mod runtime;
#[cfg(feature = "cpu")]
pub use runtime::CandleBackend;
