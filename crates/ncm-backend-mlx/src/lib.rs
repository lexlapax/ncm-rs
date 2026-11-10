//! Native MLX operations for the shared codec; no implicit CPU fallback.
#[cfg(all(
    feature = "mlx",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!("the mlx feature requires Apple Silicon macOS");
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod codec;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod ops;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub use codec::{MlxBackend, MlxTensor};
