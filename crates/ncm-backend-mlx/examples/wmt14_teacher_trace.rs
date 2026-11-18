//! Capture the fixed-history WMT14 Metal trace for the pinned CUDA oracle.
#[path = "../../ncm-model/examples/support/wmt14_teacher_trace.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(ncm_backend_mlx::MlxBackend::gpu()?)
}
