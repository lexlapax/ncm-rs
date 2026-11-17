//! Forced XSum row with both explicitly selected experimental Metal operations.
#[path = "../../ncm-model/examples/support/xsum_top1_row_diagnostic.rs"]
mod shared;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(ncm_backend_mlx::MlxBackend::gpu_xsum_combined_experimental()?)
}
