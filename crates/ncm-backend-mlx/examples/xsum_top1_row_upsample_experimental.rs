//! Forced XSum row with explicitly selected experimental Metal upsampling.
#[path = "../../ncm-model/examples/support/xsum_top1_row_diagnostic.rs"]
mod shared;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(ncm_backend_mlx::MlxBackend::gpu_xsum_upsample_experimental()?)
}
