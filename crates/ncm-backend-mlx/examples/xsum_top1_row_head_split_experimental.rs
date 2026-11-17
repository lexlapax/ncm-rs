//! Forced XSum row with experimental two-part head projection.
#[path = "../../ncm-model/examples/support/xsum_top1_row_diagnostic.rs"]
mod shared;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(ncm_backend_mlx::MlxBackend::gpu_xsum_head_split_experimental()?)
}
