//! Capture a forced XSum row's level-one hidden and projected values on Metal.
#[path = "../../ncm-model/examples/support/xsum_top1_row_diagnostic.rs"]
mod shared;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(ncm_backend_mlx::MlxBackend::gpu()?)
}
