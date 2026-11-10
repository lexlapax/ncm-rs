//! Real-Metal capture using the same authenticated inputs and shared trace harness.
#[path = "../../../tests/support/trained_codec.rs"]
mod trained;
use ncm_backend_mlx::MlxBackend;
use std::{error::Error, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: mlx_codec_trace ARTIFACT REFERENCE OUTPUT".into());
    }
    trained::run(
        &args[1],
        Path::new(&args[2]),
        Path::new(&args[3]),
        "rust-mlx",
        "metal",
        MlxBackend::gpu,
        &[
            "crates/ncm-backend-mlx/src/codec.rs",
            "crates/ncm-backend-mlx/src/ops.rs",
            "crates/ncm-backend-mlx/examples/mlx_codec_trace.rs",
        ],
    )
}
