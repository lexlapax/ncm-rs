//! Offline native capture against authenticated Sudoku inputs; no Python runtime.
#[path = "../../../tests/support/trained_codec.rs"]
mod trained;
use ncm_backend_cuda::CandleBackend;
use std::{error::Error, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: codec_trace ARTIFACT REFERENCE OUTPUT cpu|cuda".into());
    }
    trained::run(
        &args[1],
        Path::new(&args[2]),
        Path::new(&args[3]),
        "rust-candle",
        &args[4],
        || match args[4].as_str() {
            "cpu" => Ok(CandleBackend::cpu()),
            "cuda" => CandleBackend::cuda(0),
            _ => Err(ncm_core::Error::Input("explicit cpu|cuda required".into())),
        },
        &[
            "crates/ncm-backend-cuda/src/codec.rs",
            "crates/ncm-backend-cuda/src/ops.rs",
            "crates/ncm-backend-cuda/src/runtime.rs",
            "crates/ncm-backend-cuda/examples/codec_trace.rs",
        ],
    )
}
