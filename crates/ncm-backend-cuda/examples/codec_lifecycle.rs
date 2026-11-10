#[path = "../../../tests/support/codec_lifecycle.rs"]
mod lifecycle;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    lifecycle::run(|| Ok(ncm_backend_cuda::CandleBackend::cuda(0)?))
}
