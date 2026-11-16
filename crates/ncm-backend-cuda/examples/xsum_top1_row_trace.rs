//! Capture one row of strict CUDA XSum logits under a forced batch history.
#[path = "../../ncm-model/examples/support/xsum_top1_row_trace.rs"]
mod shared;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?)
}
