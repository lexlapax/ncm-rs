//! Capture a WMT14 fixed-history CUDA trace against the pinned oracle.
#[path = "../../ncm-model/examples/support/wmt14_teacher_trace.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(ncm_backend_cuda::CandleBackend::cuda(0)?)
}
