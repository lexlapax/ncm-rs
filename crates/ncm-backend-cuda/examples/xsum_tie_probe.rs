//! Compare XSum reference tie selection against captured PyTorch CUDA rows.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use safetensors::{Dtype, SafeTensors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected reference path")?;
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let logits = tensors.tensor("logits")?;
    let selected = tensors.tensor("selected")?;
    if logits.dtype() != Dtype::F32
        || logits.shape() != [1024, 4096]
        || selected.dtype() != Dtype::I64
        || selected.shape() != [1024]
    {
        return Err("invalid reference shapes".into());
    }
    let expected: Vec<u32> = selected
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| u32::try_from(i64::from_le_bytes(*v)))
        .collect::<Result<_, _>>()?;
    let backend = CandleBackend::cuda_cudnn(0)?.xsum_reference_backend()?;
    let mut differences = Vec::new();
    for batch in 0..8 {
        let start = batch * 128 * 4096 * 4;
        let values: Vec<f32> = logits.data()[start..start + 128 * 4096 * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let tensor = backend.upload_f32(&values, &[1, 128, 4096])?;
        let actual = backend.read_indices(&backend.xsum_reference_top1_4096(&tensor)?)?;
        for (position, (&a, &b)) in actual
            .iter()
            .zip(&expected[batch * 128..(batch + 1) * 128])
            .enumerate()
        {
            if a != b {
                differences
                    .push(serde_json::json!({"row":batch * 128 + position,"native":a,"python":b}));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"rows":expected.len(),"differences":differences})
    );
    if !differences.is_empty() {
        return Err("XSum tie selection differs".into());
    }
    Ok(())
}
