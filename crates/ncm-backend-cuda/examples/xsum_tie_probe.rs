//! Compare XSum reference tie selection against captured PyTorch CUDA rows.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use safetensors::{Dtype, SafeTensors};
use std::time::Instant;

fn benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let backend = CandleBackend::cuda_cudnn(0)?.xsum_reference_backend()?;
    let mut results = Vec::new();
    for (batch, length) in [(1, 4), (4, 48), (32, 48)] {
        let rows = batch * length;
        let mut values = vec![-1.0f32; rows * 4096];
        let expected: Vec<u32> = (0..rows).map(|row| ((row * 37) % 4096) as u32).collect();
        for (row, &index) in expected.iter().enumerate() {
            values[row * 4096 + index as usize] = 1.0;
        }
        let logits = backend.upload_f32(&values, &[batch, length, 4096])?;
        let mut old = Vec::new();
        let mut new = Vec::new();
        for trial in 0..110 {
            for reference in if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                backend.device().synchronize()?;
                let start = Instant::now();
                let output = if reference {
                    backend.xsum_reference_top1_4096(&logits)?
                } else {
                    backend.unique_max_4096(&logits)?
                };
                let selected = backend.read_indices(&output)?;
                let seconds = start.elapsed().as_secs_f64();
                if selected != expected {
                    return Err("selector output differs on unique rows".into());
                }
                if trial >= 10 {
                    if reference {
                        new.push(seconds);
                    } else {
                        old.push(seconds);
                    }
                }
            }
        }
        old.sort_by(f64::total_cmp);
        new.sort_by(f64::total_cmp);
        results.push(
            serde_json::json!({"batch":batch,"length":length,"warmups":10,"trials":100,
            "old_median_seconds":(old[49] + old[50]) / 2.0,
            "new_median_seconds":(new[49] + new[50]) / 2.0,
            "old_p95_seconds":old[94],"new_p95_seconds":new[94]}),
        );
    }
    println!(
        "{}",
        serde_json::json!({"scope":"alternating selector plus readback, unique synthetic rows", "results":results})
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected reference path")?;
    if path == "--benchmark" {
        return benchmark();
    }
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
