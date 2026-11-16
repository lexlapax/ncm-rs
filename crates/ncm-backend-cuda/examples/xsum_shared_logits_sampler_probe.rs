//! Resample saved CUDA shared-history logits with the original random words.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use safetensors::{Dtype, SafeTensors};
fn next_word(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 32) as u32
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: xsum_shared_logits_sampler_probe LOGITS REFERENCE".into());
    }
    let bytes = std::fs::read(&args[1])?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let codes: Vec<u32> = serde_json::from_value(reference["codes"].clone())?;
    if codes.len() != 32 * 200 || reference["seed"] != 42 {
        return Err("invalid reference".into());
    }
    let backend = CandleBackend::cuda_cudnn(0)?;
    let mut state = 42u64;
    let mut offset = 0usize;
    let mut differences = Vec::new();
    for level in 0..9 {
        let tensor = tensors.tensor(&format!("generator.level_{level:02}.logits"))?;
        let [32, length, 4096] = tensor.shape() else {
            return Err("invalid logit shape".into());
        };
        if tensor.dtype() != Dtype::F32 {
            return Err("invalid dtype".into());
        }
        let values: Vec<f32> = tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let words: Vec<u32> = (0..32 * length).map(|_| next_word(&mut state)).collect();
        let scores = backend.upload_f32(&values, &[32, *length, 4096])?;
        let uniforms = backend.upload_indices(&words, &[32, *length])?;
        let actual = backend.read_indices(&backend.categorical_4096(&scores, &uniforms)?)?;
        for row in 0..32 {
            for pos in 0..*length {
                let index = offset + pos;
                let expected = codes[row * 200 + index];
                let got = actual[row * length + pos];
                if got != expected {
                    differences.push(serde_json::json!({"level":level,"row":row,
                    "position":pos,"flat_position":index,"expected":expected,"actual":got}));
                }
            }
        }
        offset += length;
    }
    if offset != 200 {
        return Err("invalid code lengths".into());
    }
    println!(
        "{}",
        serde_json::json!({"rows":32,"codes":6400,"differences":differences})
    );
    Ok(())
}
