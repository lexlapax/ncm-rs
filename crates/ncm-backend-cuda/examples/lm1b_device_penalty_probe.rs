//! Compare LM1B device FP64 penalties with checked CPU-selected IDs.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use ncm_io::{Artifact, sha256};
use safetensors::{Dtype, SafeTensors};
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: lm1b_device_penalty_probe ARTIFACT TRACE".into());
    }
    if sha256(&fs::read(format!("{}/manifest.json", args[1]))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("artifact differs".into());
    }
    let data = fs::read(&args[2])?;
    if sha256(&data) != "8d5184bc9ffeb0600c724d2454682cf3acf79e6488de4079fd8d4c162bb364e5" {
        return Err("trace differs".into());
    }
    let tensors = SafeTensors::deserialize(&data)?;
    let logits = tensors.tensor("generator.final_logits")?;
    let ids = tensors.tensor("generator.penalized_ids")?;
    if logits.dtype() != Dtype::F32
        || logits.shape() != [2, 128, 30528]
        || ids.dtype() != Dtype::I64
        || ids.shape() != [2, 128]
    {
        return Err("trace shapes differ".into());
    }
    let values: Vec<f32> = logits
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let expected: Vec<u32> = ids
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|x| u32::try_from(i64::from_le_bytes(*x)))
        .collect::<Result<_, _>>()?;
    let artifact = Artifact::load_inference(&args[1])?;
    let data = artifact.file("inference.safetensors")?;
    let inference = SafeTensors::deserialize(data)?;
    let t = inference.tensor("token_unigram")?;
    if t.dtype() != Dtype::F32 || t.shape() != [30528] {
        return Err("allowance differs".into());
    }
    let allowance: Vec<f32> = t
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let b = CandleBackend::cuda(0)?;
    let actual = b.read_indices(&b.lm1b_penalty_fp64(
        &b.upload_f32(&values, &[2, 128, 30528])?,
        &b.upload_f32(&allowance, &[30528])?,
    )?)?;
    if actual != expected {
        return Err("captured penalty output differs".into());
    }
    let zero = vec![0.0_f32; 128 * 30528];
    let allow = b.upload_f32(&vec![0.0; 30528], &[30528])?;
    let result =
        b.read_indices(&b.lm1b_penalty_fp64(&b.upload_f32(&zero, &[1, 128, 30528])?, &allow)?)?;
    if result != (0..128).collect::<Vec<_>>() {
        return Err("tie/repetition result differs".into());
    }
    let mut invalid = zero;
    invalid[0] = f32::NAN;
    let result =
        b.read_indices(&b.lm1b_penalty_fp64(&b.upload_f32(&invalid, &[1, 128, 30528])?, &allow)?)?;
    if result.iter().any(|&x| x != u32::MAX) {
        return Err("invalid row not rejected".into());
    }
    let mut negative = vec![0.0_f32; 30528];
    negative[3] = -1.0;
    let bad_allow = b.upload_f32(&negative, &[30528])?;
    let zeros = b.upload_f32(&vec![0.0; 128 * 30528], &[1, 128, 30528])?;
    let result = b.read_indices(&b.lm1b_penalty_fp64(&zeros, &bad_allow)?)?;
    if result.iter().any(|&x| x != u32::MAX) {
        return Err("invalid allowance not rejected".into());
    }
    if b.lm1b_penalty_fp64(
        &b.upload_f32(&[0.0], &[1, 1, 1])?,
        &b.upload_f32(&[0.0], &[1])?,
    )
    .is_ok()
    {
        return Err("invalid shape accepted".into());
    }
    println!(
        "{}",
        serde_json::json!({"captured_ids":256,"matched":true,"tie_nonfinite_allowance_and_shape_tests":true})
    );
    Ok(())
}
