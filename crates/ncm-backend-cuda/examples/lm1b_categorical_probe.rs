//! Compare 8192-way device draws with checked controlled LM1B CPU outputs.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use ncm_io::sha256;
use safetensors::{Dtype, SafeTensors};
use std::{error::Error, fs};

fn next_word(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 32) as u32
}
fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("expected Rust trace")?;
    let bytes = fs::read(path)?;
    if sha256(&bytes) != "8d5184bc9ffeb0600c724d2454682cf3acf79e6488de4079fd8d4c162bb364e5" {
        return Err("unexpected trace hash".into());
    }
    let tensors = SafeTensors::deserialize(&bytes)?;
    let codes = tensors.tensor("generator.generated_codes")?;
    if codes.dtype() != Dtype::I64 || codes.shape() != [2, 641] {
        return Err("invalid codes".into());
    }
    let expected: Vec<i64> = codes
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|x| i64::from_le_bytes(*x))
        .collect();
    let b = CandleBackend::cuda(0)?;
    let mut state = 11;
    next_word(&mut state);
    next_word(&mut state);
    let mut differences = Vec::new();
    let mut offset = 1;
    for (level, length) in [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
        .into_iter()
        .enumerate()
    {
        let t = tensors.tensor(&format!("generator.level_{:02}.logits", level + 1))?;
        if t.dtype() != Dtype::F32 || t.shape() != [2, length, 8192] {
            return Err("invalid logits".into());
        }
        let values: Vec<f32> = t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let words: Vec<u32> = (0..2 * length).map(|_| next_word(&mut state)).collect();
        let logits = b.upload_f32(&values, &[2, length, 8192])?;
        let words = b.upload_indices(&words, &[2, length])?;
        let actual = b.read_indices(&b.categorical_8192(&logits, &words)?)?;
        for row in 0..2 {
            for col in 0..length {
                let got = actual[row * length + col];
                let want = expected[row * 641 + offset + col];
                if i64::from(got) != want {
                    differences.push((level + 1, row, col, got, want));
                }
            }
        }
        offset += length;
    }
    let mut rows = vec![0.0_f32; 4 * 8192];
    rows[8192] = f32::NAN;
    rows[2 * 8192] = f32::INFINITY;
    rows[3 * 8192] = f32::NEG_INFINITY;
    let logits = b.upload_f32(&rows, &[1, 4, 8192])?;
    let words = b.upload_indices(&[0, 0, 0, 0], &[1, 4])?;
    let invalid = b.read_indices(&b.categorical_8192(&logits, &words)?)?;
    if invalid != [0, u32::MAX, u32::MAX, u32::MAX] {
        return Err(format!("invalid row handling: {invalid:?}").into());
    }
    let mut boundary_rows = vec![0.0_f32; 4 * 8192];
    boundary_rows[8192..2 * 8192].fill(-1000.0);
    boundary_rows[8192 + 4097] = 0.0;
    boundary_rows[2 * 8192..3 * 8192].fill(-1000.0);
    boundary_rows[2 * 8192] = 0.0;
    boundary_rows[2 * 8192 + 8191] = 0.0;
    boundary_rows[3 * 8192..4 * 8192].fill(-1000.0);
    boundary_rows[3 * 8192] = 0.0;
    boundary_rows[3 * 8192 + 8191] = 0.0;
    let logits = b.upload_f32(&boundary_rows, &[1, 4, 8192])?;
    let words = b.upload_indices(&[u32::MAX, u32::MAX, 0x7fff_ffff, 0x8000_0000], &[1, 4])?;
    let boundary = b.read_indices(&b.categorical_8192(&logits, &words)?)?;
    if boundary != [8191, 4097, 0, 8191] {
        return Err(format!("boundary handling: {boundary:?}").into());
    }
    println!(
        "{}",
        serde_json::json!({"choices":1280,"differences":differences,"invalid_cases_passed":true,"boundary_cases_passed":true})
    );
    if !differences.is_empty() {
        return Err("device and CPU selected differently".into());
    }
    Ok(())
}
