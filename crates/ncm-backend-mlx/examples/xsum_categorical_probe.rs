//! Compare controlled categorical draws on captured Python XSum logits.
use ncm_backend_mlx::MlxBackend;
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
    let path = std::env::args().nth(1).ok_or("expected trace path")?;
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let expected = tensors.tensor("generator.selected_codes")?;
    if expected.dtype() != Dtype::I64 || expected.shape() != [8, 200] {
        return Err("invalid expected codes".into());
    }
    let expected: Vec<i64> = expected
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| i64::from_le_bytes(*bytes))
        .collect();
    let backend = MlxBackend::gpu()?;
    let mut state = 42u64;
    let mut offset = 0;
    let mut differences = Vec::new();
    for level in 0..9 {
        let tensor = tensors.tensor(&format!("generator.level_{level:02}.logits"))?;
        let [8, length, 4096] = tensor.shape() else {
            return Err("invalid logits shape".into());
        };
        if tensor.dtype() != Dtype::F32 {
            return Err("invalid logits type".into());
        }
        let logits: Vec<f32> = tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        let words: Vec<u32> = (0..8 * length).map(|_| next_word(&mut state)).collect();
        let logits = backend.upload_f32(&logits, &[8, *length, 4096])?;
        let words = backend.upload_indices(&words, &[8, *length])?;
        let actual = backend.read_indices(&backend.xsum_categorical_4096(&logits, &words)?)?;
        for row in 0..8 {
            for position in 0..*length {
                let native = actual[row * length + position];
                let reference = expected[row * 200 + offset + position];
                if i64::from(native) != reference {
                    differences.push(
                        serde_json::json!({"level":level,"row":row,"position":position,
                        "native":native,"reference":reference}),
                    );
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
        serde_json::json!({"rows":8,"codes":1600,"differences":differences})
    );
    Ok(())
}
