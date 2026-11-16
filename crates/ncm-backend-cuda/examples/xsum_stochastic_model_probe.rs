//! Inspect controlled draws on forced-code XSum model logits.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use ncm_io::Artifact;
use ncm_model::{TraceSink, XsumTeacher};
use safetensors::{Dtype, SafeTensors};

struct Capture<'a> {
    backend: CandleBackend,
    reference: SafeTensors<'a>,
    words: Vec<Vec<u32>>,
    results: Vec<serde_json::Value>,
}
impl TraceSink<CandleBackend> for Capture<'_> {
    fn tensor(&mut self, name: &str, tensor: &candle_core::Tensor) -> ncm_core::Result<()> {
        for (level, row, position) in [(5, 4, 25), (8, 3, 36)] {
            if name != format!("generator.level_{level:02}.logits") {
                continue;
            }
            let expected = self
                .reference
                .tensor(name)
                .map_err(|e| ncm_core::Error::Weights(e.to_string()))?;
            let length = expected.shape()[1];
            let start = (row * length + position) * 4096 * 4;
            let values: Vec<f32> = expected.data()[start..start + 4096 * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let reference = self.backend.upload_f32(&values, &[1, 1, 4096])?;
            let native =
                self.backend
                    .narrow(&self.backend.narrow(tensor, 0, row, 1)?, 1, position, 1)?;
            let native_values = self.backend.read_f32(&native)?;
            let max_delta = native_values
                .iter()
                .zip(&values)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let word = self.words[level][row * length + position];
            let word_tensor = self.backend.upload_indices(&[word], &[1, 1])?;
            let reference_choice = self
                .backend
                .read_indices(&self.backend.categorical_4096(&reference, &word_tensor)?)?[0];
            let native_choice = self
                .backend
                .read_indices(&self.backend.categorical_4096(&native, &word_tensor)?)?[0];
            self.results.push(
                serde_json::json!({"level":level,"row":row,"position":position,
                "word":word,"max_logit_delta":max_delta,"reference_choice":reference_choice,
                "native_choice":native_choice}),
            );
        }
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &candle_core::Tensor) -> ncm_core::Result<()> {
        Ok(())
    }
}
fn next_word(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 32) as u32
}
fn indices(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let t = tensors.tensor(name)?;
    if t.dtype() != Dtype::I64 || t.shape() != shape {
        return Err("invalid trace indices".into());
    }
    Ok(t.data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| u32::try_from(i64::from_le_bytes(*v)))
        .collect::<Result<Vec<_>, _>>()?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected artifact and trace paths".into());
    }
    let data = std::fs::read(&args[2])?;
    let reference = SafeTensors::deserialize(&data)?;
    let prefix = indices(&reference, "generator.prefix_ids", &[8, 512])?;
    let codes = indices(&reference, "generator.selected_codes", &[8, 200])?;
    let mut state = 42;
    let mut words = Vec::new();
    for level in 0..9 {
        let logits = reference.tensor(&format!("generator.level_{level:02}.logits"))?;
        words.push(
            (0..8 * logits.shape()[1])
                .map(|_| next_word(&mut state))
                .collect(),
        );
    }
    let backend = CandleBackend::cuda_cudnn(0)?;
    let artifact = Artifact::load_inference(&args[1])?;
    let teacher = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut capture = Capture {
        backend,
        reference,
        words,
        results: Vec::new(),
    };
    teacher.run_traced(&prefix, &codes, 8, &mut capture)?;
    if capture.results.len() != 2 {
        return Err("missing target rows".into());
    }
    println!("{}", serde_json::json!({"targets":capture.results}));
    Ok(())
}
