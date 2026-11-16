//! Inspect controlled draws on forced-code XSum model logits.
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use ncm_io::Artifact;
use ncm_model::{TraceSink, XsumTeacher};
use safetensors::{Dtype, SafeTensors};

struct Capture<'a> {
    backend: MlxBackend,
    reference: SafeTensors<'a>,
    words: Vec<Vec<u32>>,
    results: Vec<serde_json::Value>,
    levels: Vec<serde_json::Value>,
    reference_codes: Vec<u32>,
}
impl TraceSink<MlxBackend> for Capture<'_> {
    fn tensor(&mut self, name: &str, tensor: &ncm_backend_mlx::MlxTensor) -> ncm_core::Result<()> {
        for level in 0..9 {
            if name != format!("generator.level_{level:02}.logits") {
                continue;
            }
            let expected = self
                .reference
                .tensor(name)
                .map_err(|e| ncm_core::Error::Weights(e.to_string()))?;
            let length = expected.shape()[1];
            let native = self.backend.read_f32(tensor)?;
            let expected_values: Vec<f32> = expected
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            if expected.dtype() != Dtype::F32
                || native.len() != 8 * length * 4096
                || expected_values.len() != native.len()
                || native
                    .iter()
                    .chain(&expected_values)
                    .any(|v| !v.is_finite())
            {
                return Err(ncm_core::Error::Weights("invalid level trace".into()));
            }
            let mut max_delta = 0.0f64;
            let mut outside = 0usize;
            let mut max_tv = 0.0f64;
            let mut sum_tv = 0.0f64;
            for (a, b) in native
                .as_chunks::<4096>()
                .0
                .iter()
                .zip(expected_values.as_chunks::<4096>().0)
            {
                let ma = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mb = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sa: f64 = a
                    .iter()
                    .map(|v| (f64::from(*v) - f64::from(ma)).exp())
                    .sum();
                let sb: f64 = b
                    .iter()
                    .map(|v| (f64::from(*v) - f64::from(mb)).exp())
                    .sum();
                let mut distance = 0.0;
                for (&x, &y) in a.iter().zip(b) {
                    let delta = (f64::from(x) - f64::from(y)).abs();
                    max_delta = max_delta.max(delta);
                    outside += usize::from(delta > 5e-4 + 2e-4 * f64::from(y).abs());
                    distance += (((f64::from(x) - f64::from(ma)).exp() / sa)
                        - ((f64::from(y) - f64::from(mb)).exp() / sb))
                        .abs();
                }
                let tv = distance * 0.5;
                max_tv = max_tv.max(tv);
                sum_tv += tv;
            }
            let word_tensor = self
                .backend
                .upload_indices(&self.words[level], &[8, length])?;
            let choices = self
                .backend
                .read_indices(&self.backend.xsum_categorical_4096(tensor, &word_tensor)?)?;
            let offset: usize = (0..level).map(|prior| self.words[prior].len() / 8).sum();
            let changed = (0..8)
                .flat_map(|row| (0..length).map(move |position| (row, position)))
                .filter(|&(row, position)| {
                    choices[row * length + position]
                        != self.reference_codes[row * 200 + offset + position]
                })
                .count();
            self.levels
                .push(serde_json::json!({"level":level,"rows":8 * length,
                "max_logit_delta":max_delta,"outside_band":outside,"max_tv":max_tv,
                "mean_tv":sum_tv / (8 * length) as f64,"controlled_choice_differences":changed}));
        }
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
            let reference_choice = self.backend.read_indices(
                &self
                    .backend
                    .xsum_categorical_4096(&reference, &word_tensor)?,
            )?[0];
            let native_choice = self
                .backend
                .read_indices(&self.backend.xsum_categorical_4096(&native, &word_tensor)?)?[0];
            self.results.push(
                serde_json::json!({"level":level,"row":row,"position":position,
                "word":word,"max_logit_delta":max_delta,"reference_choice":reference_choice,
                "native_choice":native_choice}),
            );
        }
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &ncm_backend_mlx::MlxTensor) -> ncm_core::Result<()> {
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
    let backend = MlxBackend::gpu()?;
    let artifact = Artifact::load_inference(&args[1])?;
    let teacher = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut capture = Capture {
        backend,
        reference,
        words,
        results: Vec::new(),
        levels: Vec::new(),
        reference_codes: codes.clone(),
    };
    teacher.run_traced(&prefix, &codes, 8, &mut capture)?;
    if capture.results.len() != 2 || capture.levels.len() != 9 {
        return Err("missing target rows".into());
    }
    println!(
        "{}",
        serde_json::json!({"targets":capture.results,"levels":capture.levels})
    );
    Ok(())
}
