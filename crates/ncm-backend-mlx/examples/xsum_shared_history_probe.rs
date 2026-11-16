//! Compare Metal sampling and model logits against saved CUDA histories.
use ncm_backend_mlx::{MlxBackend, MlxTensor};
use ncm_core::{CodecBackend, Error, GeneratorBackend, WeightBackend};
use ncm_io::Artifact;
use ncm_model::{TraceSink, XsumTeacher};
use safetensors::{Dtype, SafeTensors};

struct Capture<'a> {
    backend: MlxBackend,
    reference: SafeTensors<'a>,
    codes: Vec<u32>,
    words: Vec<Vec<u32>>,
    levels: Vec<serde_json::Value>,
    differences: Vec<serde_json::Value>,
}
impl TraceSink<MlxBackend> for Capture<'_> {
    fn tensor(&mut self, name: &str, tensor: &MlxTensor) -> ncm_core::Result<()> {
        let Some(level) = (0..9).find(|i| name == format!("generator.level_{i:02}.logits")) else {
            return Ok(());
        };
        let saved = self
            .reference
            .tensor(name)
            .map_err(|e| Error::Weights(e.to_string()))?;
        let shape = saved.shape();
        if saved.dtype() != Dtype::F32 || shape.len() != 3 || shape[0] != 32 || shape[2] != 4096 {
            return Err(Error::Weights("invalid CUDA head shape".into()));
        }
        let len = shape[1];
        let cuda: Vec<f32> = saved
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let metal = self.backend.read_f32(tensor)?;
        if metal.len() != cuda.len() || metal.iter().chain(&cuda).any(|v| !v.is_finite()) {
            return Err(Error::Weights("invalid shared-history logits".into()));
        }
        let mut outside = 0usize;
        let mut max_delta = 0.0f64;
        let mut max_tv = 0.0f64;
        for (a, b) in metal
            .as_chunks::<4096>()
            .0
            .iter()
            .zip(cuda.as_chunks::<4096>().0)
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
                let d = (f64::from(x) - f64::from(y)).abs();
                max_delta = max_delta.max(d);
                outside += usize::from(d > 5e-4 + 2e-4 * f64::from(y).abs());
                distance += (((f64::from(x) - f64::from(ma)).exp() / sa)
                    - ((f64::from(y) - f64::from(mb)).exp() / sb))
                    .abs();
            }
            max_tv = max_tv.max(distance * 0.5);
        }
        let words = self
            .backend
            .upload_indices(&self.words[level], &[32, len])?;
        let saved_logits = self.backend.upload_f32(&cuda, &[32, len, 4096])?;
        let same = self
            .backend
            .read_indices(&self.backend.categorical_4096(&saved_logits, &words)?)?;
        let forced = self
            .backend
            .read_indices(&self.backend.categorical_4096(tensor, &words)?)?;
        let offset: usize = self.words[..level].iter().map(|x| x.len() / 32).sum();
        let mut same_count = 0usize;
        let mut forced_count = 0usize;
        for row in 0..32 {
            for pos in 0..len {
                let expected = self.codes[row * 200 + offset + pos];
                let a = same[row * len + pos];
                let b = forced[row * len + pos];
                same_count += usize::from(a != expected);
                forced_count += usize::from(b != expected);
                let original_first =
                    [(0, 27), (7, 15), (14, 2), (25, 144)].contains(&(row, offset + pos));
                if a != expected || b != expected || original_first {
                    self.differences.push(
                        serde_json::json!({"level":level,"row":row,"position":pos,
                        "flat_position":offset+pos,"word":self.words[level][row*len+pos],
                        "cuda_choice":expected,"metal_same_logits":a,"metal_forced_logits":b,
                        "original_first_difference":original_first}),
                    );
                }
            }
        }
        self.levels.push(
            serde_json::json!({"level":level,"rows":32*len,"outside_band":outside,
            "max_logit_delta":max_delta,"max_tv":max_tv,"same_logit_differences":same_count,
            "forced_logit_differences":forced_count}),
        );
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &MlxTensor) -> ncm_core::Result<()> {
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: xsum_shared_history_probe ARTIFACT INPUT REFERENCE LOGITS".into());
    }
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let codes: Vec<u32> = serde_json::from_value(record["codes"].clone())?;
    if rows.len() != 32
        || rows.iter().any(|r| r.len() != 512)
        || codes.len() != 32 * 200
        || record["seed"] != 42
    {
        return Err("invalid shared-history inputs".into());
    }
    let data = std::fs::read(&args[4])?;
    let reference = SafeTensors::deserialize(&data)?;
    let mut state = 42;
    let mut words = Vec::new();
    for level in 0..9 {
        let t = reference.tensor(&format!("generator.level_{level:02}.logits"))?;
        words.push(
            (0..32 * t.shape()[1])
                .map(|_| next_word(&mut state))
                .collect(),
        );
    }
    let prefix: Vec<u32> = rows.into_iter().flatten().collect();
    let backend = MlxBackend::gpu()?;
    let artifact = Artifact::load_inference(&args[1])?;
    let model = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut capture = Capture {
        backend,
        reference,
        codes: codes.clone(),
        words,
        levels: Vec::new(),
        differences: Vec::new(),
    };
    model.run_traced(&prefix, &codes, 32, &mut capture)?;
    if capture.levels.len() != 9 {
        return Err("missing shared-history levels".into());
    }
    println!(
        "{}",
        serde_json::json!({"rows":32,"levels":capture.levels,"differences":capture.differences})
    );
    Ok(())
}
