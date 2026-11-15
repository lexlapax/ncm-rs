//! Compare controlled TinyStories teacher-forced tensors on Metal with pinned Python.
use ncm_backend_mlx::MlxBackend;
use ncm_backend_mlx::MlxTensor as Tensor;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::{Codec, TinyStoriesTeacher, TraceSink};
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;

struct Capture<'a> {
    backend: MlxBackend,
    reference: SafeTensors<'a>,
    stats: BTreeMap<String, serde_json::Value>,
}
impl TraceSink<MlxBackend> for Capture<'_> {
    fn tensor(&mut self, name: &str, tensor: &Tensor) -> ncm_core::Result<()> {
        let name = if name == "encode.pre_quant" {
            "codec.pre_quant"
        } else if name.starts_with("encode.") {
            return Ok(());
        } else {
            name
        };
        let expected = self
            .reference
            .tensor(name)
            .map_err(|e| ncm_core::Error::Weights(e.to_string()))?;
        let actual = self.backend.read_f32(tensor)?;
        if expected.dtype() != Dtype::F32
            || expected.shape() != self.backend.shape(tensor)
            || actual.len() * 4 != expected.data().len()
        {
            return Err(ncm_core::Error::Weights(format!(
                "trace shape mismatch {name}"
            )));
        }
        let mut max: f64 = 0.0;
        let mut max_normalized: f64 = 0.0;
        let mut outside = 0usize;
        let mut accumulated_outside = 0usize;
        let mut accumulated_max_normalized: f64 = 0.0;
        let mut sum = 0.0;
        for (x, raw) in actual.iter().zip(expected.data().as_chunks::<4>().0) {
            let y = f32::from_le_bytes(*raw);
            if !x.is_finite() || !y.is_finite() {
                return Err(ncm_core::Error::Weights(format!("nonfinite trace {name}")));
            }
            let diff = f64::from(*x) - f64::from(y);
            max = max.max(diff.abs());
            let normalized = diff.abs() / (1e-5 + 1e-4 * f64::from(y).abs());
            max_normalized = max_normalized.max(normalized);
            outside += usize::from(normalized > 1.0);
            let accumulated = diff.abs() / (5e-4 + 2e-4 * f64::from(y).abs());
            accumulated_max_normalized = accumulated_max_normalized.max(accumulated);
            accumulated_outside += usize::from(accumulated > 1.0);
            sum += diff * diff;
        }
        let mut stats = serde_json::json!({"elements":actual.len(),"max_abs":max,"rms":(sum / actual.len() as f64).sqrt(),"operator_starting_band_atol_1e-5_rtol_1e-4_outside":outside,"max_normalized_error":max_normalized,"phase2_accumulated_band_atol_5e-4_rtol_2e-4_outside":accumulated_outside,"phase2_accumulated_max_normalized":accumulated_max_normalized});
        if name.starts_with("generator.level_") && name.ends_with(".logits") {
            let mut changed = 0usize;
            for (a, bytes) in actual
                .as_chunks::<4096>()
                .0
                .iter()
                .zip(expected.data().as_chunks::<{ 4096 * 4 }>().0.iter())
            {
                let (mut ai, mut yi) = (0usize, 0usize);
                let mut ymax = f32::NEG_INFINITY;
                for (i, raw) in bytes.as_chunks::<4>().0.iter().enumerate() {
                    let y = f32::from_le_bytes(*raw);
                    if a[i] > a[ai] {
                        ai = i;
                    }
                    if y > ymax {
                        yi = i;
                        ymax = y;
                    }
                }
                if ai != yi {
                    changed += 1;
                }
            }
            stats["argmax_differences"] = changed.into();
        }
        self.stats.insert(name.into(), stats);
        Ok(())
    }
    fn indices(&mut self, name: &str, tensor: &Tensor) -> ncm_core::Result<()> {
        let name = if name == "encode.codes" {
            "codec.codes"
        } else if name.starts_with("encode.") {
            return Ok(());
        } else {
            name
        };
        let expected = self
            .reference
            .tensor(name)
            .map_err(|e| ncm_core::Error::Weights(e.to_string()))?;
        let actual = self.backend.read_indices(tensor)?;
        if expected.dtype() != Dtype::I64
            || expected.shape() != self.backend.indices_shape(tensor)
            || actual.len() * 8 != expected.data().len()
        {
            return Err(ncm_core::Error::Weights(format!(
                "trace shape mismatch {name}"
            )));
        }
        let differences = actual
            .iter()
            .zip(expected.data().as_chunks::<8>().0)
            .filter(|(x, raw)| i64::from(**x) != i64::from_le_bytes(**raw))
            .count();
        self.stats.insert(
            name.into(),
            serde_json::json!({"elements":actual.len(),"differences":differences}),
        );
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: tinystories_teacher_probe ARTIFACT PYTHON_TRACE".into());
    }
    let data = std::fs::read(&args[2])?;
    let reference = SafeTensors::deserialize(&data)?;
    let raw = reference.tensor("generator.forced_codes")?;
    let [batch, total] = raw.shape() else {
        return Err("invalid forced-code rank".into());
    };
    let batch = *batch;
    if raw.dtype() != Dtype::I64 || !(1..=16).contains(&batch) || *total != 641 {
        return Err("invalid forced codes".into());
    }
    let codes: Vec<u32> = raw
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|x| u32::try_from(i64::from_le_bytes(*x)))
        .collect::<Result<_, _>>()?;
    let backend = MlxBackend::gpu()?;
    let artifact = Artifact::load_inference(&args[1])?;
    let teacher = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut capture = Capture {
        backend,
        reference,
        stats: BTreeMap::new(),
    };
    teacher.run_traced(&vec![0; batch], &codes, batch, &mut capture)?;
    drop(teacher);
    let raw = capture.reference.tensor("codec.input_ids")?;
    if raw.dtype() != Dtype::I64 || raw.shape() != [batch, 128] {
        return Err("invalid codec inputs".into());
    }
    let inputs: Vec<u32> = raw
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|x| u32::try_from(i64::from_le_bytes(*x)))
        .collect::<Result<_, _>>()?;
    let codec = Codec::from_artifact(capture.backend.clone(), &artifact)?;
    let encoded = codec.encode_traced(&inputs, batch, &mut capture)?;
    let encoded_ids = capture.backend.read_indices(&encoded.codes)?;
    let decoded = codec.decode(&encoded_ids, batch)?;
    capture.tensor("codec.logits", &decoded.logits)?;
    println!("{}", serde_json::to_string_pretty(&capture.stats)?);
    Ok(())
}
