//! Compare ordinary top-k-1 XSum generation traces against pinned Python CUDA.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::{TraceSink, XsumTeacher};
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;

struct Capture<'a> {
    backend: CandleBackend,
    reference: SafeTensors<'a>,
    stats: BTreeMap<String, serde_json::Value>,
}
impl TraceSink<CandleBackend> for Capture<'_> {
    fn tensor(&mut self, name: &str, tensor: &candle_core::Tensor) -> ncm_core::Result<()> {
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
        let mut sum = 0.0;
        let mut outside = 0usize;
        for (x, bytes) in actual.iter().zip(expected.data().as_chunks::<4>().0) {
            let y = f32::from_le_bytes(*bytes);
            if !x.is_finite() || !y.is_finite() {
                return Err(ncm_core::Error::Weights("nonfinite trace".into()));
            }
            let delta = (f64::from(*x) - f64::from(y)).abs();
            max = max.max(delta);
            sum += delta * delta;
            outside += usize::from(delta > 5e-4 + 2e-4 * f64::from(y).abs());
        }
        let mut stats = serde_json::json!({"elements":actual.len(), "max_abs":max,
            "rms":(sum / actual.len() as f64).sqrt(), "accumulated_band_outside":outside});
        if name.starts_with("generator.level_") && name.ends_with(".logits") {
            let mut changed = 0;
            for (a, raw) in actual
                .as_chunks::<4096>()
                .0
                .iter()
                .zip(expected.data().as_chunks::<{ 4096 * 4 }>().0.iter())
            {
                let mut ai = 0;
                let (mut yi, mut ymax) = (0, f32::NEG_INFINITY);
                for (i, bytes) in raw.as_chunks::<4>().0.iter().enumerate() {
                    let y = f32::from_le_bytes(*bytes);
                    if a[i] > a[ai] {
                        ai = i;
                    }
                    if y > ymax {
                        yi = i;
                        ymax = y;
                    }
                }
                changed += usize::from(ai != yi);
            }
            stats["argmax_differences"] = changed.into();
        }
        self.stats.insert(name.into(), stats);
        Ok(())
    }
    fn indices(&mut self, name: &str, tensor: &candle_core::Tensor) -> ncm_core::Result<()> {
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

fn indices(
    reference: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let tensor = reference.tensor(name)?;
    if tensor.dtype() != Dtype::I64 || tensor.shape() != shape {
        return Err(format!("invalid {name}").into());
    }
    Ok(tensor
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|raw| u32::try_from(i64::from_le_bytes(*raw)))
        .collect::<Result<Vec<_>, _>>()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: xsum_top1_probe ARTIFACT REFERENCE".into());
    }
    let data = std::fs::read(&args[2])?;
    let reference = SafeTensors::deserialize(&data)?;
    let prefix = indices(&reference, "generator.prefix_ids", &[2, 512])?;
    let codes = indices(&reference, "generator.selected_codes", &[2, 200])?;
    let expected = indices(&reference, "generator.penalized_ids", &[2, 48])?;
    let backend = CandleBackend::cuda_cudnn(0)?;
    let artifact = Artifact::load_inference(&args[1])?;
    let teacher = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    let mut capture = Capture {
        backend: backend.clone(),
        reference,
        stats: BTreeMap::new(),
    };
    let output = teacher.generate_top1_traced(&prefix, 2, &mut capture)?;
    let actual_codes = backend.read_indices(&output.codes)?;
    capture.stats.insert(
        "generator.codes".into(),
        serde_json::json!({"elements": codes.len(), "differences": actual_codes.iter().zip(&codes).filter(|(a,b)| a != b).count()}),
    );
    let actual = output.ids;
    let differences = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    capture.stats.insert(
        "generator.penalized_ids".into(),
        serde_json::json!({"elements": actual.len(),"differences":differences}),
    );
    println!("{}", serde_json::to_string_pretty(&capture.stats)?);
    let complete = capture.stats.len() == 22
        && [
            "generator.codes",
            "generator.final_logits",
            "generator.greedy_ids",
            "generator.penalized_ids",
        ]
        .iter()
        .all(|name| capture.stats.contains_key(*name))
        && (0..9).all(|level| {
            capture
                .stats
                .contains_key(&format!("generator.level_{level:02}.logits"))
                && capture
                    .stats
                    .contains_key(&format!("reconstruction.level_{level:02}"))
        });
    if !complete {
        return Err("incomplete XSum top1 trace".into());
    }
    if capture.stats.values().any(|row| {
        [
            "accumulated_band_outside",
            "argmax_differences",
            "differences",
        ]
        .iter()
        .any(|key| {
            row.get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                != 0
        })
    }) {
        return Err("XSum top1 comparison failed".into());
    }
    Ok(())
}
