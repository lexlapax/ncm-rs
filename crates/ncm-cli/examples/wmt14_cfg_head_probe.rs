//! Identify tied or nonfinite WMT14 guided head scores without changing selection.
use std::{error::Error, fs};

#[cfg(any(
    feature = "cuda-cudnn",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
mod probe {
    use ncm_core::{Error, GeneratorBackend, Result};
    use ncm_io::Artifact;
    use ncm_model::{TraceSink, Wmt14Teacher};
    use serde_json::{Value, json};

    struct HeadCapture<'a, B> {
        backend: &'a B,
        batch: usize,
        observed_levels: usize,
        rejected: Vec<Value>,
    }

    impl<B: GeneratorBackend> TraceSink<B> for HeadCapture<'_, B> {
        fn tensor(&mut self, name: &str, value: &B::Tensor) -> Result<()> {
            let Some(level) = name
                .strip_prefix("generator.level_")
                .and_then(|name| name.strip_suffix(".logits"))
                .and_then(|level| level.parse::<usize>().ok())
            else {
                return Ok(());
            };
            let shape = self.backend.shape(value);
            let [batch, positions, 4096] = shape.as_slice() else {
                return Err(Error::Backend("unexpected WMT14 head shape".into()));
            };
            if *batch != self.batch || *positions == 0 || level != self.observed_levels {
                return Err(Error::Backend(
                    "unexpected WMT14 head batch or level".into(),
                ));
            }
            let values = self.backend.read_f32(value)?;
            if values.len() != batch * positions * 4096 {
                return Err(Error::Backend("incomplete WMT14 head scores".into()));
            }
            self.observed_levels += 1;
            for (index, row) in values.as_chunks::<4096>().0.iter().enumerate() {
                let finite = row.iter().copied().filter(|x| x.is_finite());
                let maximum = finite.clone().reduce(f32::max);
                let nonfinite = row.len() - finite.count();
                let matching: Vec<_> = row
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &x)| (Some(x) == maximum).then_some(i))
                    .collect();
                if nonfinite > 0 || matching.len() != 1 {
                    self.rejected.push(json!({
                        "level": level,
                        "row": index / positions,
                        "position": index % positions,
                        "maximum": maximum,
                        "maximum_indices": matching,
                        "nonfinite": nonfinite,
                        "score_bits": row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    }));
                }
            }
            Ok(())
        }

        fn indices(&mut self, _: &str, _: &B::Indices) -> Result<()> {
            Ok(())
        }
    }

    pub fn run<B: GeneratorBackend + Clone>(
        backend: B,
        path: &str,
        rows: &[Vec<u32>],
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let artifact = Artifact::load_inference(path)?;
        let model = Wmt14Teacher::from_artifact(backend.clone(), &artifact)?;
        let prefixes: Vec<_> = rows.iter().flatten().copied().collect();
        let mut trace = HeadCapture {
            backend: &backend,
            batch: rows.len(),
            observed_levels: 0,
            rejected: Vec::new(),
        };
        let result = model.generate_cfg_top1_traced(&prefixes, rows.len(), 1.0, &mut trace);
        println!(
            "{}",
            json!({"observed_levels": trace.observed_levels, "rejected": trace.rejected,
                   "error": result.as_ref().err().map(ToString::to_string)})
        );
        Ok(result?)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact, device, input] = args.as_slice() else {
        return Err("usage: wmt14_cfg_head_probe ARTIFACT cuda|mlx INPUT_JSON".into());
    };
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&fs::read(input)?)?;
    if !(1..=16).contains(&rows.len()) || rows.iter().any(|row| row.len() != 96) {
        return Err("expected 1..=16 WMT14 prefixes with 96 tokens".into());
    }
    let _ = artifact;
    match device.as_str() {
        #[cfg(feature = "cuda-cudnn")]
        "cuda" => probe::run(
            ncm_backend_cuda::CandleBackend::cuda_wmt14_fast_argmax_experimental(0)?,
            artifact,
            &rows,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => probe::run(ncm_backend_mlx::MlxBackend::gpu()?, artifact, &rows),
        _ => Err("requested device is unavailable".into()),
    }
}
