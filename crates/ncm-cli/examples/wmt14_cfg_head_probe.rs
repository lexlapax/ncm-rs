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

    struct HeadCapture<'a, B: GeneratorBackend> {
        backend: &'a B,
        batch: usize,
        observed_levels: usize,
        rejected: Vec<Value>,
        branches: Option<B::Tensor>,
    }

    impl<B: GeneratorBackend> TraceSink<B> for HeadCapture<'_, B> {
        fn tensor(&mut self, name: &str, value: &B::Tensor) -> Result<()> {
            if name.starts_with("diagnostic.level_") && name.ends_with(".branches") {
                self.branches = Some(value.clone());
                return Ok(());
            }
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
            let branches = self
                .branches
                .take()
                .ok_or_else(|| Error::Backend("WMT14 branch trace missing".into()))?;
            if self.backend.shape(&branches) != [2 * batch, *positions, 4096] {
                return Err(Error::Backend("unexpected WMT14 branch shape".into()));
            }
            let mut branch_values = None;
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
                    let all = match &branch_values {
                        Some(all) => all,
                        None => branch_values.insert(self.backend.read_f32(&branches)?),
                    };
                    let (branch_rows, remainder) = all.as_chunks::<4096>();
                    if !remainder.is_empty() || branch_rows.len() != 2 * batch * positions {
                        return Err(Error::Backend("incomplete WMT14 branch scores".into()));
                    }
                    self.rejected.push(json!({
                        "level": level,
                        "row": index / positions,
                        "position": index % positions,
                        "maximum": maximum,
                        "maximum_indices": matching,
                        "nonfinite": nonfinite,
                        "score_bits": row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "conditional_bits": branch_rows[index].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "null_bits": branch_rows[index + batch * positions].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
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
        let model = Wmt14Teacher::from_artifact_cfg_internals_diagnostic(
            backend.clone(),
            &artifact,
            false,
        )?;
        let prefixes: Vec<_> = rows.iter().flatten().copied().collect();
        let mut trace = HeadCapture {
            backend: &backend,
            batch: rows.len(),
            observed_levels: 0,
            rejected: Vec::new(),
            branches: None,
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
