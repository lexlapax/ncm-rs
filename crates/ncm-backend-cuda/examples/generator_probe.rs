//! Diagnostic: run the fixed synthetic Sudoku prefixes on an explicit device.
use candle_core::Tensor;
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::{Generator, TraceSink};
use std::collections::BTreeMap;
struct Capture {
    backend: CandleBackend,
    values: BTreeMap<String, Vec<f32>>,
}
impl TraceSink<CandleBackend> for Capture {
    fn tensor(&mut self, name: &str, x: &Tensor) -> ncm_core::Result<()> {
        self.values
            .insert(name.to_owned(), self.backend.read_f32(x)?);
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &Tensor) -> ncm_core::Result<()> {
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: generator_probe ARTIFACT cpu|cuda [INPUT_JSON]".into());
    }
    let backend = match args[2].as_str() {
        "cpu" => CandleBackend::cpu(),
        "cuda" => CandleBackend::cuda(0)?,
        _ => return Err("explicit cpu|cuda required".into()),
    };
    let artifact = Artifact::load(&args[1])?;
    let generator = Generator::from_artifact(backend.clone(), &artifact)?;
    let mut prefixes = Vec::new();
    if let Some(path) = args.get(3) {
        let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(path)?)?;
        if rows.is_empty() || rows.iter().any(|row| row.len() != 91) {
            return Err("invalid prefixes".into());
        }
        prefixes = rows.into_iter().flatten().collect();
    } else {
        for shift in 0..2 {
            prefixes.push(11);
            for row in 0..9 {
                if row > 0 {
                    prefixes.push(10);
                }
                for col in 0..9 {
                    let digit = (row * 3 + row / 3 + col + shift) % 9 + 1;
                    prefixes.push(if (row * 9 + col) % 3 == 0 {
                        0
                    } else {
                        digit as u32
                    });
                }
            }
            prefixes.push(11);
        }
    }
    let batch = prefixes.len() / 91;
    let mut capture = Capture {
        backend: backend.clone(),
        values: BTreeMap::new(),
    };
    let output = generator.generate_traced(&prefixes, batch, &mut capture)?;
    println!(
        "{}",
        serde_json::json!({"codes":backend.read_indices(&output.codes)?,"ids":backend.read_indices(&output.ids)?, "logits":capture.values})
    );
    Ok(())
}
