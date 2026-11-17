//! Shared diagnostic capture of one row from a forced 32-row XSum history.
use ncm_core::{Error, GeneratorBackend};
use ncm_io::Artifact;
use ncm_model::{TraceSink, XsumTeacher, select_xsum_tokens_cpu};
use safetensors::{Dtype, tensor::TensorView};
use std::{collections::BTreeMap, io::Write};

struct Capture<B: GeneratorBackend> {
    backend: B,
    row: usize,
    values: BTreeMap<String, (Vec<usize>, Vec<u8>)>,
    final_logits: Option<Vec<f32>>,
}
impl<B: GeneratorBackend> TraceSink<B> for Capture<B> {
    fn tensor(&mut self, name: &str, tensor: &B::Tensor) -> ncm_core::Result<()> {
        let head = name.starts_with("generator.level_") && name.ends_with(".logits");
        let diagnostic = name.starts_with("diagnostic.level_01.");
        if !head && !diagnostic && name != "generator.final_logits" {
            return Ok(());
        }
        let shape = self.backend.shape(tensor);
        let width = if diagnostic {
            *shape
                .last()
                .ok_or_else(|| Error::Weights("missing width".into()))?
        } else if head {
            4096
        } else {
            50304
        };
        if shape.len() != 3
            || shape[0] != 32
            || shape[2] != width
            || (!head && !diagnostic && shape[1] != 48)
        {
            return Err(Error::Weights(format!("invalid diagnostic tensor {name}")));
        }
        let all = self.backend.read_f32(tensor)?;
        let length = shape[1] * width;
        if all.len() != 32 * length {
            return Err(Error::Weights(format!("invalid diagnostic length {name}")));
        }
        let row = &all[self.row * length..(self.row + 1) * length];
        if row.iter().any(|v| !v.is_finite()) {
            return Err(Error::Weights(format!(
                "nonfinite diagnostic tensor {name}"
            )));
        }
        if !head && !diagnostic {
            self.final_logits = Some(row.to_vec());
        }
        self.values.insert(
            name.to_owned(),
            (
                vec![1, shape[1], width],
                row.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        );
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &B::Indices) -> ncm_core::Result<()> {
        Ok(())
    }
}

pub fn run<B: GeneratorBackend + Clone>(backend: B) -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: xsum_top1_row_trace ARTIFACT INPUT_JSON CUDA_OUTPUT_JSON ROW_SLOT OUTPUT"
                .into(),
        );
    }
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let source: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let codes: Vec<Vec<u32>> = serde_json::from_value(source["codes"].clone())?;
    let row: usize = args[4].parse()?;
    if rows.len() != 32
        || codes.len() != 32
        || row >= 32
        || rows
            .iter()
            .any(|r| r.len() != 512 || r.iter().any(|&x| x >= 50304))
        || codes
            .iter()
            .any(|r| r.len() != 200 || r.iter().any(|&x| x >= 4096))
    {
        return Err("invalid diagnostic inputs".into());
    }
    let prefix: Vec<u32> = rows.into_iter().flatten().collect();
    let forced: Vec<u32> = codes.into_iter().flatten().collect();
    let artifact = Artifact::load_inference(&args[1])?;
    let model = XsumTeacher::from_owned_artifact(backend.clone(), artifact)?;
    let mut capture = Capture {
        backend: backend.clone(),
        row,
        values: BTreeMap::new(),
        final_logits: None,
    };
    let output = model.run_traced(&prefix, &forced, 32, &mut capture)?;
    if backend.read_indices(&output.codes)? != forced || capture.values.len() != 25 {
        return Err("incomplete or changed forced history".into());
    }
    let selected = select_xsum_tokens_cpu(
        capture
            .final_logits
            .as_ref()
            .ok_or("missing final logits")?,
        model.token_allowance(),
        1,
        48,
    )?;
    let views = capture
        .values
        .iter()
        .map(|(name, (shape, bytes))| {
            Ok((
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            ))
        })
        .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
    let serialized = safetensors::tensor::serialize(views, None)?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[5])?
        .write_all(&serialized)?;
    println!(
        "{}",
        serde_json::json!({"batch":32,"row_slot":row,"forced_codes":200,
        "selected_ids":selected,"tensors":capture.values.keys().collect::<Vec<_>>() })
    );
    Ok(())
}
