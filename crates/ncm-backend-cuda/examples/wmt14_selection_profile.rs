//! Synchronized WMT14 final vocabulary selection diagnostic.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Wmt14Teacher;
use serde_json::{Value, json};
use std::{error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: wmt14_selection_profile ARTIFACT PREFIXES BATCH CFG generic|candidate".into(),
        );
    }
    if !["generic", "candidate"].contains(&args[5].as_str()) {
        return Err("invalid selector".into());
    }
    let batch: usize = args[3].parse()?;
    let cfg: u32 = args[4].parse()?;
    if ![1, 4, 16, 32].contains(&batch) || cfg > 1 || (cfg == 1 && batch > 16) {
        return Err("unsupported workload".into());
    }
    if sha256(&fs::read(format!("{}/manifest.json", args[1]))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected artifact manifest".into());
    }
    let data = fs::read(&args[2])?;
    if sha256(&data) != "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f" {
        return Err("unexpected prefix input".into());
    }
    let doc: Value = serde_json::from_slice(&data)?;
    let rows = doc["prefixes"].as_array().ok_or("missing prefixes")?;
    if rows.len() != 3003 {
        return Err("invalid prefix count".into());
    }
    let mut prefix = Vec::with_capacity(batch * 96);
    for row in &rows[..batch] {
        let row = row.as_array().ok_or("invalid row")?;
        if row.len() != 96 {
            return Err("invalid row width".into());
        }
        for value in row {
            let id = u32::try_from(value.as_u64().ok_or("invalid ID")?)?;
            if id >= 151680 {
                return Err("ID out of range".into());
            }
            prefix.push(id);
        }
    }
    let b = CandleBackend::cuda_wmt14_reference(0)?;
    let model = Wmt14Teacher::from_artifact(b.clone(), &Artifact::load_inference(&args[1])?)?;
    let output = if cfg == 0 {
        model.generate_top1(&prefix, batch)?
    } else {
        model.generate_cfg_top1(&prefix, batch, 1.0)?
    };
    let expected = b.read_indices(&output.ids)?;
    let mut times = Vec::new();
    for run in 0..25 {
        b.evaluate(&[])?;
        let start = Instant::now();
        let selected = if args[5] == "generic" {
            b.select_last(&output.logits, true)?
        } else {
            ncm_backend_cuda::wmt14_argmax::select(&b, &output.logits)?
        };
        b.evaluate(&[])?;
        let elapsed = start.elapsed().as_secs_f64();
        if b.read_indices(&selected)? != expected {
            return Err("selection changed model output".into());
        }
        if run >= 5 {
            times.push(elapsed);
        }
    }
    println!(
        "{}",
        json!({"batch":batch,"cfg":cfg,"warmups":5,"samples_seconds":times,
        "selector":args[5], "scope":"first-tie selection only, synchronized; model and logits construction excluded",
        "ids":expected})
    );
    Ok(())
}
