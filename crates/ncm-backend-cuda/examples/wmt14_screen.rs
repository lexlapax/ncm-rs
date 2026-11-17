//! Restricted WMT14 development screen; not a quality or performance gate.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Wmt14Teacher;
use serde_json::{Value, json};
use std::{error::Error, fs, io::Write, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: wmt14_screen ARTIFACT PREFIXES OUTPUT".into());
    }
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("output must not exist".into());
    }
    if sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected artifact manifest".into());
    }
    let data = fs::read(&args[2])?;
    if sha256(&data) != "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f" {
        return Err("unexpected prefix reference".into());
    }
    let doc: Value = serde_json::from_slice(&data)?;
    let rows = doc["prefixes"].as_array().ok_or("missing prefixes")?;
    if rows.len() != 3003 {
        return Err("unexpected prefix count".into());
    }
    let mut inputs = Vec::with_capacity(32 * 96);
    for row in &rows[..32] {
        let row = row.as_array().ok_or("invalid prefix row")?;
        if row.len() != 96 {
            return Err("invalid prefix length".into());
        }
        for value in row {
            let id = u32::try_from(value.as_u64().ok_or("invalid prefix ID")?)?;
            if id >= 151680 {
                return Err("prefix ID out of range".into());
            }
            inputs.push(id);
        }
    }
    let artifact = Artifact::load_inference(&args[1])?;
    let backend = CandleBackend::cuda(0)?;
    let teacher = Wmt14Teacher::from_artifact(backend.clone(), &artifact)?;
    let mut outputs = Vec::with_capacity(32);
    for start in (0..32).step_by(4) {
        let generated = teacher.generate_top1(&inputs[start * 96..(start + 4) * 96], 4)?;
        let codes = backend.read_indices(&generated.codes)?;
        let ids = backend.read_indices(&generated.ids)?;
        if codes.len() != 4 * 313 || ids.len() != 4 * 64 {
            return Err("unexpected generated shape".into());
        }
        for row in 0..4 {
            outputs.push(json!({"row":start + row,"codes":&codes[row * 313..(row + 1) * 313],"raw_ids":&ids[row * 64..(row + 1) * 64]}));
        }
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    writeln!(
        file,
        "{}",
        serde_json::to_string_pretty(
            &json!({"scope":"32 WMT14 CUDA rows, development screen, not quality or performance","batch_size":4,"outputs":outputs})
        )?
    )?;
    println!("32 rows captured; compare with upstream separately");
    Ok(())
}
