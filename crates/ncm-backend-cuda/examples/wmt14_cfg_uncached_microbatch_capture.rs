//! Capture a bounded WMT14 CFG=1 diagnostic range; scoring is separate.
use ncm_backend_cuda::CandleBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Wmt14Teacher;
use serde_json::{Value, json};
use std::{error::Error, fs, io::Write, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: wmt14_cfg_uncached_microbatch_capture ARTIFACT PREFIXES OUTPUT START COUNT"
                .into(),
        );
    }
    let first: usize = args[4].parse()?;
    let count: usize = args[5].parse()?;
    if count == 0 || first.checked_add(count).is_none_or(|end| end > 3003) {
        return Err("invalid row range".into());
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
    let mut inputs = Vec::with_capacity(count * 96);
    for row in &rows[first..first + count] {
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
    let backend = CandleBackend::cuda_wmt14_reference(0)?;
    let teacher = Wmt14Teacher::from_artifact(backend.clone(), &artifact)?;
    let mut outputs = Vec::with_capacity(count);
    for start in (0..count).step_by(16) {
        let size = (count - start).min(16);
        let generated = teacher.generate_cfg_uncached_microbatched(
            &inputs[start * 96..(start + size) * 96],
            size,
            1.0,
        )?;
        let codes = generated.codes;
        let ids = generated.ids;
        if codes.len() != size * 313 || ids.len() != size * 64 {
            return Err("unexpected generated shape".into());
        }
        for row in 0..size {
            outputs.push(json!({"row":first + start + row,"codes":&codes[row * 313..(row + 1) * 313],"raw_ids":&ids[row * 64..(row + 1) * 64]}));
        }
        eprintln!("{}/{}", start + size, count);
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    writeln!(
        file,
        "{}",
        serde_json::to_string_pretty(
            &json!({"scope":"experimental WMT14 uncached CFG=1 microbatch diagnostic capture; scoring and performance are separate","batch_size":16,"internal_batch_size":4,"cfg_scale":1.0,"cfg_null_eos":151645,"start":first,"count":count,"outputs":outputs})
        )?
    )?;
    println!("{count} rows captured; compare with upstream separately");
    Ok(())
}
