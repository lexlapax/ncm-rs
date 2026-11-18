//! Synchronized WMT14 Metal generation and memory for one fixed batch.
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Wmt14Teacher;
use std::{error::Error, fs, path::Path, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact, prefixes, mode, batch, warmups, trials] = args.as_slice() else {
        return Err("usage: wmt14_perf ARTIFACT PREFIXES 0|1 BATCH WARMUPS TRIALS".into());
    };
    let guided = match mode.as_str() {
        "0" => false,
        "1" => true,
        _ => return Err("invalid CFG mode".into()),
    };
    let batch: usize = batch.parse()?;
    let warmups: usize = warmups.parse()?;
    let trials: usize = trials.parse()?;
    if !(1..=if guided { 16 } else { 32 }).contains(&batch)
        || !(1..=100).contains(&warmups)
        || !(1..=100).contains(&trials)
    {
        return Err("invalid WMT14 batch, warmup or trial count".into());
    }
    let raw = fs::read(prefixes)?;
    if sha256(&raw) != "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f"
        || sha256(&fs::read(Path::new(artifact).join("manifest.json"))?)
            != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected WMT14 fixture or artifact".into());
    }
    let doc: serde_json::Value = serde_json::from_slice(&raw)?;
    let rows = doc["prefixes"].as_array().ok_or("missing WMT14 inputs")?;
    let input: Vec<Vec<u32>> = rows
        .get(..batch)
        .ok_or("not enough WMT14 prefixes")?
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<serde_json::Result<_>>()?;
    if input.iter().any(|row| row.len() != 96) {
        return Err("unexpected WMT14 prefix length".into());
    }
    let prefix: Vec<_> = input.into_iter().flatten().collect();
    drop(doc);
    drop(raw);
    let load_start = Instant::now();
    let backend = MlxBackend::gpu()?;
    let artifact = Artifact::load_inference(artifact)?;
    let model = Wmt14Teacher::from_owned_artifact(backend.clone(), artifact)?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let mut samples = Vec::with_capacity(trials);
    let mut signature = None;
    for trial in 0..warmups + trials {
        let start = Instant::now();
        let generated = if guided {
            model.generate_cfg_top1(&prefix, batch, 1.0)?
        } else {
            model.generate_top1(&prefix, batch)?
        };
        let codes = backend.read_indices(&generated.codes)?;
        let ids = backend.read_indices(&generated.ids)?;
        let elapsed = start.elapsed().as_secs_f64();
        if codes.len() != batch * 313 || ids.len() != batch * 64 {
            return Err("unexpected WMT14 output shape".into());
        }
        let current = (codes, ids);
        if signature.as_ref().is_some_and(|prior| prior != &current) {
            return Err("WMT14 output changed between trials".into());
        }
        signature = Some(current);
        if trial >= warmups {
            samples.push(elapsed);
        }
    }
    let (codes, ids) = signature.ok_or("no WMT14 trials")?;
    println!(
        "{}",
        serde_json::json!({
            "backend": "mlx", "batch": batch, "cfg": u8::from(guided), "warmups": warmups,
            "scope": "generation and code/ID readback; model load excluded",
            "load_seconds": load_seconds, "trials_seconds": samples, "codes": codes, "raw_ids": ids,
            "mlx_memory": {"active": memory::active_memory()?, "peak": memory::peak_memory()?,
                           "cache": memory::cache_memory()?}
        }),
    );
    Ok(())
}
