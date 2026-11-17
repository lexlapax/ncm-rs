//! Synchronized WMT14 level-stage diagnostic; not headline benchmark timing.
use candle_core::Tensor;
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{TraceSink, Wmt14Teacher};
use serde_json::{Value, json};
use std::{error::Error, fs, time::Instant};

struct Capture {
    backend: CandleBackend,
    previous: Instant,
    stages: Vec<(String, f64)>,
    codes: Vec<u32>,
    ids: Vec<u32>,
}
impl Capture {
    fn checkpoint(&mut self, name: &str) -> ncm_core::Result<()> {
        self.backend.evaluate(&[])?;
        let now = Instant::now();
        self.stages
            .push((name.into(), now.duration_since(self.previous).as_secs_f64()));
        self.previous = now;
        Ok(())
    }
}
impl TraceSink<CandleBackend> for Capture {
    fn tensor(&mut self, name: &str, _: &Tensor) -> ncm_core::Result<()> {
        self.checkpoint(name)
    }
    fn indices(&mut self, name: &str, value: &Tensor) -> ncm_core::Result<()> {
        match name {
            "generator.codes" => self.codes = self.backend.read_indices(value)?,
            "generator.greedy_ids" => self.ids = self.backend.read_indices(value)?,
            _ => return Ok(()),
        }
        self.checkpoint(name)
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: wmt14_stage_profile ARTIFACT PREFIXES BATCH CFG".into());
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
    let mut records = Vec::new();
    let mut signature = None;
    for run in 0..8 {
        b.evaluate(&[])?;
        let mut trace = Capture {
            backend: b.clone(),
            previous: Instant::now(),
            stages: Vec::new(),
            codes: Vec::new(),
            ids: Vec::new(),
        };
        if cfg == 0 {
            model.generate_top1_traced(&prefix, batch, &mut trace)?;
        } else {
            model.generate_cfg_top1_traced(&prefix, batch, 1.0, &mut trace)?;
        }
        if trace.codes.len() != batch * 313 || trace.ids.len() != batch * 64 {
            return Err("incomplete trace".into());
        }
        if signature
            .as_ref()
            .is_some_and(|s| s != &(trace.codes.clone(), trace.ids.clone()))
        {
            return Err("output changed across runs".into());
        }
        signature = Some((trace.codes, trace.ids));
        if run >= 3 {
            records.push(trace.stages);
        }
    }
    let (codes, ids) = signature.ok_or("missing outputs")?;
    println!(
        "{}",
        json!({"batch":batch,"cfg":cfg,"warmups":3,"records":records,"scope":"instrumented wall intervals with synchronization at every named checkpoint; excludes model load","codes":codes,"ids":ids})
    );
    Ok(())
}
