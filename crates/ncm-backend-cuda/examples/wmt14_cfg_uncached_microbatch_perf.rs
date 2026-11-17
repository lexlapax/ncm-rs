//! Synchronized WMT14 CUDA timing; quality and acceptance are separate.
use candle_core::cuda_backend::cudarc::driver::sys;
use ncm_backend_cuda::CandleBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Wmt14Teacher;
use serde_json::{Value, json};
use std::{error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: wmt14_cfg_uncached_microbatch_perf ARTIFACT PREFIXES BATCH CFG".into());
    }
    let batch: usize = args[3].parse()?;
    let cfg: u32 = args[4].parse()?;
    if ![1, 4, 16].contains(&batch) || cfg != 1 {
        return Err("unsupported batch or CFG".into());
    }
    if sha256(&fs::read(format!("{}/manifest.json", args[1]))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected artifact manifest".into());
    }
    let bytes = fs::read(&args[2])?;
    if sha256(&bytes) != "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f" {
        return Err("unexpected prefix reference".into());
    }
    let doc: Value = serde_json::from_slice(&bytes)?;
    let rows = doc["prefixes"].as_array().ok_or("missing prefixes")?;
    if rows.len() != 3003 {
        return Err("invalid prefix count".into());
    }
    let mut input = Vec::with_capacity(batch * 96);
    for row in &rows[..batch] {
        let row = row.as_array().ok_or("invalid row")?;
        if row.len() != 96 {
            return Err("invalid prefix width".into());
        }
        for x in row {
            let id = u32::try_from(x.as_u64().ok_or("invalid ID")?)?;
            if id >= 151680 {
                return Err("ID out of range".into());
            }
            input.push(id);
        }
    }
    let artifact = Artifact::load_inference(&args[1])?;
    let backend = CandleBackend::cuda_wmt14_reference(0)?;
    let model = Wmt14Teacher::from_artifact(backend.clone(), &artifact)?;
    let mut samples = Vec::new();
    let mut device_samples = Vec::new();
    let mut signature = None;
    for trial in 0..35 {
        backend.device().synchronize()?;
        let stream = backend.device().as_cuda_device()?.cuda_stream();
        let device_start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let start = Instant::now();
        let output = model.generate_cfg_uncached_microbatched(&input, batch, 1.0)?;
        let codes = output.codes;
        let ids = output.ids;
        let device_end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        backend.device().synchronize()?;
        let elapsed = start.elapsed().as_secs_f64();
        if codes.len() != batch * 313 || ids.len() != batch * 64 {
            return Err("unexpected output shape".into());
        }
        if signature
            .as_ref()
            .is_some_and(|s| s != &(codes.clone(), ids.clone()))
        {
            return Err("output changed between trials".into());
        }
        signature = Some((codes, ids));
        if trial >= 5 {
            samples.push(elapsed);
            device_samples.push(f64::from(device_start.elapsed_ms(&device_end)?) / 1000.0);
        }
    }
    let status = fs::read_to_string("/proc/self/status")?;
    let rss_kib: u64 = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .ok_or("missing peak RSS")?
        .split_whitespace()
        .next()
        .ok_or("missing peak RSS value")?
        .parse()?;
    let (codes, ids) = signature.ok_or("missing output")?;
    let code_rows: Vec<_> = codes.chunks_exact(codes.len() / batch).collect();
    let id_rows: Vec<_> = ids.chunks_exact(ids.len() / batch).collect();
    println!(
        "{}",
        json!({"scope":"experimental uncached CFG in sequential batches of up to four and codes/IDs host readback; model load excluded","batch":batch,"cfg":cfg,"warmups":5,"trials_seconds":samples,"device_event_seconds":device_samples,"peak_rss_kib":rss_kib,"outputs":{"codes":code_rows,"ids":id_rows}})
    );
    Ok(())
}
