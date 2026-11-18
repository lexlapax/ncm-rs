//! Synchronized WMT14 CUDA default-pool measurements.
use candle_core::cuda_backend::cudarc::driver::sys;
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Wmt14HostOutput, Wmt14Teacher};
use serde_json::{Value, json};
use std::{error::Error, fs, time::Instant};

#[path = "support/cuda_pool.rs"]
mod cuda_pool;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact, prefixes, batch, cfg, extras @ ..] = args.as_slice() else {
        return Err("usage: wmt14_fast_argmax_memory ARTIFACT PREFIXES BATCH CFG [deferred-head] [decoder-rows=N]".into());
    };
    let mut deferred_head = false;
    let mut decoder_rows = None;
    for option in extras {
        match option.as_str() {
            "deferred-head" if !deferred_head => deferred_head = true,
            option if decoder_rows.is_none() && option.starts_with("decoder-rows=") => {
                decoder_rows = Some(option["decoder-rows=".len()..].parse::<usize>()?);
            }
            _ => return Err("invalid or repeated WMT14 memory option".into()),
        }
    }
    if decoder_rows.is_some_and(|rows| !(1..=32).contains(&rows)) {
        return Err("WMT14 decoder tile must use 1..=32 rows".into());
    }
    let batch: usize = batch.parse()?;
    let cfg: u32 = cfg.parse()?;
    if ![1, 4, 16, 32].contains(&batch) || cfg > 1 || (cfg == 1 && batch > 16) {
        return Err("unsupported batch or CFG".into());
    }
    if sha256(&fs::read(format!("{artifact}/manifest.json"))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected artifact manifest".into());
    }
    let bytes = fs::read(prefixes)?;
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
    let load_start = Instant::now();
    let artifact = Artifact::load_inference(artifact)?;
    let backend = CandleBackend::cuda_wmt14_fast_argmax_experimental(0)?;
    let model = Wmt14Teacher::from_artifact(backend.clone(), &artifact)?;
    let model = if deferred_head {
        model.with_deferred_head_check_experimental()
    } else {
        model
    };
    backend.device().synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let after_load = cuda_pool::snapshot(&backend)?;
    let mut live_pool = None;
    let mut dropped_pool = None;
    let mut samples = Vec::new();
    let mut device_samples = Vec::new();
    let mut signature = None;
    for trial in 0..4 {
        backend.device().synchronize()?;
        let stream = backend.device().as_cuda_device()?.cuda_stream();
        let device_start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let start = Instant::now();
        let output = if decoder_rows.is_some() {
            None
        } else if cfg == 0 {
            Some(model.generate_top1(&input, batch)?)
        } else {
            Some(model.generate_cfg_top1(&input, batch, 1.0)?)
        };
        let (Wmt14HostOutput { codes, ids }, device_end) = if let Some(tile_rows) = decoder_rows {
            let selected =
                model.generate_top1_with_tiled_decoder(&input, batch, f64::from(cfg), tile_rows)?;
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            (selected, end)
        } else {
            let output = output.as_ref().ok_or("missing full projection output")?;
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            let codes = backend.read_indices(&output.codes)?;
            let ids = backend.read_indices(&output.ids)?;
            (Wmt14HostOutput { codes, ids }, end)
        };
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
        if trial == 3 {
            if output.is_some() {
                live_pool = Some(cuda_pool::snapshot(&backend)?);
            }
            drop(output);
            backend.device().synchronize()?;
            dropped_pool = Some(cuda_pool::snapshot(&backend)?);
        }
        if trial >= 3 {
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
        json!({"scope":"generation and codes/IDs host readback; model load excluded","batch":batch,"cfg":cfg,"deferred_head":deferred_head,"decoder_rows":decoder_rows,"warmups":3,"load_seconds":load_seconds,"cuda_pool_after_load":after_load,"cuda_pool_live_output":live_pool,"cuda_pool_after_output_drop":dropped_pool,"pool_scope":"default CUDA async pool only; high water includes load, current after drop retains model; excludes other allocations","trials_seconds":samples,"device_event_seconds":device_samples,"peak_rss_kib":rss_kib,"outputs":{"codes":code_rows,"ids":id_rows}})
    );
    Ok(())
}
