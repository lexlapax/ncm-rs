//! Synchronized WMT14 Metal generation and memory for one fixed batch.
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::{Artifact, sha256};
use ncm_model::{Wmt14HostOutput, Wmt14Teacher};
use std::{
    error::Error,
    fs,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn allocator_counters() -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(serde_json::json!({
        "active": memory::active_memory()?,
        "peak": memory::peak_memory()?,
        "cache": memory::cache_memory()?,
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [
        _,
        artifact,
        prefixes,
        mode,
        batch,
        warmups,
        trials,
        extras @ ..,
    ] = args.as_slice()
    else {
        return Err(
            "usage: wmt14_perf ARTIFACT PREFIXES 0|1 BATCH WARMUPS TRIALS [reusable] [deferred-head] [start=N] [materialize-after=N] [decoder-rows=N] [cache-mib=N] [allocator-trace] [fused-rope-batch8] [drop-logits|release-outputs]".into(),
        );
    };
    let mut reusable = false;
    let mut deferred_head = false;
    let mut start_row = None;
    let mut materialize_after = None;
    let mut decoder_rows = None;
    let mut cache_mib = None;
    let mut allocator_trace = false;
    let mut fused_rope_batch8 = false;
    let mut drop_logits = false;
    let mut release_outputs = false;
    for opt in extras {
        match opt.as_str() {
            "reusable" if !reusable => reusable = true,
            "deferred-head" if !deferred_head => deferred_head = true,
            "allocator-trace" if !allocator_trace => allocator_trace = true,
            "fused-rope-batch8" if !fused_rope_batch8 => fused_rope_batch8 = true,
            "drop-logits" if !drop_logits && !release_outputs => drop_logits = true,
            "release-outputs" if !drop_logits && !release_outputs => release_outputs = true,
            opt if cache_mib.is_none() && opt.starts_with("cache-mib=") => {
                cache_mib = Some(opt[10..].parse::<usize>()?);
            }
            opt if materialize_after.is_none() && opt.starts_with("materialize-after=") => {
                materialize_after = Some(opt["materialize-after=".len()..].parse::<usize>()?);
            }
            opt if decoder_rows.is_none() && opt.starts_with("decoder-rows=") => {
                decoder_rows = Some(opt["decoder-rows=".len()..].parse::<usize>()?);
            }
            opt if start_row.is_none() && opt.starts_with("start=") => {
                start_row = Some(opt[6..].parse::<usize>()?);
            }
            _ => return Err("invalid or repeated WMT14 timing option".into()),
        }
    }
    if decoder_rows.is_some() && !drop_logits && !release_outputs {
        return Err(
            "tiled decoder does not retain logits: select drop-logits or release-outputs".into(),
        );
    }
    let start_row = start_row.unwrap_or(0);
    if cache_mib.is_some_and(|mib| mib > 16 * 1024) {
        return Err("cache limit exceeds diagnostic bound".into());
    }
    let cache_limit = cache_mib.map(|mib| mib * 1024 * 1024);
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
    let end_row = start_row.checked_add(batch).ok_or("WMT14 range overflow")?;
    let input: Vec<Vec<u32>> = rows
        .get(start_row..end_row)
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
    let previous_cache_limit = cache_limit.map(memory::set_cache_limit).transpose()?;
    let load_start = Instant::now();
    let backend = if fused_rope_batch8 {
        MlxBackend::gpu_wmt14_fused_rope_batch8_experimental()?
    } else {
        MlxBackend::gpu()?
    };
    let model = if reusable {
        let artifact = Artifact::load_wmt14_inference_reusable(artifact)?;
        Wmt14Teacher::from_reusable_artifact(backend.clone(), artifact)?
    } else {
        let artifact = Artifact::load_inference(artifact)?;
        Wmt14Teacher::from_owned_artifact(backend.clone(), artifact)?
    };
    let model = if deferred_head {
        model.with_deferred_head_check_experimental()
    } else {
        model
    };
    let model = if let Some(level) = materialize_after {
        model.with_level_materialization_experimental(level)?
    } else {
        model
    };
    let load_seconds = load_start.elapsed().as_secs_f64();
    let marker = release_outputs
        .then(|| backend.upload_f32(&[1.0], &[1]))
        .transpose()?;
    let mut samples = Vec::with_capacity(trials);
    let mut starts = Vec::with_capacity(trials);
    let mut allocator_boundaries = allocator_trace.then(|| Vec::with_capacity(trials + 1));
    let mut signature = None;
    for trial in 0..warmups + trials {
        if trial >= warmups {
            if let Some(boundaries) = &mut allocator_boundaries {
                boundaries.push(allocator_counters()?);
            }
            starts.push(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64());
        }
        let start = Instant::now();
        let generated;
        let Wmt14HostOutput { codes, ids } = if let Some(tile_rows) = decoder_rows {
            model.generate_top1_with_tiled_decoder(&prefix, batch, f64::from(guided), tile_rows)?
        } else {
            generated = if guided {
                model.generate_cfg_top1(&prefix, batch, 1.0)?
            } else {
                model.generate_top1(&prefix, batch)?
            };
            if drop_logits || release_outputs {
                drop(generated.logits);
            }
            let output = Wmt14HostOutput {
                codes: backend.read_indices(&generated.codes)?,
                ids: backend.read_indices(&generated.ids)?,
            };
            if marker.is_some() {
                drop(generated.codes);
                drop(generated.ids);
            }
            output
        };
        if let Some(marker) = &marker {
            backend.evaluate(&[backend.add(marker, marker)?])?;
        }
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
    let memory = allocator_counters()?;
    if let Some(boundaries) = &mut allocator_boundaries {
        boundaries.push(memory.clone());
    }
    println!(
        "{}",
        serde_json::json!({
            "backend": "mlx", "batch": batch, "cfg": u8::from(guided), "warmups": warmups,
            "scope": "generation and code/ID readback; model load excluded",
            "load_seconds": load_seconds, "reusable_loader": reusable,
            "deferred_head": deferred_head, "start": start_row,
            "materialize_after": materialize_after, "decoder_rows": decoder_rows,
            "drop_logits": drop_logits || release_outputs,
            "release_outputs": release_outputs, "cache_limit_bytes": cache_limit,
            "prior_cache_limit_bytes": previous_cache_limit,
            "trial_start_unix_seconds": starts, "trials_seconds": samples, "codes": codes, "raw_ids": ids,
            "mlx_memory": memory,
            "allocator_boundaries": allocator_boundaries,
            "fused_rope_batch8": fused_rope_batch8
        }),
    );
    Ok(())
}
