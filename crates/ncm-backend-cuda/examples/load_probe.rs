//! Phase 1 copied canonical load/upload timings; not model inference.
use candle_core::{Device, Tensor};
use ncm_io::{Artifact, sha256};
use serde_json::json;
use std::{collections::HashMap, fs, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn load(root: &str, device: &Device) -> Result<(Artifact, Vec<HashMap<String, Tensor>>)> {
    let artifact = Artifact::load(root)?;
    let tensors = artifact
        .manifest()
        .tensors
        .keys()
        .map(|name| {
            candle_core::safetensors::load_buffer(artifact.file(name)?, device).map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    device.synchronize()?;
    Ok((artifact, tensors))
}
fn used(device: &Device) -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        let (free, total) = device
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .mem_get_info()?;
        Ok(total - free)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Err("CUDA build required".into())
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || cfg!(debug_assertions) {
        return Err("usage (release CUDA build): load_probe <bundle> <new-report.json>".into());
    }
    let device = Device::new_cuda(0)?;
    device.synchronize()?;
    let baseline = used(&device)?;
    let start = Instant::now();
    let loaded = load(&args[0], &device)?;
    let first_ms = start.elapsed().as_secs_f64() * 1000.;
    let resident = used(&device)?;
    let tensor_bytes: usize = loaded
        .0
        .manifest()
        .tensors
        .values()
        .flatten()
        .map(|r| r.bytes)
        .sum();
    let tensor_count: usize = loaded.0.manifest().tensors.values().map(Vec::len).sum();
    drop(loaded);
    device.synchronize()?;
    let after_drop = used(&device)?;
    for _ in 0..5 {
        drop(load(&args[0], &device)?);
    }
    let mut trials = Vec::new();
    for _ in 0..30 {
        device.synchronize()?;
        let start = Instant::now();
        let loaded = load(&args[0], &device)?;
        trials.push(start.elapsed().as_secs_f64() * 1000.);
        drop(loaded);
    }
    device.synchronize()?;
    let report = json!({"schema_version": 1, "runtime": "candle-0.11.1", "device": "cuda",
        "scope": "read + validate file/tensor SHA256 + copied canonical load + upload + synchronize; destruction excluded",
        "first_load_ms": first_ms, "trials_ms": trials, "warmups": 5,
        "tensor_count": tensor_count, "tensor_bytes": tensor_bytes,
        "device_used_bytes": {"before": baseline, "loaded": resident, "after_drop": after_drop, "after_trials": used(&device)?},
        "device_memory_scope": "global driver snapshots; not process allocation peaks",
        "host_proc_status": fs::read_to_string("/proc/self/status")?.lines().filter(|line| line.starts_with("VmHWM:") || line.starts_with("VmRSS:")).collect::<Vec<_>>(),
        "manifest_sha256": sha256(&fs::read(std::path::Path::new(&args[0]).join("manifest.json"))?),
        "source_sha256": {"probe": sha256(include_bytes!("load_probe.rs")), "loader": sha256(include_bytes!("../../ncm-io/src/lib.rs")), "lock": sha256(include_bytes!("../../../Cargo.lock"))},
        "cold_disk_cache": false, "context_initialization_timed": false, "model_quality": null});
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[1])?,
        &report,
    )?;
    Ok(())
}
