//! Verify copied artifact tensor upload/readback byte-for-byte on CPU or CUDA.
use candle_core::{DType, Device};
use ncm_io::{Artifact, sha256};
use std::{fs, io::Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: artifact_probe <cpu|cuda> <bundle-dir> <new-report.json>".into());
    }
    let device = match args[0].as_str() {
        "cpu" => Device::Cpu,
        "cuda" if cfg!(feature = "cuda") => Device::new_cuda(0)?,
        _ => return Err("unsupported or unbuilt device; no fallback".into()),
    };
    let artifact = Artifact::load(&args[1])?;
    let mut counts = std::collections::BTreeMap::new();
    for (file, records) in &artifact.manifest().tensors {
        let tensors = candle_core::safetensors::load_buffer(artifact.file(file)?, &device)?;
        for record in records {
            let flat = tensors[&record.name].flatten_all()?;
            let raw: Vec<u8> = match flat.dtype() {
                DType::F32 => flat
                    .to_vec1::<f32>()?
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
                DType::I64 => flat
                    .to_vec1::<i64>()?
                    .into_iter()
                    .flat_map(i64::to_le_bytes)
                    .collect(),
                dtype => return Err(format!("unsupported dtype: {dtype:?}").into()),
            };
            if sha256(&raw) != record.sha256 {
                return Err(format!("readback mismatch: {file}/{}", record.name).into());
            }
        }
        counts.insert(file, records.len());
    }
    device.synchronize()?;
    let report = serde_json::json!({"schema_version": 1, "device": args[0], "runtime": "candle-0.11.1",
        "profile": artifact.manifest().profile, "upstream_commit": artifact.manifest().upstream_commit,
        "manifest_sha256": sha256(&fs::read(std::path::Path::new(&args[1]).join("manifest.json"))?),
        "bundle_tensor_counts": counts, "byte_exact_readback": true,
        "model_quality": null, "performance": null});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    writeln!(file)?;
    println!(
        "artifact {}: byte-exact tensor upload/readback passed",
        args[0]
    );
    Ok(())
}
