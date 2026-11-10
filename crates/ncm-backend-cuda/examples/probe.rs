//! Phase 1 operator parity and synchronized wall-clock microprobe.

#[path = "../tests/support/mod.rs"]
mod support;

use std::{fs, process::Command, time::Instant};

use candle_core::Device;
use serde_json::json;
use sha2::{Digest, Sha256};
use support::{Fixture, Prepared, Result, check_output};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 || !matches!(args[0].as_str(), "cpu" | "cuda") {
        return Err("usage: probe <cpu|cuda> <fixture.json> <new-output.json>".into());
    }
    if cfg!(debug_assertions) {
        return Err("timing probes must be built with --release".into());
    }
    let bytes = fs::read(&args[1])?;
    let fixture = Fixture::parse(&bytes)?;
    let device = match args[0].as_str() {
        "cpu" => Device::Cpu,
        "cuda" if cfg!(feature = "cuda") => Device::new_cuda(0)?,
        _ => return Err("CUDA requested but the cuda feature is not built".into()),
    };
    #[cfg(feature = "cuda")]
    candle_core::cuda_backend::set_gemm_reduced_precision_f32(false);
    let mut results = Vec::new();
    let mut passed = true;
    for case in &fixture.cases {
        let prepared = Prepared::new(case, &device)?;
        device.synchronize()?;
        let start = Instant::now();
        let output = prepared.execute()?;
        device.synchronize()?;
        let first_ms = start.elapsed().as_secs_f64() * 1000.0;
        let errors = check_output(&output, &case.expected)?;
        passed &= errors.passed;
        drop(output);
        for _ in 0..5 {
            let output = prepared.execute()?;
            device.synchronize()?;
            drop(output);
        }
        let mut trials = Vec::with_capacity(30);
        for _ in 0..30 {
            device.synchronize()?;
            let start = Instant::now();
            let output = prepared.execute()?;
            device.synchronize()?;
            trials.push(start.elapsed().as_secs_f64() * 1000.0);
            drop(output);
        }
        results.push(json!({"name": case.name, "errors": errors,
            "first_execution_ms": first_ms, "trials_ms": trials}));
    }
    let git = |args: &[&str]| -> String {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .unwrap_or_else(|| "unavailable".into())
    };
    let sources: [(&str, &[u8]); 6] = [
        ("lib.rs", include_bytes!("../src/lib.rs")),
        ("Cargo.toml", include_bytes!("../Cargo.toml")),
        ("ops.rs", include_bytes!("../src/ops.rs")),
        ("probe.rs", include_bytes!("probe.rs")),
        ("support.rs", include_bytes!("../tests/support/mod.rs")),
        ("Cargo.lock", include_bytes!("../../../Cargo.lock")),
    ];
    let hashes: std::collections::BTreeMap<_, _> = sources
        .into_iter()
        .map(|(name, content)| (name, format!("{:x}", Sha256::digest(content))))
        .collect();
    let report = json!({
        "schema_version": 1, "runtime": "candle-0.11.1", "device": args[0],
        "git_commit": git(&["rev-parse", "HEAD"]),
        "working_tree_status": git(&["status", "--porcelain"]),
        "compiled_source_sha256": hashes,
        "fixture_sha256": format!("{:x}", Sha256::digest(&bytes)),
        "upstream_commit": fixture.upstream_commit, "oracle_torch": fixture.torch_version,
        "dtype": "float32", "tf32": false, "warmups": 5, "trials": 30,
        "scope": "operator dispatch + device synchronization; uploads/readback excluded",
        "atol": support::ATOL, "rtol": support::RTOL,
        "model_quality": null, "peak_memory": null,
        "passed": passed, "results": results,
    });
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    serde_json::to_writer_pretty(file, &report)?;
    if !passed {
        return Err("operator parity failed; see report".into());
    }
    println!(
        "{}: {} operator cases passed; report {}",
        args[0],
        fixture.cases.len(),
        args[2]
    );
    Ok(())
}
