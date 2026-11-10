//! Real-Metal operator parity and forced-evaluation microprobe.
#[path = "../tests/support/mod.rs"]
mod support;
use mlx_rs::{Device, memory, with_device};
use ncm_io::sha256;
use serde_json::json;
use std::{fs, time::Instant};
use support::{Fixture, Prepared, Result, check_output};
fn main() -> Result<()> {
    with_device(Device::gpu(), run)
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || cfg!(debug_assertions) {
        return Err("usage (release): mlx_probe <fixture.json> <new-report.json>".into());
    }
    let bytes = fs::read(&args[0])?;
    let fixture = Fixture::parse(&bytes)?;
    let mut results = Vec::new();
    let mut passed = true;
    memory::reset_peak_memory()?;
    for case in &fixture.cases {
        let prepared = Prepared::new(case)?;
        let start = Instant::now();
        let output = prepared.execute()?;
        output.eval()?;
        let first = start.elapsed().as_secs_f64() * 1000.;
        let errors = check_output(&output, &case.expected)?;
        passed &= errors.passed;
        drop(output);
        for _ in 0..5 {
            let output = prepared.execute()?;
            output.eval()?;
        }
        let mut trials = Vec::new();
        for _ in 0..30 {
            let start = Instant::now();
            let output = prepared.execute()?;
            output.eval()?;
            trials.push(start.elapsed().as_secs_f64() * 1000.);
        }
        results.push(
            json!({"name":case.name,"errors":errors,"first_execution_ms":first,"trials_ms":trials}),
        );
    }
    let report = json!({"schema_version":1,"runtime":"mlx-rs-0.34.0/mlx-0.32.2","device":"metal",
        "fixture_sha256":sha256(&bytes),"upstream_commit":fixture.upstream_commit,"oracle_torch":fixture.torch_version,
        "dtype":"float32","warmups":5,"trials":30,"atol":support::ATOL,"rtol":support::RTOL,
        "scope":"graph construction + synchronous eval; uploads/readback excluded; FP16 cast explicitly materialized",
        "peak_memory_bytes":memory::peak_memory()?,"active_memory_bytes":memory::active_memory()?,"cache_memory_bytes":memory::cache_memory()?,
        "source_sha256":{"probe":sha256(include_bytes!("mlx_probe.rs")),"ops":sha256(include_bytes!("../src/ops.rs")),"support":sha256(include_bytes!("../tests/support/mod.rs")),"contract":sha256(include_bytes!("../../../tests/support/operators.rs")),"lock":sha256(include_bytes!("../../../Cargo.lock"))},
        "model_quality":null,"passed":passed,"results":results});
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[1])?,
        &report,
    )?;
    if !passed {
        return Err("MLX operator parity failed".into());
    }
    println!("Metal: {} cases passed", fixture.cases.len());
    Ok(())
}
