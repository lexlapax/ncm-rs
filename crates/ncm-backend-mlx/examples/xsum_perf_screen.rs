//! XSum Metal latency and memory probe with explicit warmup and trial counts.
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::XsumTeacher;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=7).contains(&args.len()) {
        return Err(
            "usage: xsum_perf_screen ARTIFACT INPUT_JSON BATCH [SEED] [WARMUPS TRIALS]".into(),
        );
    }
    let batch: usize = args[3].parse()?;
    let (seed, warmups, trials) = match args.len() {
        4 => (None, 2, 5),
        5 => (Some(args[4].parse::<u64>()?), 2, 5),
        6 => (None, args[4].parse::<usize>()?, args[5].parse::<usize>()?),
        7 => (
            Some(args[4].parse::<u64>()?),
            args[5].parse::<usize>()?,
            args[6].parse::<usize>()?,
        ),
        _ => unreachable!(),
    };
    if !(1..=100).contains(&warmups) || !(1..=100).contains(&trials) {
        return Err("warmups and trials must be 1..=100".into());
    }
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if !(1..=32).contains(&batch)
        || rows.len() < batch
        || rows[..batch].iter().any(|r| r.len() != 512)
    {
        return Err("need 1..=32 padded input rows".into());
    }
    let prefix: Vec<u32> = rows[..batch].iter().flatten().copied().collect();
    let backend = MlxBackend::gpu()?;
    let artifact = Artifact::load_inference(&args[1])?;
    let model = XsumTeacher::from_owned_artifact(backend.clone(), artifact)?;
    let mut samples = Vec::with_capacity(trials);
    let mut signature = None;
    for trial in 0..warmups + trials {
        let start = Instant::now();
        let output = if let Some(seed) = seed {
            model.sample_seeded_cpu(&prefix, batch, seed)?
        } else {
            model.generate_top1_cpu(&prefix, batch)?
        };
        let codes = backend.read_indices(&output.codes)?;
        let elapsed = start.elapsed().as_secs_f64();
        if codes.len() != batch * 200 || output.ids.len() != batch * 48 {
            return Err("wrong output shape".into());
        }
        let current = (codes, output.ids);
        if signature.as_ref().is_some_and(|prior| prior != &current) {
            return Err("output changed between trials".into());
        }
        signature = Some(current);
        if trial >= warmups {
            samples.push(elapsed);
        }
    }
    let (codes, ids) = signature.ok_or("no sample")?;
    println!(
        "{}",
        serde_json::json!({"batch":batch,"warmups":warmups,"trials_seconds":samples,
        "seed":seed,"backend":"mlx","scope":"generation plus code and final-ID host readback; model load excluded",
        "status":"exploratory","codes":codes,"ids":ids,
        "mlx_memory":{"active":memory::active_memory()?,"peak":memory::peak_memory()?,"cache":memory::cache_memory()?}})
    );
    Ok(())
}
