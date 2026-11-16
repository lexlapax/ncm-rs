//! Short XSum Metal latency and memory screen; not a formal performance gate.
use mlx_rs::memory;
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::XsumTeacher;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: xsum_perf_screen ARTIFACT INPUT_JSON BATCH [STOCHASTIC_SEED]".into());
    }
    let batch: usize = args[3].parse()?;
    let seed = args.get(4).map(|s| s.parse::<u64>()).transpose()?;
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
    let mut samples = Vec::with_capacity(5);
    let mut signature = None;
    for trial in 0..7 {
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
        if trial >= 2 {
            samples.push(elapsed);
        }
    }
    let (codes, ids) = signature.ok_or("no sample")?;
    println!(
        "{}",
        serde_json::json!({"batch":batch,"warmups":2,"trials_seconds":samples,
        "seed":seed,"backend":"mlx","scope":"generation plus code and final-ID host readback; model load excluded",
        "status":"exploratory","codes":codes,"ids":ids,
        "mlx_memory":{"active":memory::active_memory()?,"peak":memory::peak_memory()?,"cache":memory::cache_memory()?}})
    );
    Ok(())
}
