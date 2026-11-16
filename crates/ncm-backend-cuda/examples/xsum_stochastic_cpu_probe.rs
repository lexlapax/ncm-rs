//! Compare seeded XSum CPU- and device-final selection on strict CUDA.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::XsumTeacher;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: xsum_stochastic_cpu_probe ARTIFACT INPUT_JSON".into());
    }
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if rows.is_empty() || rows.len() > 128 || rows.iter().any(|r| r.len() != 512) {
        return Err("expected 1..=128 padded prefixes".into());
    }
    let batch = rows.len();
    let input: Vec<u32> = rows.into_iter().flatten().collect();
    let backend = CandleBackend::cuda_cudnn(0)?;
    let artifact = Artifact::load_inference(&args[1])?;
    let model = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let device = model.sample_seeded_device(&input, batch, 0.0, 42)?;
    let cpu = model.sample_seeded_cpu(&input, batch, 42)?;
    let device_codes = backend.read_indices(&device.codes)?;
    let cpu_codes = backend.read_indices(&cpu.codes)?;
    let code_differences = device_codes
        .iter()
        .zip(&cpu_codes)
        .filter(|(a, b)| a != b)
        .count();
    let id_differences = device
        .ids
        .iter()
        .zip(&cpu.ids)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "{}",
        serde_json::json!({"batch": batch, "seed": 42,
        "code_differences": code_differences, "id_differences": id_differences,
        "codes": cpu_codes, "ids": cpu.ids})
    );
    if code_differences != 0 || id_differences != 0 {
        return Err("seeded CPU- and device-final outputs differ".into());
    }
    Ok(())
}
