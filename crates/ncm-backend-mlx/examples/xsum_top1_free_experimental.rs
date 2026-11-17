//! Bounded free-generation capture for explicitly selected Metal diagnostics.
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::XsumTeacher;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: xsum_top1_free_experimental ARTIFACT INPUT_JSON BACKEND".into());
    }
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if rows.len() != 32
        || rows
            .iter()
            .any(|r| r.len() != 512 || r.iter().any(|&v| v >= 50304))
    {
        return Err("expected 32 valid 512-token rows".into());
    }
    let backend = match args[3].as_str() {
        "default" => MlxBackend::gpu()?,
        "upsample" => MlxBackend::gpu_xsum_upsample_experimental()?,
        "groupnorm" => MlxBackend::gpu_xsum_groupnorm_experimental()?,
        "gelu" => MlxBackend::gpu_xsum_gelu_experimental()?,
        "compiled-gelu" => MlxBackend::gpu_xsum_compiled_gelu_experimental()?,
        "all" => MlxBackend::gpu_xsum_all_experimental()?,
        _ => return Err("unknown diagnostic backend".into()),
    };
    let input: Vec<u32> = rows.into_iter().flatten().collect();
    let artifact = Artifact::load_inference(&args[1])?;
    let model = XsumTeacher::from_owned_artifact(backend.clone(), artifact)?;
    let output = model.generate_top1_cpu(&input, 32)?;
    let codes = backend.read_indices(&output.codes)?;
    if codes.len() != 32 * 200 || output.ids.len() != 32 * 48 {
        return Err("unexpected output shape".into());
    }
    let codes: Vec<_> = codes.chunks_exact(200).collect();
    let ids: Vec<_> = output.ids.chunks_exact(48).collect();
    println!("{}", serde_json::json!({"codes": codes, "ids": ids}));
    Ok(())
}
