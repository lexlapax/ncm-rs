//! Checked shared codec construction/readback; not a forward parity benchmark.
use ncm_backend_cuda::CandleBackend;
use ncm_io::Artifact;
use ncm_model::Codec;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: checked_codec_load ARTIFACT cpu|cuda".into());
    }
    let backend = match args[2].as_str() {
        "cpu" => CandleBackend::cpu(),
        "cuda" => CandleBackend::cuda(0)?,
        _ => return Err("unsupported device; choose cpu or cuda".into()),
    };
    let artifact = Artifact::load(&args[1])?;
    let codec = Codec::from_artifact(backend, &artifact)?;
    let records = &artifact.manifest().tensors["codec.safetensors"];
    let mut checked_bytes = 0;
    for (name, tensor) in codec.weights() {
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        let record = records
            .iter()
            .find(|r| &r.name == name)
            .ok_or("missing manifest record")?;
        if ncm_io::sha256(&bytes) != record.sha256 {
            return Err(format!("readback mismatch: {name}").into());
        }
        checked_bytes += bytes.len();
    }
    println!(
        "{}",
        serde_json::json!({"device": args[2], "tensors": codec.weights().len(), "checked_bytes": checked_bytes,
        "code_length": codec.config().code_length(), "forward_parity": "pending"})
    );
    Ok(())
}
