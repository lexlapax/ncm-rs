//! JSON-array Sudoku inference on an explicitly selected backend.
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
use ncm_core::GeneratorBackend;
use ncm_io::Artifact;
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
use ncm_model::{Codec, Generator};
use std::{error::Error, fs, io::Read, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn run<B: GeneratorBackend + Clone>(
    backend: B,
    command: &str,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    batch_size: usize,
) -> Result<serde_json::Value> {
    if inputs.is_empty()
        || inputs.len() > 2000
        || !(1..=128).contains(&batch_size)
        || inputs.iter().any(|r| r.len() != inputs[0].len())
    {
        return Err(
            "inputs must be 1..=2000 equal-length token arrays; batch size must be 1..=128".into(),
        );
    }
    let mut codes = Vec::with_capacity(inputs.len());
    let mut ids = Vec::with_capacity(inputs.len());
    let mut append = |batch: usize, flat_codes: Vec<u32>, flat_ids: Vec<u32>| -> Result<()> {
        if flat_codes.is_empty()
            || flat_ids.is_empty()
            || !flat_codes.len().is_multiple_of(batch)
            || !flat_ids.len().is_multiple_of(batch)
        {
            return Err("invalid backend output shape".into());
        }
        codes.extend(
            flat_codes
                .chunks_exact(flat_codes.len() / batch)
                .map(<[u32]>::to_vec),
        );
        ids.extend(
            flat_ids
                .chunks_exact(flat_ids.len() / batch)
                .map(<[u32]>::to_vec),
        );
        Ok(())
    };
    match command {
        "generate" => {
            let model = Generator::from_artifact(backend.clone(), &artifact)?;
            drop(artifact);
            for rows in inputs.chunks(batch_size) {
                let flat: Vec<u32> = rows.iter().flatten().copied().collect();
                let generated = model.generate(&flat, rows.len())?;
                append(
                    rows.len(),
                    backend.read_indices(&generated.codes)?,
                    backend.read_indices(&generated.ids)?,
                )?;
            }
        }
        "reconstruct" => {
            let model = Codec::from_artifact(backend.clone(), &artifact)?;
            drop(artifact);
            for rows in inputs.chunks(batch_size) {
                let flat: Vec<u32> = rows.iter().flatten().copied().collect();
                let (encoded, decoded) = model.reconstruct(&flat, rows.len())?;
                append(
                    rows.len(),
                    backend.read_indices(&encoded.codes)?,
                    backend.read_indices(&decoded.ids)?,
                )?;
            }
        }
        _ => return Err("unsupported command".into()),
    }
    Ok(serde_json::json!({"codes": codes, "ids": ids}))
}
fn dispatch(
    device: &str,
    command: &str,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    batch_size: usize,
) -> Result<serde_json::Value> {
    let _ = (command, &artifact, inputs, batch_size);
    match device {
        #[cfg(feature = "cpu")]
        "cpu" => run(
            ncm_backend_cuda::CandleBackend::cpu(),
            command,
            artifact,
            inputs,
            batch_size,
        ),
        #[cfg(feature = "cuda")]
        "cuda" => run(
            ncm_backend_cuda::CandleBackend::cuda(0)?,
            command,
            artifact,
            inputs,
            batch_size,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => run(
            ncm_backend_mlx::MlxBackend::gpu()?,
            command,
            artifact,
            inputs,
            batch_size,
        ),
        _ => Err("requested device is unavailable in this build; no fallback".into()),
    }
}
fn main_impl() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 3 && args[1] == "inspect" {
        let artifact = Artifact::load(&args[2])?;
        println!("{}", serde_json::to_string_pretty(artifact.manifest())?);
        return Ok(());
    }
    if !(5..=6).contains(&args.len()) || !matches!(args[1].as_str(), "generate" | "reconstruct") {
        return Err("usage: ncm inspect ARTIFACT | ncm generate|reconstruct ARTIFACT cpu|cuda|mlx INPUT_JSON [BATCH_SIZE]".into());
    }
    let artifact = Artifact::load(&args[2])?;
    let input_path = Path::new(&args[4]);
    let file = fs::File::open(input_path)?;
    if !file.metadata()?.is_file() {
        return Err("input must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err("input exceeds 1 MiB".into());
    }
    let inputs: Vec<Vec<u32>> = serde_json::from_slice(&bytes)?;
    let batch_size = args.get(5).map_or(Ok(128), |s| s.parse::<usize>())?;
    let result = dispatch(&args[3], &args[1], artifact, &inputs, batch_size)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn main() {
    if let Err(e) = main_impl() {
        eprintln!("ncm: {e}");
        std::process::exit(1);
    }
}
