//! JSON-array inference on an explicitly selected backend.
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
use ncm_core::GeneratorBackend;
use ncm_io::Artifact;
#[cfg(any(
    feature = "cuda-cudnn",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
use ncm_model::XsumTeacher;
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
use ncm_model::{Codec, Generator, TinyStoriesTeacher};
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
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn tinystories_prefixes(inputs: &[Vec<u32>]) -> Result<Vec<u32>> {
    if inputs.is_empty() || inputs.len() > 128 || inputs.iter().any(|row| row.len() != 1) {
        return Err("TinyStories expects 1..=128 single-token prefixes".into());
    }
    Ok(inputs.iter().map(|row| row[0]).collect())
}

/// Generate one seeded TinyStories batch and return host-owned codes and IDs.
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn sample_tinystories<B: GeneratorBackend + Clone>(
    backend: B,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    seed: u64,
) -> Result<serde_json::Value> {
    let prefixes = tinystories_prefixes(inputs)?;
    let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let generated = model.sample_seeded(&prefixes, inputs.len(), seed)?;
    drop(generated.logits);
    let codes = backend.read_indices(&generated.codes)?;
    let ids = backend.read_indices(&generated.ids)?;
    if codes.len() != inputs.len() * 641 || ids.len() != inputs.len() * 128 {
        return Err("invalid TinyStories backend output shape".into());
    }
    let codes: Vec<_> = codes
        .as_chunks::<641>()
        .0
        .iter()
        .map(<[u32; 641]>::as_slice)
        .collect();
    let ids: Vec<_> = ids
        .as_chunks::<128>()
        .0
        .iter()
        .map(<[u32; 128]>::as_slice)
        .collect();
    Ok(serde_json::json!({"seed": seed, "codes": codes, "ids": ids}))
}

fn dispatch_tinystories(
    device: &str,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    seed: u64,
) -> Result<serde_json::Value> {
    let _ = (&artifact, inputs, seed);
    match device {
        #[cfg(feature = "cpu")]
        "cpu" => sample_tinystories(
            ncm_backend_cuda::CandleBackend::cpu(),
            artifact,
            inputs,
            seed,
        ),
        #[cfg(feature = "cuda")]
        "cuda" => sample_tinystories(
            ncm_backend_cuda::CandleBackend::cuda(0)?,
            artifact,
            inputs,
            seed,
        ),
        #[cfg(feature = "cuda-cudnn")]
        "cuda-cudnn" => sample_tinystories(
            ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?,
            artifact,
            inputs,
            seed,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => sample_tinystories(ncm_backend_mlx::MlxBackend::gpu()?, artifact, inputs, seed),
        _ => Err("requested device is unavailable in this build; no fallback".into()),
    }
}

#[cfg(any(
    feature = "cuda-cudnn",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn xsum_prefixes(inputs: &[Vec<u32>]) -> Result<Vec<u32>> {
    if inputs.is_empty()
        || inputs.len() > 128
        || inputs.iter().any(|row| {
            row.len() != 512
                || row.iter().any(|&token| token >= 50304)
                || row.iter().all(|&token| token == 50257)
        })
    {
        return Err("XSum expects 1..=128 valid 512-token padded prefixes".into());
    }
    Ok(inputs.iter().flatten().copied().collect())
}

#[cfg(feature = "cuda-cudnn")]
fn sample_xsum(
    artifact: Artifact,
    inputs: &[Vec<u32>],
    cfg_scale: f64,
    seed: Option<u64>,
) -> Result<serde_json::Value> {
    let prefixes = xsum_prefixes(inputs)?;
    let backend = ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?;
    let model = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let generated = if let Some(seed) = seed {
        model.sample_seeded_device(&prefixes, inputs.len(), cfg_scale, seed)?
    } else {
        model.generate_top1_cfg_device(&prefixes, inputs.len(), cfg_scale)?
    };
    xsum_output(&backend, generated, inputs.len())
}

#[cfg(any(
    feature = "cuda-cudnn",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn xsum_output<B: ncm_core::GeneratorBackend>(
    backend: &B,
    generated: ncm_model::XsumGenerated<B>,
    batch: usize,
) -> Result<serde_json::Value> {
    let codes = backend.read_indices(&generated.codes)?;
    if codes.len() != batch * 200 || generated.ids.len() != batch * 48 {
        return Err("invalid XSum backend output shape".into());
    }
    let codes: Vec<_> = codes
        .as_chunks::<200>()
        .0
        .iter()
        .map(<[u32; 200]>::as_slice)
        .collect();
    let ids: Vec<_> = generated
        .ids
        .as_chunks::<48>()
        .0
        .iter()
        .map(<[u32; 48]>::as_slice)
        .collect();
    Ok(serde_json::json!({"codes": codes, "ids": ids}))
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn sample_xsum_mlx(
    artifact: Artifact,
    inputs: &[Vec<u32>],
    cfg_scale: f64,
    seed: Option<u64>,
) -> Result<serde_json::Value> {
    if cfg_scale != 0.0 {
        return Err("XSum Metal currently supports only CFG 0".into());
    }
    let prefixes = xsum_prefixes(inputs)?;
    let backend = ncm_backend_mlx::MlxBackend::gpu()?;
    let model = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let generated = if let Some(seed) = seed {
        model.sample_seeded_cpu(&prefixes, inputs.len(), seed)?
    } else {
        model.generate_top1_cpu(&prefixes, inputs.len())?
    };
    xsum_output(&backend, generated, inputs.len())
}

fn dispatch_xsum(
    device: &str,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    cfg_scale: f64,
    seed: Option<u64>,
) -> Result<serde_json::Value> {
    let _ = (&artifact, inputs, cfg_scale, seed);
    match device {
        #[cfg(feature = "cuda-cudnn")]
        "cuda-cudnn" => sample_xsum(artifact, inputs, cfg_scale, seed),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => sample_xsum_mlx(artifact, inputs, cfg_scale, seed),
        _ => Err("XSum requested device is unavailable in this build; no fallback".into()),
    }
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
    if !(5..=7).contains(&args.len())
        || !matches!(
            args[1].as_str(),
            "generate" | "reconstruct" | "sample-tinystories" | "sample-xsum"
        )
        || (args[1] == "sample-tinystories" && args.len() != 6)
        || (args.len() == 7 && args[1] != "sample-xsum")
    {
        return Err("usage: ncm inspect ARTIFACT | ncm generate|reconstruct ARTIFACT cpu|cuda|mlx INPUT_JSON [BATCH_SIZE] | ncm sample-tinystories ARTIFACT cpu|cuda|cuda-cudnn|mlx INPUT_JSON SEED | ncm sample-xsum ARTIFACT cuda-cudnn|mlx INPUT_JSON [CFG_SCALE [EXPERIMENTAL_STOCHASTIC_SEED]]".into());
    }
    let sample_seed = if args[1] == "sample-tinystories" {
        Some(args[5].parse::<u64>()?)
    } else {
        None
    };
    let artifact = if sample_seed.is_some() || args[1] == "sample-xsum" {
        Artifact::load_inference(&args[2])?
    } else {
        Artifact::load(&args[2])?
    };
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
    let result = if args[1] == "sample-xsum" {
        let cfg_scale = args.get(5).map_or(Ok(0.0), |s| s.parse::<f64>())?;
        let seed = args.get(6).map(|s| s.parse::<u64>()).transpose()?;
        dispatch_xsum(&args[3], artifact, &inputs, cfg_scale, seed)?
    } else if let Some(seed) = sample_seed {
        dispatch_tinystories(&args[3], artifact, &inputs, seed)?
    } else {
        let batch_size = args.get(5).map_or(Ok(128), |s| s.parse::<usize>())?;
        dispatch(&args[3], &args[1], artifact, &inputs, batch_size)?
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn main() {
    if let Err(e) = main_impl() {
        eprintln!("ncm: {e}");
        std::process::exit(1);
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::tinystories_prefixes;

    #[test]
    fn tinystories_requires_one_token_per_bounded_batch() {
        assert!(tinystories_prefixes(&[]).is_err());
        assert!(tinystories_prefixes(&[vec![]]).is_err());
        assert!(tinystories_prefixes(&[vec![0, 2]]).is_err());
        assert!(tinystories_prefixes(&vec![vec![0]; 129]).is_err());
        assert_eq!(tinystories_prefixes(&[vec![0], vec![2]]).unwrap(), [0, 2]);
        assert_eq!(
            tinystories_prefixes(&vec![vec![0]; 128]).unwrap().len(),
            128
        );
    }
}

#[cfg(all(test, feature = "cuda-cudnn"))]
mod xsum_tests {
    use super::xsum_prefixes;

    #[test]
    fn xsum_prefixes_require_bounded_nonempty_padded_rows() {
        assert!(xsum_prefixes(&[]).is_err());
        assert!(xsum_prefixes(&[vec![0; 511]]).is_err());
        assert!(xsum_prefixes(&[vec![50257; 512]]).is_err());
        let mut valid = vec![50257; 512];
        valid[0] = 50256;
        assert_eq!(xsum_prefixes(&[valid.clone()]).unwrap(), valid);
        let mut invalid = valid.clone();
        invalid[0] = 50304;
        assert!(xsum_prefixes(&[invalid]).is_err());
        assert!(xsum_prefixes(&vec![valid; 129]).is_err());
    }
}
