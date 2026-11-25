//! JSON-array inference on an explicitly selected backend.
mod decode;
mod doctor;
mod help;
mod lm1b;
mod paths;
mod pull;
mod quantize;
mod sample;
mod serve;
mod train;

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
use ncm_model::{Codec, Generator, LocalNcmModel, TinyStoriesTeacher};
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

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(crate) fn load_sample_adapter<B: GeneratorBackend>(
    backend: &B,
    base_manifest_sha256: &str,
    adapter_path: Option<&Path>,
    n_layer: usize,
    n_embd: usize,
) -> Result<Option<ncm_model::RuntimeLoraAdapter<B::Tensor>>> {
    let Some(path) = adapter_path else {
        return Ok(None);
    };
    let loaded = ncm_io::LoraAdapterArtifact::load(path, Some(base_manifest_sha256))?;
    Ok(Some(ncm_model::RuntimeLoraAdapter::from_artifact(
        backend, &loaded, n_layer, n_embd,
    )?))
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
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    let prefixes = tinystories_prefixes(inputs)?;
    let base_sha = artifact.manifest_sha256().to_owned();
    let mut model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let (n_layer, n_embd) = model.lora_dimensions();
    if let Some(adapter) = load_sample_adapter(&backend, &base_sha, adapter_path, n_layer, n_embd)?
    {
        model.set_lora_adapter(Some(adapter))?;
    }
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
    precision: ncm_core::ComputePrecision,
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    let _ = (&artifact, inputs, seed, precision, adapter_path);
    match device {
        #[cfg(feature = "cpu")]
        "cpu" => sample_tinystories(
            ncm_backend_cuda::CandleBackend::cpu().with_precision(precision)?,
            artifact,
            inputs,
            seed,
            adapter_path,
        ),
        #[cfg(feature = "cuda")]
        "cuda" => sample_tinystories(
            ncm_backend_cuda::CandleBackend::cuda(0)?.with_precision(precision)?,
            artifact,
            inputs,
            seed,
            adapter_path,
        ),
        #[cfg(feature = "cuda-cudnn")]
        "cuda-cudnn" => sample_tinystories(
            ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?.with_precision(precision)?,
            artifact,
            inputs,
            seed,
            adapter_path,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => sample_tinystories(
            ncm_backend_mlx::MlxBackend::gpu()?.with_precision(precision)?,
            artifact,
            inputs,
            seed,
            adapter_path,
        ),
        _ => Err("requested device is unavailable in this build; no fallback".into()),
    }
}

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn sample_local<B: GeneratorBackend + Clone>(
    backend: B,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    seed: Option<u64>,
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    if inputs.is_empty() || inputs.len() > 128 || inputs.iter().any(|r| r.len() != inputs[0].len())
    {
        return Err("local-ncm-v1 expects 1..=128 equal-length prefix token rows".into());
    }
    let base_sha = artifact.manifest_sha256().to_owned();
    let mut model = LocalNcmModel::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let (n_layer, n_embd) = model.lora_dimensions();
    if let Some(adapter) = load_sample_adapter(&backend, &base_sha, adapter_path, n_layer, n_embd)?
    {
        model.set_lora_adapter(Some(adapter))?;
    }
    let prefix_len = model.config().prefix_len;
    if inputs[0].len() != prefix_len {
        return Err(format!(
            "local-ncm-v1 expects prefix length {prefix_len}, got {}",
            inputs[0].len()
        )
        .into());
    }
    let flat: Vec<u32> = inputs.iter().flatten().copied().collect();
    let generated = model.generate(&flat, inputs.len(), seed)?;
    drop(generated.logits);
    let flat_codes = backend.read_indices(&generated.codes)?;
    let flat_ids = backend.read_indices(&generated.ids)?;
    let code_len = model.codec_config().code_length();
    let context_len = model.codec_config().raw().context_length;
    if flat_codes.len() != inputs.len() * code_len || flat_ids.len() != inputs.len() * context_len {
        return Err("invalid local-ncm-v1 backend output shape".into());
    }
    let codes: Vec<Vec<u32>> = flat_codes
        .chunks_exact(code_len)
        .map(<[u32]>::to_vec)
        .collect();
    let ids: Vec<Vec<u32>> = flat_ids
        .chunks_exact(context_len)
        .map(<[u32]>::to_vec)
        .collect();
    Ok(match seed {
        Some(seed) => serde_json::json!({"seed": seed, "codes": codes, "ids": ids}),
        None => serde_json::json!({"codes": codes, "ids": ids}),
    })
}

fn dispatch_local(
    device: &str,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    seed: Option<u64>,
    precision: ncm_core::ComputePrecision,
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    let _ = (&artifact, inputs, seed, precision, adapter_path);
    match device {
        #[cfg(feature = "cpu")]
        "cpu" => sample_local(
            ncm_backend_cuda::CandleBackend::cpu().with_precision(precision)?,
            artifact,
            inputs,
            seed,
            adapter_path,
        ),
        #[cfg(feature = "cuda")]
        "cuda" => sample_local(
            ncm_backend_cuda::CandleBackend::cuda(0)?.with_precision(precision)?,
            artifact,
            inputs,
            seed,
            adapter_path,
        ),
        #[cfg(feature = "cuda-cudnn")]
        "cuda-cudnn" => sample_local(
            ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?.with_precision(precision)?,
            artifact,
            inputs,
            seed,
            adapter_path,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => sample_local(
            ncm_backend_mlx::MlxBackend::gpu()?.with_precision(precision)?,
            artifact,
            inputs,
            seed,
            adapter_path,
        ),
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
    precision: ncm_core::ComputePrecision,
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    let prefixes = xsum_prefixes(inputs)?;
    let base_sha = artifact.manifest_sha256().to_owned();
    let backend = ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?.with_precision(precision)?;
    let mut model =
        XsumTeacher::from_owned_artifact(backend.clone(), artifact)?.with_prefix_cache(true);
    let (n_layer, n_embd) = model.lora_dimensions();
    if let Some(adapter) = load_sample_adapter(&backend, &base_sha, adapter_path, n_layer, n_embd)?
    {
        model.set_lora_adapter(Some(adapter))?;
    }
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
    precision: ncm_core::ComputePrecision,
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    let prefixes = xsum_prefixes(inputs)?;
    let base_sha = artifact.manifest_sha256().to_owned();
    let backend = ncm_backend_mlx::MlxBackend::gpu()?.with_precision(precision)?;
    let mut model =
        XsumTeacher::from_owned_artifact(backend.clone(), artifact)?.with_prefix_cache(true);
    let (n_layer, n_embd) = model.lora_dimensions();
    if let Some(adapter) = load_sample_adapter(&backend, &base_sha, adapter_path, n_layer, n_embd)?
    {
        model.set_lora_adapter(Some(adapter))?;
    }
    let generated = if let Some(seed) = seed {
        model.sample_seeded_cfg_cpu(&prefixes, inputs.len(), cfg_scale, seed)?
    } else {
        model.generate_top1_cfg_cpu(&prefixes, inputs.len(), cfg_scale)?
    };
    xsum_output(&backend, generated, inputs.len())
}

fn dispatch_xsum(
    device: &str,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    cfg_scale: f64,
    seed: Option<u64>,
    precision: ncm_core::ComputePrecision,
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    let _ = (&artifact, inputs, cfg_scale, seed, precision, adapter_path);
    match device {
        #[cfg(feature = "cuda-cudnn")]
        "cuda-cudnn" => sample_xsum(artifact, inputs, cfg_scale, seed, precision, adapter_path),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => sample_xsum_mlx(artifact, inputs, cfg_scale, seed, precision, adapter_path),
        _ => Err("XSum requested device is unavailable in this build; no fallback".into()),
    }
}

#[cfg(all(
    feature = "wmt14-experimental",
    any(
        feature = "cuda-cudnn",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    )
))]
fn sample_wmt14<B: ncm_core::GeneratorBackend + Clone>(
    backend: B,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    scale: f64,
    metal_optimized: bool,
    padded_retry: bool,
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    if !scale.is_finite() || scale < 0.0 || !(scale as f32).is_finite() {
        return Err("invalid WMT14 CFG scale".into());
    }
    let max_batch = if padded_retry {
        4
    } else if scale == 0.0 {
        32
    } else {
        16
    };
    if !(1..=max_batch).contains(&inputs.len()) || inputs.iter().any(|row| row.len() != 96) {
        return Err("invalid WMT14 batch size or prefix length".into());
    }
    let prefixes: Vec<_> = inputs.iter().flatten().copied().collect();
    let base_sha = artifact.manifest_sha256().to_owned();
    let mut teacher = if metal_optimized {
        ncm_model::Wmt14Teacher::from_owned_artifact(backend.clone(), artifact)?
            .with_prefix_cache(scale == 0.0)
            .with_deferred_head_check_experimental()
    } else {
        ncm_model::Wmt14Teacher::from_owned_artifact(backend.clone(), artifact)?
            .with_prefix_cache(scale == 0.0)
    };
    let (n_layer, n_embd) = teacher.lora_dimensions();
    if let Some(adapter) = load_sample_adapter(&backend, &base_sha, adapter_path, n_layer, n_embd)?
    {
        teacher.set_lora_adapter(Some(adapter))?;
    }
    let batch = inputs.len();
    let (output, retried) = if metal_optimized && (scale == 0.0 || scale == 1.0) {
        let teacher = teacher.with_level_materialization_experimental(5)?;
        let teacher = if scale == 1.0 && batch == 4 {
            teacher.with_last_transformer_checkpoint_experimental(3)?
        } else {
            teacher
        };
        if padded_retry {
            teacher.generate_top1_tiled_decoder_with_padded_retry(&prefixes, batch, scale, 2)?
        } else {
            (
                teacher.generate_top1_with_tiled_decoder(&prefixes, batch, scale, 2)?,
                false,
            )
        }
    } else {
        let (output, retried) = if padded_retry {
            teacher.generate_top1_with_padded_retry(&prefixes, batch, scale)?
        } else if scale == 0.0 {
            (teacher.generate_top1(&prefixes, batch)?, false)
        } else {
            (teacher.generate_cfg_top1(&prefixes, batch, scale)?, false)
        };
        drop(output.logits);
        (
            ncm_model::Wmt14HostOutput {
                codes: backend.read_indices(&output.codes)?,
                ids: backend.read_indices(&output.ids)?,
            },
            retried,
        )
    };
    if retried {
        eprintln!("WMT14 retried the rejected head with eight internal rows");
    }
    let ncm_model::Wmt14HostOutput { codes, ids } = output;
    if codes.len() != inputs.len() * 313 || ids.len() != inputs.len() * 64 {
        return Err("invalid WMT14 backend output shape".into());
    }
    let codes: Vec<_> = codes
        .as_chunks::<313>()
        .0
        .iter()
        .map(<[u32; 313]>::as_slice)
        .collect();
    let ids: Vec<_> = ids
        .as_chunks::<64>()
        .0
        .iter()
        .map(<[u32; 64]>::as_slice)
        .collect();
    Ok(serde_json::json!({"codes": codes, "ids": ids}))
}

fn dispatch_wmt14(
    device: &str,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    scale: f64,
    precision: ncm_core::ComputePrecision,
    adapter_path: Option<&Path>,
) -> Result<serde_json::Value> {
    let _ = (&artifact, inputs, scale, precision, adapter_path);
    match device {
        #[cfg(all(feature = "wmt14-experimental", feature = "cuda-cudnn"))]
        "cuda" => sample_wmt14(
            ncm_backend_cuda::CandleBackend::cuda_wmt14_fast_argmax_experimental(0)?
                .with_precision(precision)?,
            artifact,
            inputs,
            scale,
            false,
            false,
            adapter_path,
        ),
        #[cfg(all(
            feature = "wmt14-experimental",
            feature = "mlx",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        "mlx" | "mlx-retry-padded" => sample_wmt14(
            ncm_backend_mlx::MlxBackend::gpu()?.with_precision(precision)?,
            artifact,
            inputs,
            scale,
            true,
            device == "mlx-retry-padded",
            adapter_path,
        ),
        _ => Err("experimental WMT14 requested device is unavailable in this build".into()),
    }
}

fn dispatch(
    device: &str,
    command: &str,
    artifact: Artifact,
    inputs: &[Vec<u32>],
    batch_size: usize,
    precision: ncm_core::ComputePrecision,
) -> Result<serde_json::Value> {
    let _ = (command, &artifact, inputs, batch_size, precision);
    match device {
        #[cfg(feature = "cpu")]
        "cpu" => run(
            ncm_backend_cuda::CandleBackend::cpu().with_precision(precision)?,
            command,
            artifact,
            inputs,
            batch_size,
        ),
        #[cfg(feature = "cuda")]
        "cuda" => run(
            ncm_backend_cuda::CandleBackend::cuda(0)?.with_precision(precision)?,
            command,
            artifact,
            inputs,
            batch_size,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => run(
            ncm_backend_mlx::MlxBackend::gpu()?.with_precision(precision)?,
            command,
            artifact,
            inputs,
            batch_size,
        ),
        _ => Err("requested device is unavailable in this build; no fallback".into()),
    }
}
fn read_input_bytes(input_path: &Path) -> Result<Vec<u8>> {
    let file = fs::File::open(input_path)?;
    if !file.metadata()?.is_file() {
        return Err("input must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err("input exceeds 1 MiB".into());
    }
    Ok(bytes)
}

fn read_inputs(input_path: &Path) -> Result<Vec<Vec<u32>>> {
    Ok(serde_json::from_slice(&read_input_bytes(input_path)?)?)
}

fn main_impl() -> Result<()> {
    let mut args: Vec<_> = std::env::args().collect();
    if let Some(message) = help::requested(&args) {
        print!("{message}");
        return Ok(());
    }
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V" || args[1] == "version") {
        println!("{}", doctor::version_string());
        return Ok(());
    }
    if args.get(1).is_some_and(|arg| arg == "doctor") {
        return doctor::run_doctor(&args);
    }
    if args.get(1).is_some_and(|arg| arg == "list") {
        return serve::run_list(&args);
    }
    if args.get(1).is_some_and(|arg| arg == "pull") {
        return serve::run_pull(&args);
    }
    if args.get(1).is_some_and(|arg| arg == "run") {
        return serve::run_repl(&args);
    }
    if args.get(1).is_some_and(|arg| arg == "serve") {
        return serve::run_serve(&args);
    }
    if args.get(1).is_some_and(|arg| arg == "train") {
        return train::run_train(&args);
    }
    if args.get(1).is_some_and(|arg| arg == "quantize") {
        return quantize::run_quantize(&args);
    }
    let unified = args.get(1).is_some_and(|arg| arg == "sample");
    let inline_prompt = if unified {
        let (normalized, prompt) = sample::extract_inline_prompt(&args)?;
        args = normalized;
        prompt
    } else {
        None
    };
    let plain = unified && args.iter().skip(5).any(|arg| arg == "--plain");
    let decode = unified && (plain || args.iter().skip(5).any(|arg| arg == "--decode"));
    let text = unified && args.iter().skip(5).any(|arg| arg == "--text");
    let (precision, adapter_path) = if unified {
        let mut parsed = ncm_core::ComputePrecision::Fp32;
        let mut adapter: Option<std::path::PathBuf> = None;
        let mut i = 5;
        while i < args.len() {
            if args[i] == "--precision" {
                let raw = args.get(i + 1).ok_or(
                    "missing value for --precision (expected fp32, bf16, fp16, q8, or q4)",
                )?;
                parsed = raw.parse()?;
                i += 2;
            } else if args[i] == "--adapter" {
                let raw = args
                    .get(i + 1)
                    .ok_or("missing value for --adapter (expected adapter directory)")?;
                adapter = Some(std::path::PathBuf::from(raw));
                i += 2;
            } else {
                i += 1;
            }
        }
        (parsed, adapter)
    } else {
        (ncm_core::ComputePrecision::Fp32, None)
    };
    if unified {
        args = sample::translate(&args)?;
    } else if args.get(1).is_some_and(|arg| arg == "sample-xsum") {
        sample::validate_xsum_alias(&mut args)?;
    } else if args.get(1).is_some_and(|arg| arg == "sample-wmt14") {
        sample::validate_wmt14_alias(&mut args)?;
    }
    if args.len() == 3 && args[1] == "inspect" {
        let artifact = Artifact::load_inference_streaming(&args[2])?;
        println!("{}", serde_json::to_string_pretty(artifact.manifest())?);
        return Ok(());
    }
    if args
        .get(1)
        .is_some_and(|arg| arg == "sample-lm1b" || arg == "sample-lm1b-small")
    {
        let result = lm1b::sample(
            &args,
            text,
            inline_prompt.as_deref(),
            precision,
            adapter_path.as_deref(),
        )?;
        print_output(&result, plain)?;
        return Ok(());
    }
    if !(5..=7).contains(&args.len())
        || !matches!(
            args[1].as_str(),
            "generate"
                | "reconstruct"
                | "sample-tinystories"
                | "sample-xsum"
                | "sample-wmt14"
                | "sample-local"
        )
        || (args[1] == "sample-tinystories" && args.len() != 6)
        || (args.len() == 7 && args[1] != "sample-xsum")
    {
        return Err("invalid command or arguments; run `ncm --help`".into());
    }
    let sample_seed = if args[1] == "sample-tinystories" {
        Some(args[5].parse::<u64>()?)
    } else {
        None
    };
    let artifact = quantize::load_artifact_for_device(&args[2], &args[3], precision)?;
    let decoder = if decode || text {
        Some(decode::Decoder::from_artifact(&artifact)?)
    } else {
        None
    };
    let inputs = if let Some(prompt) = inline_prompt {
        decoder
            .as_ref()
            .ok_or("missing tokenizer")?
            .encode_texts(&[prompt])?
    } else if text {
        let texts: Vec<String> = serde_json::from_slice(&read_input_bytes(Path::new(&args[4]))?)?;
        decoder
            .as_ref()
            .ok_or("missing tokenizer")?
            .encode_texts(&texts)?
    } else {
        read_inputs(Path::new(&args[4]))?
    };
    let mut result = if args[1] == "sample-xsum" {
        let cfg_scale = args.get(5).map_or(Ok(0.0), |s| s.parse::<f64>())?;
        let seed = args.get(6).map(|s| s.parse::<u64>()).transpose()?;
        dispatch_xsum(
            &args[3],
            artifact,
            &inputs,
            cfg_scale,
            seed,
            precision,
            adapter_path.as_deref(),
        )?
    } else if args[1] == "sample-wmt14" {
        let scale = args.get(5).map_or(Ok(0.0), |s| s.parse::<f64>())?;
        dispatch_wmt14(
            &args[3],
            artifact,
            &inputs,
            scale,
            precision,
            adapter_path.as_deref(),
        )?
    } else if args[1] == "sample-local" {
        let seed = args.get(5).map(|s| s.parse::<u64>()).transpose()?;
        dispatch_local(
            &args[3],
            artifact,
            &inputs,
            seed,
            precision,
            adapter_path.as_deref(),
        )?
    } else if let Some(seed) = sample_seed {
        dispatch_tinystories(
            &args[3],
            artifact,
            &inputs,
            seed,
            precision,
            adapter_path.as_deref(),
        )?
    } else {
        let batch_size = args.get(5).map_or(Ok(128), |s| s.parse::<usize>())?;
        dispatch(&args[3], &args[1], artifact, &inputs, batch_size, precision)?
    };
    if decode {
        decoder.ok_or("missing tokenizer")?.add_texts(&mut result)?;
    }
    print_output(&result, plain)?;
    Ok(())
}

fn print_output(result: &serde_json::Value, plain: bool) -> Result<()> {
    if plain {
        let texts = result
            .get("texts")
            .and_then(serde_json::Value::as_array)
            .ok_or("missing decoded texts for --plain output")?;
        for entry in texts {
            println!("{}", entry.as_str().unwrap_or_default());
        }
    } else {
        println!("{}", serde_json::to_string(result)?);
    }
    Ok(())
}
fn main() {
    if let Err(e) = main_impl() {
        eprintln!("ncm: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod feature_tests {
    #[test]
    fn ordinary_gpu_features_enable_qualified_profiles() {
        const {
            assert!(!cfg!(feature = "cuda") || cfg!(feature = "wmt14-experimental"));
            assert!(!cfg!(feature = "cuda-cudnn") || cfg!(feature = "lm1b-experimental"));
            assert!(
                !cfg!(feature = "mlx")
                    || (cfg!(feature = "wmt14-experimental")
                        && cfg!(feature = "lm1b-mlx-experimental"))
            );
        }
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
