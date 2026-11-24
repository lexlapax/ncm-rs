//! Native Rust `ncm quantize` command and automatic verified pre-quantized sidecar loading.

use ncm_core::ComputePrecision;
use ncm_io::Artifact;
#[cfg(any(
    feature = "cuda",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
use ncm_io::{
    PrequantKind, PrequantWriteEntry, SafetensorsDtype as Dtype, TensorDtype,
    f32_bytes_to_bf16_bytes, write_prequant_sidecar,
};
use std::{
    error::Error,
    path::{Path, PathBuf},
};

pub(crate) const QUANTIZE_HELP: &str = "\
Usage: ncm quantize ARTIFACT --precision q8|q4 --device cuda|mlx [--force]

Pre-quantize a verified NCM artifact into a verified sidecar cache stored beside
manifest.json (`prequant-{ggml|mlx}-{q8|q4}.{safetensors,json}`):

  --precision q8|q4    Target quantization precision (required).
  --device cuda|mlx    Target backend format: `cuda` writes GGML Q8_0/Q4_0;
                       `mlx` writes group-64 affine MLX tensors (required).
  --force              Overwrite an existing sidecar for the same format/precision.

Parent FP32 weights and manifest.json are never modified. Subsequent `ncm sample`,
`ncm run`, and `ncm serve` invocations with `--precision q8|q4` automatically load
the matching verified sidecar (failing closed on any manifest or SHA-256 mismatch).
";

type CliResult<T> = std::result::Result<T, Box<dyn Error>>;

/// Load an inference artifact, automatically using a verified pre-quantized sidecar
/// when `precision` is `q8` or `q4` and a sidecar exists for `device`.
pub(crate) fn load_artifact_for_device(
    path: impl AsRef<Path>,
    device: &str,
    precision: ComputePrecision,
) -> ncm_io::Result<Artifact> {
    if precision.is_quantized() {
        let format = match device {
            "cuda" | "cuda-cudnn" => Some("ggml"),
            "mlx" => Some("mlx"),
            _ => None,
        };
        if let Some(fmt) = format {
            return Artifact::load_inference_with_prequant(path, fmt, precision.as_str());
        }
    }
    Artifact::load_inference_streaming(path)
}

#[derive(Debug, PartialEq, Eq)]
struct QuantizeArgs {
    artifact: PathBuf,
    precision: ComputePrecision,
    device: String,
    force: bool,
}

fn parse_quantize_args(args: &[String]) -> CliResult<QuantizeArgs> {
    if args.len() < 3 {
        return Err(
            "usage: ncm quantize ARTIFACT --precision q8|q4 --device cuda|mlx [--force]".into(),
        );
    }
    let artifact = PathBuf::from(&args[2]);
    let mut precision: Option<ComputePrecision> = None;
    let mut device: Option<String> = None;
    let mut force = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--precision" => {
                let raw = args
                    .get(i + 1)
                    .ok_or("missing value for --precision (expected q8 or q4)")?;
                let parsed: ComputePrecision = raw.parse()?;
                if !parsed.is_quantized() {
                    return Err(format!(
                        "ncm quantize requires --precision q8 or q4, got {parsed}"
                    )
                    .into());
                }
                precision = Some(parsed);
                i += 2;
            }
            "--device" => {
                let raw = args
                    .get(i + 1)
                    .ok_or("missing value for --device (expected cuda or mlx)")?;
                if !matches!(raw.as_str(), "cuda" | "mlx") {
                    return Err(
                        format!("ncm quantize requires --device cuda or mlx, got {raw:?}").into(),
                    );
                }
                device = Some(raw.clone());
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            other => {
                return Err(format!("unknown ncm quantize option: {other}").into());
            }
        }
    }
    let precision = precision.ok_or("ncm quantize requires --precision q8|q4")?;
    let device = device.ok_or("ncm quantize requires --device cuda|mlx")?;
    Ok(QuantizeArgs {
        artifact,
        precision,
        device,
        force,
    })
}

#[cfg(any(
    feature = "cuda",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn bytes_to_f32_vec(data: &[u8]) -> CliResult<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        return Err("invalid F32 tensor payload length".into());
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect())
}

#[cfg(any(
    feature = "cuda",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn keep_f32_verbatim(name: &str) -> bool {
    name == "attn_mask_for_masking" || name.ends_with(".log_inv_T")
}

pub(crate) fn run_quantize(args: &[String]) -> CliResult<()> {
    let parsed = parse_quantize_args(args)?;
    #[cfg(not(any(
        feature = "cuda",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    )))]
    {
        Err(format!(
            "ncm quantize --device {} requires a GPU build (--features cuda or --features mlx)",
            parsed.device
        )
        .into())
    }
    #[cfg(any(
        feature = "cuda",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    run_quantize_gpu(&parsed)
}

#[cfg(any(
    feature = "cuda",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn run_quantize_gpu(parsed: &QuantizeArgs) -> CliResult<()> {
    let started = std::time::Instant::now();
    let artifact = Artifact::load_inference_streaming(&parsed.artifact)?;

    let mut parent_weight_bytes = 0u64;
    for bundle in ["codec.safetensors", "generator.safetensors"] {
        if let Ok(bytes) = artifact.file_bytes(bundle) {
            parent_weight_bytes = parent_weight_bytes.saturating_add(bytes);
        }
    }

    let (format, entries) = match parsed.device.as_str() {
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                // Validate real CUDA device capability before quantizing.
                let _backend =
                    ncm_backend_cuda::CandleBackend::cuda(0)?.with_precision(parsed.precision)?;
                let mut entries = Vec::new();
                for bundle in ["codec.safetensors", "generator.safetensors"] {
                    if artifact.bundle_tensors(bundle).is_err() {
                        continue;
                    }
                    artifact.visit_bundle_tensors(
                        bundle,
                        |e| Box::new(e) as Box<dyn Error>,
                        |name, shape, dtype, data| {
                            let key = format!("{bundle}/{name}");
                            match dtype {
                                TensorDtype::I64 => {
                                    entries.push(PrequantWriteEntry {
                                        bundle: bundle.to_owned(),
                                        name: name.to_owned(),
                                        shape: shape.to_vec(),
                                        kind: PrequantKind::I64,
                                        parts: vec![(
                                            key,
                                            Dtype::I64,
                                            shape.to_vec(),
                                            data.to_vec(),
                                        )],
                                    });
                                }
                                TensorDtype::F32 if keep_f32_verbatim(name) => {
                                    entries.push(PrequantWriteEntry {
                                        bundle: bundle.to_owned(),
                                        name: name.to_owned(),
                                        shape: shape.to_vec(),
                                        kind: PrequantKind::F32,
                                        parts: vec![(
                                            key,
                                            Dtype::F32,
                                            shape.to_vec(),
                                            data.to_vec(),
                                        )],
                                    });
                                }
                                TensorDtype::F32 => {
                                    if ncm_model::is_linear_projection_weight(name, shape) {
                                        let vals = bytes_to_f32_vec(data)?;
                                        if let Some(qbytes) =
                                            ncm_backend_cuda::quantize_linear_ggml_bytes(
                                                &vals,
                                                shape,
                                                parsed.precision,
                                            )?
                                        {
                                            let kind = match parsed.precision {
                                                ComputePrecision::Q8 => PrequantKind::GgmlQ8_0,
                                                ComputePrecision::Q4 => PrequantKind::GgmlQ4_0,
                                                _ => unreachable!(),
                                            };
                                            let qlen = qbytes.len();
                                            entries.push(PrequantWriteEntry {
                                                bundle: bundle.to_owned(),
                                                name: name.to_owned(),
                                                shape: shape.to_vec(),
                                                kind,
                                                parts: vec![(key, Dtype::U8, vec![qlen], qbytes)],
                                            });
                                            return Ok(());
                                        }
                                    }
                                    let bf16 = f32_bytes_to_bf16_bytes(data)?;
                                    entries.push(PrequantWriteEntry {
                                        bundle: bundle.to_owned(),
                                        name: name.to_owned(),
                                        shape: shape.to_vec(),
                                        kind: PrequantKind::Bf16,
                                        parts: vec![(key, Dtype::BF16, shape.to_vec(), bf16)],
                                    });
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
                ("ggml", entries)
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(
                    "ncm quantize --device cuda requires a binary built with --features cuda"
                        .into(),
                );
            }
        }
        "mlx" => {
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            {
                let backend =
                    ncm_backend_mlx::MlxBackend::gpu()?.with_precision(parsed.precision)?;
                let mut entries = Vec::new();
                for bundle in ["codec.safetensors", "generator.safetensors"] {
                    if artifact.bundle_tensors(bundle).is_err() {
                        continue;
                    }
                    artifact.visit_bundle_tensors(
                        bundle,
                        |e| Box::new(e) as Box<dyn Error>,
                        |name, shape, dtype, data| {
                            let key = format!("{bundle}/{name}");
                            match dtype {
                                TensorDtype::I64 => {
                                    entries.push(PrequantWriteEntry {
                                        bundle: bundle.to_owned(),
                                        name: name.to_owned(),
                                        shape: shape.to_vec(),
                                        kind: PrequantKind::I64,
                                        parts: vec![(
                                            key,
                                            Dtype::I64,
                                            shape.to_vec(),
                                            data.to_vec(),
                                        )],
                                    });
                                }
                                TensorDtype::F32 if keep_f32_verbatim(name) => {
                                    entries.push(PrequantWriteEntry {
                                        bundle: bundle.to_owned(),
                                        name: name.to_owned(),
                                        shape: shape.to_vec(),
                                        kind: PrequantKind::F32,
                                        parts: vec![(
                                            key,
                                            Dtype::F32,
                                            shape.to_vec(),
                                            data.to_vec(),
                                        )],
                                    });
                                }
                                TensorDtype::F32 => {
                                    if ncm_model::is_linear_projection_weight(name, shape) {
                                        let vals = bytes_to_f32_vec(data)?;
                                        if let Some(raw) =
                                            backend.quantize_linear_raw(&vals, shape)?
                                        {
                                            let kind = match parsed.precision {
                                                ComputePrecision::Q8 => PrequantKind::MlxQ8G64,
                                                ComputePrecision::Q4 => PrequantKind::MlxQ4G64,
                                                _ => unreachable!(),
                                            };
                                            entries.push(PrequantWriteEntry {
                                                bundle: bundle.to_owned(),
                                                name: name.to_owned(),
                                                shape: shape.to_vec(),
                                                kind,
                                                parts: vec![
                                                    (
                                                        format!("{key}.weights"),
                                                        Dtype::U32,
                                                        raw.weights_shape,
                                                        raw.weights_u32,
                                                    ),
                                                    (
                                                        format!("{key}.scales"),
                                                        Dtype::BF16,
                                                        raw.scales_shape,
                                                        raw.scales_bf16,
                                                    ),
                                                    (
                                                        format!("{key}.biases"),
                                                        Dtype::BF16,
                                                        raw.biases_shape,
                                                        raw.biases_bf16,
                                                    ),
                                                ],
                                            });
                                            return Ok(());
                                        }
                                    }
                                    let bf16 = f32_bytes_to_bf16_bytes(data)?;
                                    entries.push(PrequantWriteEntry {
                                        bundle: bundle.to_owned(),
                                        name: name.to_owned(),
                                        shape: shape.to_vec(),
                                        kind: PrequantKind::Bf16,
                                        parts: vec![(key, Dtype::BF16, shape.to_vec(), bf16)],
                                    });
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
                ("mlx", entries)
            }
            #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
            {
                return Err(
                    "ncm quantize --device mlx requires Apple Silicon built with --features mlx"
                        .into(),
                );
            }
        }
        _ => unreachable!(),
    };

    let summary = write_prequant_sidecar(
        &parsed.artifact,
        format,
        parsed.precision.as_str(),
        entries,
        parsed.force,
    )?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let out = serde_json::json!({
        "artifact": parsed.artifact.display().to_string(),
        "format": summary.format,
        "precision": summary.precision,
        "manifest_path": summary.manifest_path.display().to_string(),
        "weights_path": summary.weights_path.display().to_string(),
        "manifest_sha256": summary.manifest_sha256,
        "parent_weight_bytes": parent_weight_bytes,
        "sidecar_bytes": summary.file_bytes,
        "compression_ratio": if summary.file_bytes > 0 {
            (parent_weight_bytes as f64 / summary.file_bytes as f64 * 100.0).round() / 100.0
        } else { 0.0 },
        "file_sha256": summary.file_sha256,
        "quantized_tensors": summary.quantized_tensors,
        "total_tensors": summary.total_tensors,
        "elapsed_ms": (elapsed_ms * 10.0).round() / 10.0,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|v| (*v).to_owned()).collect()
    }

    #[test]
    fn quantize_arg_parser_validates_precision_and_device() {
        let ok = parse_quantize_args(&s(&[
            "ncm",
            "quantize",
            "/tmp/model",
            "--precision",
            "q8",
            "--device",
            "cuda",
        ]))
        .unwrap();
        assert_eq!(
            ok,
            QuantizeArgs {
                artifact: PathBuf::from("/tmp/model"),
                precision: ComputePrecision::Q8,
                device: "cuda".into(),
                force: false,
            }
        );

        let ok_force = parse_quantize_args(&s(&[
            "ncm",
            "quantize",
            "/tmp/model",
            "--precision",
            "q4",
            "--device",
            "mlx",
            "--force",
        ]))
        .unwrap();
        assert!(ok_force.force);
        assert_eq!(ok_force.precision, ComputePrecision::Q4);

        // Rejects non-quantized precisions and invalid devices.
        assert!(
            parse_quantize_args(&s(&[
                "ncm",
                "quantize",
                "/tmp/model",
                "--precision",
                "bf16",
                "--device",
                "cuda",
            ]))
            .is_err()
        );
        assert!(
            parse_quantize_args(&s(&[
                "ncm",
                "quantize",
                "/tmp/model",
                "--precision",
                "q8",
                "--device",
                "cpu",
            ]))
            .is_err()
        );
        assert!(parse_quantize_args(&s(&["ncm", "quantize", "/tmp/model"])).is_err());
    }
}
