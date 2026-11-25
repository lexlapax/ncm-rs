//! System diagnostics (`ncm doctor`) and version reporting (`ncm --version`).
use crate::{Result, paths};
use ncm_serve::ModelRegistry;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

pub(super) const DOCTOR_HELP: &str = "\
Usage: ncm doctor [OPTIONS]

Inspect the ncm binary, compiled backends, GPU/driver/library readiness,
external tools (curl), and local model/sidecar inventory.

Options:
  --models-dir DIR   Override default models directory for inventory check.
  --device DEV       Override backend device for --self-test (cuda-cudnn|cuda|mlx|cpu).
  --self-test        Run a fast zero-download operator & precision smoke test on the device.
  --json             Emit structured JSON report.
";

#[derive(Debug, Default, PartialEq, Eq)]
struct DoctorArgs {
    models_dir: Option<PathBuf>,
    device: Option<String>,
    self_test: bool,
    json: bool,
}

fn parse_doctor_args(args: &[String]) -> Result<DoctorArgs> {
    let mut cfg = DoctorArgs::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                let v = args.get(i + 1).ok_or("missing value for --models-dir")?;
                cfg.models_dir = Some(PathBuf::from(v));
                i += 2;
            }
            "--device" => {
                let v = args.get(i + 1).ok_or("missing value for --device")?;
                cfg.device = Some(v.clone());
                i += 2;
            }
            "--self-test" => {
                cfg.self_test = true;
                i += 1;
            }
            "--json" => {
                cfg.json = true;
                i += 1;
            }
            other => {
                return Err(format!("unknown option for doctor: {other}").into());
            }
        }
    }
    Ok(cfg)
}

pub(super) fn compiled_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "cpu") {
        features.push("cpu");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "cuda-cudnn") {
        features.push("cuda-cudnn");
    }
    if cfg!(all(
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    )) {
        features.push("mlx");
    }
    features
}

pub(super) fn default_runtime_device() -> &'static str {
    #[cfg(feature = "cuda-cudnn")]
    {
        "cuda-cudnn"
    }
    #[cfg(all(not(feature = "cuda-cudnn"), feature = "cuda"))]
    {
        "cuda"
    }
    #[cfg(all(
        not(feature = "cuda"),
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    {
        "mlx"
    }
    #[cfg(all(
        not(feature = "cuda"),
        not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")),
        feature = "cpu"
    ))]
    {
        "cpu"
    }
    #[cfg(all(
        not(feature = "cpu"),
        not(feature = "cuda"),
        not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))
    ))]
    {
        "none"
    }
}

pub(super) fn version_string() -> String {
    let feats = compiled_features();
    let feat_str = if feats.is_empty() {
        "none".to_owned()
    } else {
        feats.join(", ")
    };
    format!(
        "ncm {} ({}/{}; features: {})",
        env!("CARGO_PKG_VERSION"),
        env::consts::OS,
        env::consts::ARCH,
        feat_str
    )
}

fn run_cmd_stdout(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    if text.is_empty() { None } else { Some(text) }
}

fn detect_prequant_sidecars(model_dir: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(model_dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if let Some(stem) = name_str
            .strip_prefix("prequant-")
            .and_then(|s| s.strip_suffix(".manifest.json"))
        {
            found.push(stem.to_owned());
        }
    }
    found.sort();
    found
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn host_linear_ref(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += f64::from(x[r * k + j]) * f64::from(w[c * k + j]);
            }
            out[r * n + c] = acc as f32;
        }
    }
    out
}

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn host_attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    seq: usize,
    dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f64).sqrt();
    let mut out = vec![0.0f32; heads * seq * dim];
    for h in 0..heads {
        let base = h * seq * dim;
        for i in 0..seq {
            let mut scores = vec![0.0f64; seq];
            let mut max_s = f64::NEG_INFINITY;
            for j in 0..seq {
                let mut dot = 0.0f64;
                for d in 0..dim {
                    dot += f64::from(q[base + i * dim + d]) * f64::from(k[base + j * dim + d]);
                }
                let s = dot * scale;
                scores[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut denom = 0.0f64;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                denom += *s;
            }
            for d in 0..dim {
                let mut acc = 0.0f64;
                for j in 0..seq {
                    acc += (scores[j] / denom) * f64::from(v[base + j * dim + d]);
                }
                out[base + i * dim + d] = acc as f32;
            }
        }
    }
    out
}

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn run_precision_check<B: ncm_core::CodecBackend>(
    backend: &B,
    use_linear_weight: bool,
    tol_linear: f32,
    tol_attn: Option<f32>,
) -> Result<(f32, Option<f32>)> {
    let rows = 8usize;
    let dim = 64usize;
    let x_vals: Vec<f32> = (0..rows * dim)
        .map(|i| (((i * 37 + 11) % 97) as f32 - 48.0) * 0.02)
        .collect();
    let w_vals: Vec<f32> = (0..dim * dim)
        .map(|i| (((i * 29 + 7) % 89) as f32 - 44.0) * 0.02)
        .collect();
    let lin_ref = host_linear_ref(&x_vals, &w_vals, rows, dim, dim);

    let x_t = backend.upload_f32(&x_vals, &[1, rows, dim])?;
    let w_t = if use_linear_weight {
        backend.upload_linear_weight_f32(&w_vals, &[dim, dim])?
    } else {
        backend.upload_f32(&w_vals, &[dim, dim])?
    };
    let y_t = backend.project(&x_t, &w_t)?;
    backend.evaluate(std::slice::from_ref(&y_t))?;
    let y_got = backend.read_f32(&y_t)?;
    let lin_err = max_abs_diff(&y_got, &lin_ref);
    if !lin_err.is_finite() || lin_err > tol_linear {
        return Err(
            format!("linear self-test failed (max_abs={lin_err:.6}, limit={tol_linear})").into(),
        );
    }

    let attn_err = if let Some(tol) = tol_attn {
        let heads = 2usize;
        let seq = 16usize;
        let n = heads * seq * dim;
        let q_vals: Vec<f32> = (0..n)
            .map(|i| (((i * 17 + 3) % 73) as f32 - 36.0) * 0.02)
            .collect();
        let k_vals: Vec<f32> = (0..n)
            .map(|i| (((i * 23 + 5) % 79) as f32 - 39.0) * 0.02)
            .collect();
        let v_vals: Vec<f32> = (0..n)
            .map(|i| (((i * 31 + 9) % 83) as f32 - 41.0) * 0.02)
            .collect();
        let attn_ref = host_attention_ref(&q_vals, &k_vals, &v_vals, heads, seq, dim);
        let q_t = backend.upload_f32(&q_vals, &[1, heads, seq, dim])?;
        let k_t = backend.upload_f32(&k_vals, &[1, heads, seq, dim])?;
        let v_t = backend.upload_f32(&v_vals, &[1, heads, seq, dim])?;
        let a_t = backend.attention(&q_t, &k_t, &v_t, None)?;
        backend.evaluate(std::slice::from_ref(&a_t))?;
        let a_got = backend.read_f32(&a_t)?;
        let err = max_abs_diff(&a_got, &attn_ref);
        if !err.is_finite() || err > tol {
            return Err(
                format!("attention self-test failed (max_abs={err:.6}, limit={tol})").into(),
            );
        }
        Some(err)
    } else {
        None
    };

    Ok((lin_err, attn_err))
}

fn run_device_self_test(device: &str) -> Result<serde_json::Value> {
    #[allow(unused_variables)]
    let t0 = Instant::now();
    match device {
        #[cfg(feature = "cpu")]
        "cpu" => {
            let backend = ncm_backend_cuda::CandleBackend::cpu();
            let (lin_fp32, attn_fp32) = run_precision_check(&backend, false, 1e-4, Some(1e-4))?;
            Ok(serde_json::json!({
                "device": "cpu",
                "status": "ok",
                "elapsed_ms": (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "checks": {
                    "fp32_linear_max_abs": lin_fp32,
                    "fp32_attention_max_abs": attn_fp32,
                }
            }))
        }
        #[cfg(feature = "cuda")]
        "cuda" | "cuda-cudnn" => {
            #[cfg(feature = "cuda-cudnn")]
            let base = if device == "cuda-cudnn" {
                ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?
            } else {
                ncm_backend_cuda::CandleBackend::cuda(0)?
            };
            #[cfg(not(feature = "cuda-cudnn"))]
            let base = if device == "cuda-cudnn" {
                return Err("cuda-cudnn is not enabled in this build".into());
            } else {
                ncm_backend_cuda::CandleBackend::cuda(0)?
            };

            let (lin_fp32, attn_fp32) = run_precision_check(&base, false, 1e-4, Some(1e-4))?;
            let bf16 = base
                .clone()
                .with_precision(ncm_core::ComputePrecision::Bf16)?;
            let (lin_bf16, attn_bf16) = run_precision_check(&bf16, false, 3e-2, Some(3e-2))?;
            let q8 = base.with_precision(ncm_core::ComputePrecision::Q8)?;
            let (lin_q8, _) = run_precision_check(&q8, true, 5e-2, None)?;
            Ok(serde_json::json!({
                "device": device,
                "status": "ok",
                "elapsed_ms": (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "checks": {
                    "fp32_linear_max_abs": lin_fp32,
                    "fp32_attention_max_abs": attn_fp32,
                    "bf16_linear_max_abs": lin_bf16,
                    "bf16_fused_attention_max_abs": attn_bf16,
                    "q8_linear_max_abs": lin_q8,
                }
            }))
        }
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => {
            let base = ncm_backend_mlx::MlxBackend::gpu()?;
            let (lin_fp32, attn_fp32) = run_precision_check(&base, false, 1e-4, Some(1e-4))?;
            let bf16 = base
                .clone()
                .with_precision(ncm_core::ComputePrecision::Bf16)?;
            let (lin_bf16, attn_bf16) = run_precision_check(&bf16, false, 3e-2, Some(3e-2))?;
            let q8 = base.with_precision(ncm_core::ComputePrecision::Q8)?;
            let (lin_q8, _) = run_precision_check(&q8, true, 5e-2, None)?;
            Ok(serde_json::json!({
                "device": "mlx",
                "status": "ok",
                "elapsed_ms": (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "checks": {
                    "fp32_linear_max_abs": lin_fp32,
                    "fp32_attention_max_abs": attn_fp32,
                    "bf16_linear_max_abs": lin_bf16,
                    "bf16_fast_sdpa_max_abs": attn_bf16,
                    "q8_linear_max_abs": lin_q8,
                }
            }))
        }
        other => Err(format!("device `{other}` is not available in this build").into()),
    }
}

fn probe_runtime(exe_path: Option<&Path>) -> (serde_json::Value, bool) {
    #[allow(unused_mut)]
    let mut ok = true;
    let mut notes: Vec<String> = Vec::new();
    let default_dev = default_runtime_device();

    #[cfg(feature = "cuda")]
    let gpu_probe = {
        let _ = exe_path;
        let nvidia_smi = run_cmd_stdout(
            "nvidia-smi",
            &[
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
        );
        #[cfg(feature = "cuda-cudnn")]
        let init_res = ncm_backend_cuda::CandleBackend::cuda_cudnn(0).map(|_| "cuda-cudnn");
        #[cfg(not(feature = "cuda-cudnn"))]
        let init_res = ncm_backend_cuda::CandleBackend::cuda(0).map(|_| "cuda");

        match init_res {
            Ok(kind) => {
                #[cfg(not(feature = "cuda-cudnn"))]
                notes.push(
                    "Built without cuda-cudnn; XSum and LM1B require --features cuda-cudnn.".into(),
                );
                serde_json::json!({
                    "backend": kind,
                    "status": "ok",
                    "nvidia_smi": nvidia_smi,
                })
            }
            Err(e) => {
                ok = false;
                notes.push(
                    "CUDA initialization failed. Ensure the NVIDIA driver is loaded and CUDA/cuDNN libraries are on LD_LIBRARY_PATH or in lib/.".into(),
                );
                serde_json::json!({
                    "backend": default_dev,
                    "status": "error",
                    "error": e.to_string(),
                    "nvidia_smi": nvidia_smi,
                })
            }
        }
    };

    #[cfg(all(
        not(feature = "cuda"),
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    let gpu_probe = {
        let chip = run_cmd_stdout("sysctl", &["-n", "machdep.cpu.brand_string"]);
        let macos = run_cmd_stdout("sw_vers", &["-productVersion"]);
        let mem_bytes =
            run_cmd_stdout("sysctl", &["-n", "hw.memsize"]).and_then(|s| s.parse::<u64>().ok());
        let bundled_metallib = exe_path
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|root| root.join("lib/mlx.metallib"))
            .filter(|p| p.is_file());
        let prebuilt_metallib = env::var("MLX_SYS_PREBUILT")
            .ok()
            .map(|p| PathBuf::from(p).join("lib/mlx.metallib"))
            .filter(|p| p.is_file());
        let metallib = bundled_metallib
            .or(prebuilt_metallib)
            .map(|p| p.display().to_string());

        match ncm_backend_mlx::MlxBackend::gpu() {
            Ok(_) => serde_json::json!({
                "backend": "mlx",
                "status": "ok",
                "chip": chip,
                "macos_version": macos,
                "unified_memory_bytes": mem_bytes,
                "metallib": metallib,
            }),
            Err(e) => {
                ok = false;
                notes.push(
                    "Metal/MLX initialization failed. Ensure libmlx.dylib, libjaccl.dylib, and mlx.metallib are in ../lib beside the ncm binary or MLX_SYS_PREBUILT/lib.".into(),
                );
                serde_json::json!({
                    "backend": "mlx",
                    "status": "error",
                    "error": e.to_string(),
                    "chip": chip,
                    "macos_version": macos,
                    "metallib": metallib,
                })
            }
        }
    };

    #[cfg(all(
        not(feature = "cuda"),
        not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))
    ))]
    let gpu_probe = {
        let _ = exe_path;
        notes.push(
            "CPU-only or minimal build; rebuild with --features cuda-cudnn (Linux) or --features mlx (Apple Silicon) for GPU acceleration.".into(),
        );
        serde_json::json!({
            "backend": default_dev,
            "status": if default_dev == "cpu" { "ok" } else { "unavailable" },
        })
    };

    (
        serde_json::json!({
            "runtime": gpu_probe,
            "notes": notes,
        }),
        ok,
    )
}

pub(super) fn run_doctor(args: &[String]) -> Result<()> {
    let cfg = parse_doctor_args(args)?;
    let exe_path = env::current_exe().ok();
    let resolution = paths::resolve_models_dir(cfg.models_dir.as_deref());
    let upstream_dir = paths::resolve_upstream_dir(None);
    let default_dev = default_runtime_device();

    let (probe, mut all_ok) = probe_runtime(exe_path.as_deref());
    let curl_version =
        run_cmd_stdout("curl", &["--version"]).and_then(|s| s.lines().next().map(str::to_owned));

    let models_dir_exists = resolution.path.is_dir();
    let mut models_json = Vec::new();
    let mut total_model_bytes = 0u64;
    if models_dir_exists
        && let Ok(registry) = ModelRegistry::discover(Some(&resolution.path), &[], "fp32")
    {
        for entry in registry.list() {
            total_model_bytes = total_model_bytes.saturating_add(entry.size_bytes);
            let sidecars = detect_prequant_sidecars(&entry.path);
            models_json.push(serde_json::json!({
                "name": entry.name,
                "profile": entry.profile,
                "size_bytes": entry.size_bytes,
                "prequant_sidecars": sidecars,
                "path": entry.path.display().to_string(),
            }));
        }
    }

    let self_test_result = if cfg.self_test {
        let target_dev = cfg.device.as_deref().unwrap_or(default_dev);
        match run_device_self_test(target_dev) {
            Ok(v) => Some(v),
            Err(e) => {
                all_ok = false;
                Some(serde_json::json!({
                    "device": target_dev,
                    "status": "error",
                    "error": e.to_string(),
                }))
            }
        }
    } else {
        None
    };

    let report = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "version_string": version_string(),
        "os": env::consts::OS,
        "arch": env::consts::ARCH,
        "executable": exe_path.as_ref().map(|p| p.display().to_string()),
        "compiled_features": compiled_features(),
        "default_device": default_dev,
        "runtime": probe["runtime"],
        "notes": probe["notes"],
        "tools": {
            "curl": curl_version,
        },
        "models_dir": {
            "path": resolution.path.display().to_string(),
            "source": resolution.source,
            "exists": models_dir_exists,
            "count": models_json.len(),
            "total_bytes": total_model_bytes,
            "models": models_json,
        },
        "upstream_cache_dir": upstream_dir.display().to_string(),
        "self_test": self_test_result,
        "ok": all_ok,
    });

    if cfg.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", version_string());
        if let Some(exe) = exe_path {
            println!("  executable  : {}", exe.display());
        }
        println!(
            "  runtime     : {} ({})",
            default_dev,
            probe["runtime"]["status"].as_str().unwrap_or("unknown")
        );
        if let Some(smi) = probe["runtime"]["nvidia_smi"].as_str() {
            println!("  gpu         : {smi}");
        }
        if let Some(chip) = probe["runtime"]["chip"].as_str() {
            let mem = probe["runtime"]["unified_memory_bytes"]
                .as_u64()
                .map(|b| format!(", {}", format_bytes(b)))
                .unwrap_or_default();
            let mac = probe["runtime"]["macos_version"]
                .as_str()
                .map(|v| format!(" (macOS {v})"))
                .unwrap_or_default();
            println!("  hardware    : {chip}{mem}{mac}");
        }
        if let Some(mlib) = probe["runtime"]["metallib"].as_str() {
            println!("  metallib    : {mlib}");
        }
        println!(
            "  curl        : {}",
            curl_version
                .as_deref()
                .unwrap_or("missing (required for online `ncm pull`)")
        );
        println!(
            "  models_dir  : {} [{}] ({} models, {})",
            resolution.path.display(),
            resolution.source,
            models_json.len(),
            format_bytes(total_model_bytes)
        );
        for m in &models_json {
            let name = m["name"].as_str().unwrap_or("");
            let profile = m["profile"].as_str().unwrap_or("");
            let sz = format_bytes(m["size_bytes"].as_u64().unwrap_or(0));
            let sidecars: Vec<&str> = m["prequant_sidecars"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            let sc_str = if sidecars.is_empty() {
                String::new()
            } else {
                format!(" [sidecars: {}]", sidecars.join(", "))
            };
            println!("    - {name:<22} {profile:<28} {sz:>10}{sc_str}");
        }
        if let Some(st) = &self_test_result {
            let dev = st["device"].as_str().unwrap_or("");
            let status = st["status"].as_str().unwrap_or("error");
            if status == "ok" {
                println!(
                    "  self_test   : PASS on {dev} ({:.2} ms)",
                    st["elapsed_ms"].as_f64().unwrap_or(0.0)
                );
            } else {
                println!(
                    "  self_test   : FAIL on {dev} ({})",
                    st["error"].as_str().unwrap_or("unknown error")
                );
            }
        }
        if let Some(notes) = probe["notes"].as_array() {
            for note in notes.iter().filter_map(|v| v.as_str()) {
                println!("  note        : {note}");
            }
        }
    }

    if !all_ok {
        return Err("ncm doctor reported one or more failed checks".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DoctorArgs, parse_doctor_args, version_string};
    use std::path::PathBuf;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).into()).collect()
    }

    #[test]
    fn doctor_arg_parser_handles_flags_and_rejects_unknown() {
        assert_eq!(
            parse_doctor_args(&args(&["ncm", "doctor"])).unwrap(),
            DoctorArgs::default()
        );
        assert_eq!(
            parse_doctor_args(&args(&[
                "ncm",
                "doctor",
                "--models-dir",
                "/tmp/models",
                "--device",
                "cpu",
                "--self-test",
                "--json"
            ]))
            .unwrap(),
            DoctorArgs {
                models_dir: Some(PathBuf::from("/tmp/models")),
                device: Some("cpu".into()),
                self_test: true,
                json: true,
            }
        );
        assert!(parse_doctor_args(&args(&["ncm", "doctor", "--bogus"])).is_err());
        assert!(parse_doctor_args(&args(&["ncm", "doctor", "--models-dir"])).is_err());
    }

    #[test]
    fn version_string_contains_crate_version_and_platform() {
        let v = version_string();
        assert!(v.contains(env!("CARGO_PKG_VERSION")));
        assert!(v.contains(std::env::consts::OS));
        assert!(v.contains(std::env::consts::ARCH));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_self_test_passes_numerically() {
        let res = super::run_device_self_test("cpu").unwrap();
        assert_eq!(res["status"], "ok");
        assert!(res["checks"]["fp32_linear_max_abs"].as_f64().unwrap() < 1e-4);
        assert!(res["checks"]["fp32_attention_max_abs"].as_f64().unwrap() < 1e-4);
    }
}
