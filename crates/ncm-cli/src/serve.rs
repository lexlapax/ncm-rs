//! Local model listing (`ncm list`), pulling (`ncm pull`), interactive running (`ncm run`),
//! and OpenAI/Ollama-compatible HTTP serving (`ncm serve`).
use crate::Result;
use ncm_core::ComputePrecision;
use ncm_serve::{ModelEntry, ModelRegistry, inspect_model_dir};
use std::path::{Path, PathBuf};

const LIST_USAGE: &str = "usage: ncm list [--models-dir DIR] [--model NAME=PATH]... [--precision fp32|bf16|fp16|q8|q4] [--json]";

const RUN_USAGE: &str = "\
usage: ncm run MODEL [PROMPT] [--models-dir DIR] [--model NAME=PATH]... [--adapter DIR] [--device DEV] [--precision fp32|bf16|fp16|q8|q4] [--seed N] [--cfg S] [--max-tokens N]";

const PULL_USAGE: &str = "\
usage: ncm pull (--list | MODEL) [--models-dir DIR] [--upstream-dir DIR] [--output DIR] [--reference-rope FILE] [--dry-run] [--python-legacy] [--python PY]";

const SERVE_USAGE: &str = "\
usage: ncm serve [OPTIONS]

Options:
  --models-dir DIR                 Directory containing NCM artifacts (default: NCM_MODELS_DIR, ./artifacts/canonical, or user data dir)
  --model NAME=PATH                Register a model artifact under NAME (repeatable)
  --adapter ALIAS=BASE:DIR         Register a hot-swappable LoRA adapter over BASE (repeatable)
  --device DEV                     Explicit device: cuda-cudnn, cuda, mlx, or cpu (default: compiled best)
  --precision fp32|bf16|fp16|q8|q4 Compute/weight precision (default: fp32)
  --host ADDR                      Bind host (default: 127.0.0.1)
  --port PORT                      Bind port (default: 11434)
  --preload NAME                   Load and warm-cache one model before accepting requests
  --max-loaded N                   Max simultaneously resident models in LRU cache (default: 1, max: 16)
  --max-batch N                    Max concurrent requests coalesced into one GPU batch (default: 8, max: 32)
";

pub(super) const RUN_HELP: &str = "\
Run a local NCM model interactively (REPL) or for one prompt, Ollama-style.

Usage:
  ncm run MODEL [PROMPT] [OPTIONS]

Options:
  --models-dir DIR                 Directory containing NCM artifacts (default: NCM_MODELS_DIR, ./artifacts/canonical, or user data dir)
  --model NAME=PATH                Register a model artifact under NAME (repeatable)
  --adapter DIR                    Optional verified ncm-lora-adapter-v1 LoRA adapter directory
  --device DEV                     Explicit device: cuda-cudnn, cuda, mlx, or cpu
  --precision fp32|bf16|fp16|q8|q4 Compute and weight precision (default: fp32)
  --seed N                         Deterministic RNG seed for stochastic models
  --cfg S                          Classifier-free guidance scale (default: 0)

Omit PROMPT to start an interactive session keeping the model hot in GPU memory.
Type /bye, /exit, or press Ctrl-D to leave.
";

pub(super) const PULL_HELP: &str = "\
Fetch and export a verified upstream NCM model natively in pure Rust (zero Python required).

Usage:
  ncm pull --list
  ncm pull MODEL [--models-dir DIR] [--upstream-dir DIR] [--output DIR] [--reference-rope FILE] [--dry-run]

Models: tinystories, tinystories-legacy, xsum, wmt14, lm1b-small, lm1b-medium, lm1b-large, sudoku-easy, sudoku-medium, sudoku-hard.

Pass --python-legacy (or --python PY) to delegate to tools/reference/quickstart_export.py instead.
";

pub(super) const LIST_HELP: &str = "\
List discovered local NCM artifacts.

Usage:
  ncm list [--models-dir DIR] [--model NAME=PATH]... [--json]

Scans --models-dir (default: NCM_MODELS_DIR, ./artifacts/canonical if present, else user data dir)
for directories containing a valid manifest.json with a servable NCM profile.
";

pub(super) const SERVE_HELP: &str = "\
Serve local NCM models over an OpenAI- and Ollama-compatible HTTP/1.1 API.

Usage:
  ncm serve [OPTIONS]

Options:
  --models-dir DIR                 Directory containing NCM artifacts (default: NCM_MODELS_DIR, ./artifacts/canonical, or user data dir)
  --model NAME=PATH                Register a model artifact under NAME (repeatable)
  --adapter ALIAS=BASE:DIR         Register a hot-swappable LoRA adapter over BASE (repeatable)
  --device DEV                     Explicit device: cuda-cudnn, cuda, mlx, or cpu
  --precision fp32|bf16|fp16|q8|q4 Compute and weight precision (default: fp32)
  --host ADDR                      Bind address (default: 127.0.0.1)
  --port PORT                      Bind port (default: 11434)
  --preload NAME                   Preload one model on startup
  --max-loaded N                   Max resident models in LRU cache (default: 1, max: 16)
  --max-batch N                    Max concurrent requests coalesced per GPU batch (default: 8)

OpenAI endpoints:
  GET  /v1/models
  POST /v1/completions        (stream: false|true SSE)
  POST /v1/chat/completions   (stream: false|true SSE)

Ollama endpoints:
  GET  /  |  GET /health  |  GET /api/tags  |  GET /api/ps  |  POST /api/show
  POST /api/generate          (stream: true|false NDJSON)
  POST /api/chat              (stream: true|false NDJSON)
";

fn default_models_dir() -> Option<PathBuf> {
    Some(crate::paths::resolve_models_dir(None).path)
}

fn parse_override(raw: &str) -> Result<(String, PathBuf)> {
    let Some((name, path)) = raw.split_once('=') else {
        return Err(format!("invalid --model `{raw}` (expected NAME=PATH)").into());
    };
    if name.trim().is_empty() || path.trim().is_empty() {
        return Err(format!("invalid --model `{raw}` (expected NAME=PATH)").into());
    }
    Ok((name.trim().to_owned(), PathBuf::from(path.trim())))
}

fn parse_adapter_spec(raw: &str) -> Result<(String, String, PathBuf)> {
    let Some((alias, rest)) = raw.split_once('=') else {
        return Err(format!("invalid --adapter `{raw}` (expected ALIAS=BASE:DIR)").into());
    };
    let Some((base, path)) = rest.split_once(':') else {
        return Err(format!("invalid --adapter `{raw}` (expected ALIAS=BASE:DIR)").into());
    };
    if alias.trim().is_empty() || base.trim().is_empty() || path.trim().is_empty() {
        return Err(format!("invalid --adapter `{raw}` (expected ALIAS=BASE:DIR)").into());
    }
    Ok((
        alias.trim().to_owned(),
        base.trim().to_owned(),
        PathBuf::from(path.trim()),
    ))
}

fn format_size(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * MIB;
    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes as f64 >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

pub(super) fn run_list(args: &[String]) -> Result<()> {
    let mut models_dir: Option<PathBuf> = None;
    let mut overrides: Vec<(String, PathBuf)> = Vec::new();
    let mut precision = ComputePrecision::Fp32;
    let mut as_json = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                let v = args.get(i + 1).ok_or(LIST_USAGE)?;
                models_dir = Some(PathBuf::from(v));
                i += 2;
            }
            "--model" => {
                let v = args.get(i + 1).ok_or(LIST_USAGE)?;
                overrides.push(parse_override(v)?);
                i += 2;
            }
            "--precision" => {
                let v = args.get(i + 1).ok_or(LIST_USAGE)?;
                precision = v.parse()?;
                i += 2;
            }
            "--json" => {
                as_json = true;
                i += 1;
            }
            _ => return Err(LIST_USAGE.into()),
        }
    }
    let dir = models_dir.or_else(default_models_dir);
    let registry = ModelRegistry::discover(dir.as_deref(), &overrides, precision.as_str())
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let models = registry.list();
    if as_json {
        println!("{}", serde_json::to_string_pretty(&models)?);
        return Ok(());
    }
    if models.is_empty() {
        println!(
            "No local NCM models found (searched {}). Pass --models-dir DIR or --model NAME=PATH.",
            dir.as_deref().map_or(".", |p| p.to_str().unwrap_or("."))
        );
        return Ok(());
    }
    let name_w = models
        .iter()
        .map(|m| m.name.len())
        .max()
        .unwrap_or(4)
        .max(4)
        + 2;
    let prof_w = models
        .iter()
        .map(|m| m.profile.len())
        .max()
        .unwrap_or(7)
        .max(7)
        + 2;
    println!(
        "{:<name_w$}{:<prof_w$}{:>10}  {:<5} PATH",
        "NAME", "PROFILE", "SIZE", "TEXT"
    );
    for m in &models {
        println!(
            "{:<name_w$}{:<prof_w$}{:>10}  {:<5} {}",
            m.name,
            m.profile,
            format_size(m.size_bytes),
            if m.supports_text { "yes" } else { "no" },
            m.path.display()
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ServeConfig {
    models_dir: Option<PathBuf>,
    overrides: Vec<(String, PathBuf)>,
    adapters: Vec<(String, String, PathBuf)>,
    device: String,
    precision: ComputePrecision,
    host: String,
    port: u16,
    preload: Option<String>,
    max_loaded: usize,
    max_batch: usize,
}

fn default_device() -> &'static str {
    #[cfg(feature = "cuda-cudnn")]
    {
        "cuda-cudnn"
    }
    #[cfg(all(not(feature = "cuda-cudnn"), feature = "cuda"))]
    {
        "cuda"
    }
    #[cfg(all(
        not(feature = "cuda-cudnn"),
        not(feature = "cuda"),
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    {
        "mlx"
    }
    #[cfg(all(
        not(feature = "cuda-cudnn"),
        not(feature = "cuda"),
        not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")),
        feature = "cpu"
    ))]
    {
        "cpu"
    }
    #[cfg(not(any(
        feature = "cuda-cudnn",
        feature = "cuda",
        feature = "cpu",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    )))]
    {
        "none"
    }
}

fn parse_serve_args(args: &[String]) -> Result<ServeConfig> {
    let mut cfg = ServeConfig {
        models_dir: None,
        overrides: Vec::new(),
        adapters: Vec::new(),
        device: default_device().to_owned(),
        precision: ComputePrecision::Fp32,
        host: "127.0.0.1".to_owned(),
        port: 11434,
        preload: None,
        max_loaded: 1,
        max_batch: 8,
    };
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                cfg.models_dir = Some(PathBuf::from(args.get(i + 1).ok_or(SERVE_USAGE)?));
                i += 2;
            }
            "--model" => {
                cfg.overrides
                    .push(parse_override(args.get(i + 1).ok_or(SERVE_USAGE)?)?);
                i += 2;
            }
            "--adapter" => {
                cfg.adapters
                    .push(parse_adapter_spec(args.get(i + 1).ok_or(SERVE_USAGE)?)?);
                i += 2;
            }
            "--device" => {
                cfg.device = args.get(i + 1).ok_or(SERVE_USAGE)?.clone();
                i += 2;
            }
            "--precision" => {
                cfg.precision = args.get(i + 1).ok_or(SERVE_USAGE)?.parse()?;
                i += 2;
            }
            "--host" => {
                cfg.host = args.get(i + 1).ok_or(SERVE_USAGE)?.clone();
                i += 2;
            }
            "--port" => {
                cfg.port = args.get(i + 1).ok_or(SERVE_USAGE)?.parse()?;
                i += 2;
            }
            "--preload" => {
                cfg.preload = Some(args.get(i + 1).ok_or(SERVE_USAGE)?.clone());
                i += 2;
            }
            "--max-loaded" => {
                let n: usize = args.get(i + 1).ok_or(SERVE_USAGE)?.parse()?;
                if !(1..=16).contains(&n) {
                    return Err("--max-loaded must be between 1 and 16".into());
                }
                cfg.max_loaded = n;
                i += 2;
            }
            "--max-batch" => {
                let n: usize = args.get(i + 1).ok_or(SERVE_USAGE)?.parse()?;
                if !(1..=32).contains(&n) {
                    return Err("--max-batch must be between 1 and 32".into());
                }
                cfg.max_batch = n;
                i += 2;
            }
            _ => return Err(SERVE_USAGE.into()),
        }
    }
    if cfg.models_dir.is_none() && cfg.overrides.is_empty() {
        cfg.models_dir = default_models_dir();
    }
    Ok(cfg)
}

pub(super) fn run_serve(args: &[String]) -> Result<()> {
    let cfg = parse_serve_args(args)?;
    let mut registry = ModelRegistry::discover(
        cfg.models_dir.as_deref(),
        &cfg.overrides,
        cfg.precision.as_str(),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    for (alias, base_name, adapter_dir) in &cfg.adapters {
        let base_entry = registry
            .resolve(base_name)
            .cloned()
            .ok_or_else(|| format!("--adapter `{alias}` references unknown base `{base_name}`"))?;
        let base_manifest_bytes = std::fs::read(base_entry.path.join("manifest.json"))?;
        let base_sha = ncm_io::sha256(&base_manifest_bytes);
        let (adapter_manifest, adapter_sha) = ncm_io::inspect_lora_adapter_manifest(adapter_dir)
            .map_err(|e| {
                format!(
                    "invalid --adapter `{alias}` ({}): {e}",
                    adapter_dir.display()
                )
            })?;
        if adapter_manifest.base_manifest_sha256 != base_sha {
            return Err(format!(
                "--adapter `{alias}` targets base manifest {} ({}), which does not match `{base_name}` ({})",
                &adapter_manifest.base_manifest_sha256[..12],
                adapter_manifest.base_profile,
                &base_sha[..12]
            )
            .into());
        }
        registry
            .register_adapter(
                alias,
                base_name,
                adapter_dir.clone(),
                adapter_manifest.adapter_file.bytes,
                adapter_sha,
            )
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    }

    if registry.is_empty() {
        return Err(
            "no servable NCM models found; pass --models-dir DIR or --model NAME=PATH".into(),
        );
    }

    #[cfg(not(any(
        feature = "cpu",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    )))]
    {
        let _ = (&cfg, &registry);
        Err(
            "ncm serve requires a compiled backend feature (--features cuda-cudnn, --features cuda, --features mlx, or --features cpu)"
                .into(),
        )
    }

    #[cfg(any(
        feature = "cpu",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    {
        worker::start_server(cfg, registry)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PullConfig {
    list: bool,
    model: Option<String>,
    models_dir: Option<PathBuf>,
    upstream_dir: Option<PathBuf>,
    output: Option<PathBuf>,
    python: Option<String>,
    python_legacy: bool,
    reference_rope: Option<PathBuf>,
    dry_run: bool,
}

fn parse_pull_args(args: &[String]) -> Result<PullConfig> {
    let mut cfg = PullConfig {
        list: false,
        model: None,
        models_dir: None,
        upstream_dir: None,
        output: None,
        python: None,
        python_legacy: false,
        reference_rope: None,
        dry_run: false,
    };
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--list" => {
                cfg.list = true;
                i += 1;
            }
            "--models-dir" => {
                cfg.models_dir = Some(PathBuf::from(args.get(i + 1).ok_or(PULL_USAGE)?));
                i += 2;
            }
            "--upstream-dir" => {
                cfg.upstream_dir = Some(PathBuf::from(args.get(i + 1).ok_or(PULL_USAGE)?));
                i += 2;
            }
            "--output" => {
                cfg.output = Some(PathBuf::from(args.get(i + 1).ok_or(PULL_USAGE)?));
                i += 2;
            }
            "--python" => {
                cfg.python = Some(args.get(i + 1).ok_or(PULL_USAGE)?.clone());
                cfg.python_legacy = true;
                i += 2;
            }
            "--python-legacy" => {
                cfg.python_legacy = true;
                i += 1;
            }
            "--reference-rope" => {
                cfg.reference_rope = Some(PathBuf::from(args.get(i + 1).ok_or(PULL_USAGE)?));
                i += 2;
            }
            "--dry-run" => {
                cfg.dry_run = true;
                i += 1;
            }
            other if other.starts_with('-') => return Err(PULL_USAGE.into()),
            positional => {
                if cfg.model.is_some() {
                    return Err(PULL_USAGE.into());
                }
                cfg.model = Some(positional.to_owned());
                i += 1;
            }
        }
    }
    if !cfg.list && cfg.model.is_none() {
        return Err(PULL_USAGE.into());
    }
    Ok(cfg)
}

fn find_quickstart_script() -> Result<PathBuf> {
    let rel = Path::new("tools/reference/quickstart_export.py");
    if rel.is_file() {
        return Ok(rel.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors() {
            let candidate = ancestor.join(rel);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err("could not locate tools/reference/quickstart_export.py; run from the ncm-rs repository root".into())
}

fn default_python(script: &Path) -> String {
    if let Some(repo_root) = script.ancestors().nth(3) {
        let venv = repo_root.join(".reference/venv/bin/python");
        if venv.is_file() {
            return venv.to_string_lossy().into_owned();
        }
    }
    "python3".to_owned()
}

pub(super) fn run_pull(args: &[String]) -> Result<()> {
    let cfg = parse_pull_args(args)?;
    if !cfg.python_legacy {
        if cfg.list {
            crate::pull::print_model_list();
            return Ok(());
        }
        let model = cfg.model.ok_or(PULL_USAGE)?;
        let opts = crate::pull::NativePullOptions {
            model,
            models_dir: crate::paths::resolve_models_dir(cfg.models_dir.as_deref()).path,
            upstream_dir: crate::paths::resolve_upstream_dir(cfg.upstream_dir.as_deref()),
            output_override: cfg.output,
            reference_rope: cfg.reference_rope,
            dry_run: cfg.dry_run,
        };
        return crate::pull::run_native_pull(&opts);
    }

    let script = find_quickstart_script()?;
    let python = cfg.python.unwrap_or_else(|| default_python(&script));
    let mut cmd = std::process::Command::new(&python);
    cmd.arg(&script);
    if cfg.list {
        cmd.arg("--list");
    } else if let Some(model) = &cfg.model {
        cmd.arg("--model").arg(model);
    }
    cmd.arg("--python").arg(&python);
    if let Some(dir) = &cfg.models_dir {
        cmd.arg("--models-dir").arg(dir);
    }
    if let Some(dir) = &cfg.upstream_dir {
        cmd.arg("--upstream-dir").arg(dir);
    }
    if let Some(rope) = &cfg.reference_rope {
        cmd.arg("--reference-rope").arg(rope);
    }
    if cfg.dry_run {
        cmd.arg("--dry-run");
    }
    let status = cmd
        .status()
        .map_err(|e| format!("failed to launch `{python}` for ncm pull: {e}"))?;
    if !status.success() {
        return Err(format!("ncm pull exited with status {status}").into());
    }
    Ok(())
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RunConfig {
    model: String,
    prompt: Option<String>,
    models_dir: Option<PathBuf>,
    overrides: Vec<(String, PathBuf)>,
    adapter: Option<PathBuf>,
    device: String,
    precision: ComputePrecision,
    seed: Option<u64>,
    cfg_scale: Option<f64>,
    max_tokens: Option<usize>,
}

fn parse_run_args(args: &[String]) -> Result<RunConfig> {
    let mut model: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut models_dir: Option<PathBuf> = None;
    let mut overrides: Vec<(String, PathBuf)> = Vec::new();
    let mut adapter: Option<PathBuf> = None;
    let mut device = default_device().to_owned();
    let mut precision = ComputePrecision::Fp32;
    let mut seed: Option<u64> = None;
    let mut cfg_scale: Option<f64> = None;
    let mut max_tokens: Option<usize> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                models_dir = Some(PathBuf::from(args.get(i + 1).ok_or(RUN_USAGE)?));
                i += 2;
            }
            "--model" => {
                overrides.push(parse_override(args.get(i + 1).ok_or(RUN_USAGE)?)?);
                i += 2;
            }
            "--adapter" => {
                adapter = Some(PathBuf::from(args.get(i + 1).ok_or(RUN_USAGE)?));
                i += 2;
            }
            "--device" => {
                device = args.get(i + 1).ok_or(RUN_USAGE)?.clone();
                i += 2;
            }
            "--precision" => {
                precision = args.get(i + 1).ok_or(RUN_USAGE)?.parse()?;
                i += 2;
            }
            "--seed" => {
                seed = Some(args.get(i + 1).ok_or(RUN_USAGE)?.parse()?);
                i += 2;
            }
            "--cfg" => {
                let s: f64 = args.get(i + 1).ok_or(RUN_USAGE)?.parse()?;
                if !s.is_finite() || s < 0.0 {
                    return Err("--cfg must be finite and nonnegative".into());
                }
                cfg_scale = Some(s);
                i += 2;
            }
            "--max-tokens" => {
                let n: usize = args.get(i + 1).ok_or(RUN_USAGE)?.parse()?;
                if n == 0 {
                    return Err("--max-tokens must be >= 1".into());
                }
                max_tokens = Some(n);
                i += 2;
            }
            other if other.starts_with('-') => return Err(RUN_USAGE.into()),
            positional => {
                if model.is_none() {
                    model = Some(positional.to_owned());
                } else if prompt.is_none() {
                    prompt = Some(positional.to_owned());
                } else {
                    return Err(RUN_USAGE.into());
                }
                i += 1;
            }
        }
    }
    let Some(model) = model else {
        return Err(RUN_USAGE.into());
    };
    if models_dir.is_none() && overrides.is_empty() {
        models_dir = default_models_dir();
    }
    Ok(RunConfig {
        model,
        prompt,
        models_dir,
        overrides,
        adapter,
        device,
        precision,
        seed,
        cfg_scale,
        max_tokens,
    })
}

fn resolve_run_model(cfg: &RunConfig) -> Result<ModelEntry> {
    let direct = Path::new(&cfg.model);
    if direct.is_dir()
        && direct.join("manifest.json").is_file()
        && let Some(entry) = inspect_model_dir(direct, None, cfg.precision.as_str())
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
    {
        return Ok(entry);
    }
    let registry = ModelRegistry::discover(
        cfg.models_dir.as_deref(),
        &cfg.overrides,
        cfg.precision.as_str(),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    registry.resolve(&cfg.model).cloned().ok_or_else(|| {
        format!(
            "model `{}` not found; run `ncm list` or pass a path to a model directory",
            cfg.model
        )
        .into()
    })
}

pub(super) fn run_repl(args: &[String]) -> Result<()> {
    let cfg = parse_run_args(args)?;
    let entry = resolve_run_model(&cfg)?;
    #[cfg(not(any(
        feature = "cpu",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    )))]
    {
        let _ = (&cfg, &entry);
        Err(
            "ncm run requires a compiled backend feature (--features cuda-cudnn, --features cuda, --features mlx, or --features cpu)"
                .into(),
        )
    }
    #[cfg(any(
        feature = "cpu",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    {
        worker::run_local_session(cfg, entry)
    }
}

#[allow(dead_code)]
pub(super) fn vram_estimate_bytes(size_bytes: u64, precision: ComputePrecision) -> u64 {
    match precision {
        ComputePrecision::Fp32 => size_bytes,
        ComputePrecision::Bf16 | ComputePrecision::Fp16 => size_bytes / 2,
        ComputePrecision::Q8 => (size_bytes.saturating_mul(30)) / 100,
        ComputePrecision::Q4 => (size_bytes.saturating_mul(20)) / 100,
    }
}

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
mod worker {
    use super::{ModelRegistry, Result, RunConfig, ServeConfig, format_size, vram_estimate_bytes};
    use crate::decode::Decoder;
    use ncm_core::{ComputePrecision, GeneratorBackend};
    use ncm_io::Artifact;
    use ncm_model::{Generator, LocalNcmModel, TinyStoriesTeacher};
    use ncm_serve::{
        EngineError, GenerateRequest, GenerateResponse, InferenceEngine, LoadedModelInfo,
        ModelEntry, Server,
    };
    use std::{
        collections::{HashMap, VecDeque},
        io::{BufRead, IsTerminal, Write},
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
            mpsc,
        },
        thread,
        time::{Instant, SystemTime, UNIX_EPOCH},
    };

    #[cfg(any(
        feature = "cuda-cudnn",
        all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    use ncm_model::XsumTeacher;

    #[cfg(any(
        all(feature = "lm1b-experimental", target_os = "linux"),
        all(
            feature = "lm1b-mlx-experimental",
            target_os = "macos",
            target_arch = "aarch64"
        )
    ))]
    use ncm_model::{Lm1bSmallTeacher, lm1b_small_reference_rope};

    #[cfg(feature = "wmt14-experimental")]
    use ncm_model::Wmt14Teacher;

    #[cfg(any(
        all(feature = "lm1b-experimental", target_os = "linux"),
        all(
            feature = "lm1b-mlx-experimental",
            target_os = "macos",
            target_arch = "aarch64"
        )
    ))]
    use ncm_tokenizers::TextTokenizer;

    type GenReply = mpsc::Sender<std::result::Result<GenerateResponse, EngineError>>;
    type GenOutput = std::result::Result<(String, usize, usize, String), EngineError>;

    enum WorkMsg {
        Generate {
            req: GenerateRequest,
            reply: GenReply,
        },
        Preload {
            name: String,
            reply: mpsc::Sender<std::result::Result<(), EngineError>>,
        },
    }

    struct LocalEngine {
        registry: Arc<ModelRegistry>,
        ps_snapshot: Arc<Mutex<Vec<LoadedModelInfo>>>,
        tx: mpsc::Sender<WorkMsg>,
    }

    impl InferenceEngine for LocalEngine {
        fn list_models(&self) -> Vec<ModelEntry> {
            self.registry.list()
        }
        fn get_model(&self, name: &str) -> Option<ModelEntry> {
            self.registry.resolve(name).cloned()
        }
        fn loaded_models(&self) -> Vec<LoadedModelInfo> {
            self.ps_snapshot
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        }
        fn generate(
            &self,
            req: GenerateRequest,
        ) -> std::result::Result<GenerateResponse, EngineError> {
            let (reply_tx, reply_rx) = mpsc::channel();
            self.tx
                .send(WorkMsg::Generate {
                    req,
                    reply: reply_tx,
                })
                .map_err(|_| EngineError::Internal("inference worker exited".into()))?;
            reply_rx
                .recv()
                .map_err(|_| EngineError::Internal("inference worker dropped response".into()))?
        }
    }

    fn unix_now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    }

    pub(super) fn run_local_session(cfg: RunConfig, entry: ModelEntry) -> Result<()> {
        validate_device_precision(&cfg.device, cfg.precision)?;
        let t_load = Instant::now();
        let mut loaded = load_model(&entry, &cfg.device, cfg.precision)
            .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
        if let Some(ref adapter_dir) = cfg.adapter {
            loaded
                .select_adapter(Some(adapter_dir))
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
        }
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        let auto_seed = AtomicU64::new(cfg.seed.unwrap_or(42));

        if let Some(prompt) = cfg.prompt {
            let req = GenerateRequest {
                model: entry.name.clone(),
                prompt,
                seed: cfg.seed,
                cfg_scale: cfg.cfg_scale,
                max_tokens: cfg.max_tokens,
                stop: Vec::new(),
            };
            let mut results = run_loaded_batch(&mut loaded, &[&req], &auto_seed);
            let (text, _pt, _ct, _finish) = results
                .pop()
                .unwrap_or_else(|| Err(EngineError::Internal("empty batch result".into())))
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
            println!("{text}");
            return Ok(());
        }

        let is_tty = std::io::stdin().is_terminal();
        let is_lm1b = entry.profile.starts_with("lm1b-");
        eprintln!(
            "Loaded `{}` ({}, device={}, precision={}, {:.1} ms). Type a prompt or /bye to exit.",
            entry.name, entry.profile, cfg.device, cfg.precision, load_ms
        );
        if is_lm1b {
            eprintln!("Note: LM1B is unconditional; press Enter to draw a new sample.");
        }

        let mut current_seed = cfg.seed;
        let mut current_cfg = cfg.cfg_scale;
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        let mut line = String::new();
        loop {
            if is_tty {
                eprint!(">>> ");
                let _ = std::io::stderr().flush();
            }
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            let trimmed = line.trim();
            match trimmed {
                "/bye" | "/exit" | "exit" | "quit" => break,
                "/help" => {
                    eprintln!("Commands: /bye, /exit, /seed <N|auto>, /cfg <SCALE>, /help");
                    continue;
                }
                s if s.starts_with("/seed") => {
                    let arg = s.strip_prefix("/seed").unwrap_or("").trim();
                    if arg.is_empty() || arg == "auto" {
                        current_seed = None;
                        eprintln!("seed = auto");
                    } else if let Ok(v) = arg.parse::<u64>() {
                        current_seed = Some(v);
                        eprintln!("seed = {v}");
                    } else {
                        eprintln!("usage: /seed <u64|auto>");
                    }
                    continue;
                }
                s if s.starts_with("/cfg") => {
                    let arg = s.strip_prefix("/cfg").unwrap_or("").trim();
                    if let Ok(v) = arg.parse::<f64>()
                        && v.is_finite()
                        && v >= 0.0
                    {
                        current_cfg = Some(v);
                        eprintln!("cfg = {v}");
                    } else {
                        eprintln!("usage: /cfg <nonnegative float>");
                    }
                    continue;
                }
                "" if !is_lm1b => continue,
                _ => {}
            }

            let req = GenerateRequest {
                model: entry.name.clone(),
                prompt: if is_lm1b {
                    String::new()
                } else {
                    trimmed.to_owned()
                },
                seed: current_seed,
                cfg_scale: current_cfg,
                max_tokens: cfg.max_tokens,
                stop: Vec::new(),
            };
            let t0 = Instant::now();
            let mut results = run_loaded_batch(&mut loaded, &[&req], &auto_seed);
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            match results
                .pop()
                .unwrap_or_else(|| Err(EngineError::Internal("empty batch result".into())))
            {
                Ok((text, _pt, ct, _finish)) => {
                    println!("{text}");
                    if is_tty {
                        eprintln!("[{ct} tok, {elapsed_ms:.1} ms]");
                    }
                }
                Err(e) => eprintln!("error: {e}"),
            }
        }
        Ok(())
    }

    pub(super) fn start_server(cfg: ServeConfig, registry: ModelRegistry) -> Result<()> {
        validate_device_precision(&cfg.device, cfg.precision)?;
        let registry = Arc::new(registry);
        let ps_snapshot = Arc::new(Mutex::new(Vec::<LoadedModelInfo>::new()));
        let worker_registry = Arc::clone(&registry);
        let worker_ps = Arc::clone(&ps_snapshot);
        let worker_device = cfg.device.clone();
        let worker_precision = cfg.precision;
        let max_loaded = cfg.max_loaded;
        let max_batch = cfg.max_batch;
        let (tx, rx) = mpsc::channel::<WorkMsg>();

        let _worker = thread::spawn(move || {
            run_worker_loop(
                worker_registry,
                worker_ps,
                &worker_device,
                worker_precision,
                max_loaded,
                max_batch,
                rx,
            );
        });

        let bind_addr = format!("{}:{}", cfg.host, cfg.port);
        let engine = LocalEngine {
            registry: Arc::clone(&registry),
            ps_snapshot,
            tx: tx.clone(),
        };
        let server = Server::bind(&bind_addr, engine).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AddrInUse {
                format!(
                    "port {} on {} is already in use (another server such as Ollama may be running); pass --port <PORT>, e.g. --port {}",
                    cfg.port,
                    cfg.host,
                    cfg.port.saturating_add(1)
                )
                .into()
            } else {
                Box::<dyn std::error::Error>::from(e)
            }
        })?;
        let actual = server.local_addr()?;

        if let Some(preload_name) = &cfg.preload {
            eprintln!("ncm serve: preloading `{preload_name}` ...");
            let (reply_tx, reply_rx) = mpsc::channel();
            tx.send(WorkMsg::Preload {
                name: preload_name.clone(),
                reply: reply_tx,
            })
            .map_err(|_| "worker stopped during preload")?;
            reply_rx
                .recv()
                .map_err(|_| "worker dropped preload response")?
                .map_err(|e| format!("preload failed: {e}"))?;
        }

        eprintln!("ncm serve listening on http://{actual}");
        eprintln!(
            "  device: {}  precision: {}  max-loaded: {}  max-batch: {}",
            cfg.device, cfg.precision, cfg.max_loaded, cfg.max_batch
        );
        eprintln!(
            "  OpenAI API:  http://{actual}/v1/models  /v1/completions  /v1/chat/completions"
        );
        eprintln!(
            "  Ollama API:  http://{actual}/api/tags   /api/ps  /api/show  /api/generate  /api/chat"
        );
        eprintln!("  Models ({}):", registry.list().len());
        for m in registry.list() {
            if let Some(ref base) = m.base_model {
                eprintln!(
                    "    - {:<20} (LoRA adapter -> {}, {})",
                    m.name,
                    base,
                    format_size(m.size_bytes)
                );
            } else {
                eprintln!(
                    "    - {:<20} ({}, {})",
                    m.name,
                    m.profile,
                    format_size(m.size_bytes)
                );
            }
        }

        server.serve()?;
        Ok(())
    }

    fn validate_device_precision(device: &str, precision: ComputePrecision) -> Result<()> {
        match device {
            #[cfg(feature = "cpu")]
            "cpu" => {
                ncm_backend_cuda::CandleBackend::cpu().with_precision(precision)?;
                Ok(())
            }
            #[cfg(feature = "cuda")]
            "cuda" => {
                ncm_backend_cuda::CandleBackend::cuda(0)?.with_precision(precision)?;
                Ok(())
            }
            #[cfg(feature = "cuda-cudnn")]
            "cuda-cudnn" => {
                ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?.with_precision(precision)?;
                Ok(())
            }
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            "mlx" => {
                ncm_backend_mlx::MlxBackend::gpu()?.with_precision(precision)?;
                Ok(())
            }
            other => Err(format!(
                "device `{other}` is unavailable in this build (supported: {})",
                super::default_device()
            )
            .into()),
        }
    }

    enum LoadedModel {
        #[cfg(feature = "cpu")]
        Candle(LoadedSlot<ncm_backend_cuda::CandleBackend>),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        Mlx(LoadedSlot<ncm_backend_mlx::MlxBackend>),
    }

    impl LoadedModel {
        fn select_adapter(
            &mut self,
            adapter_path: Option<&std::path::Path>,
        ) -> std::result::Result<(), EngineError> {
            match self {
                #[cfg(feature = "cpu")]
                Self::Candle(slot) => slot.select_adapter(adapter_path),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                Self::Mlx(slot) => slot.select_adapter(adapter_path),
            }
        }
    }

    struct LoadedSlot<B: GeneratorBackend> {
        inner: LoadedForBackend<B>,
        base_manifest_sha256: String,
        adapters: HashMap<std::path::PathBuf, ncm_model::RuntimeLoraAdapter<B::Tensor>>,
        active_adapter: Option<std::path::PathBuf>,
    }

    impl<B: GeneratorBackend + Clone> LoadedSlot<B> {
        fn new(inner: LoadedForBackend<B>, base_manifest_sha256: String) -> Self {
            Self {
                inner,
                base_manifest_sha256,
                adapters: HashMap::new(),
                active_adapter: None,
            }
        }

        fn select_adapter(
            &mut self,
            adapter_path: Option<&std::path::Path>,
        ) -> std::result::Result<(), EngineError> {
            if self.active_adapter.as_deref() == adapter_path {
                return Ok(());
            }
            let Some(path) = adapter_path else {
                self.inner.set_lora_adapter(None)?;
                self.active_adapter = None;
                return Ok(());
            };
            if !self.adapters.contains_key(path) {
                let artifact =
                    ncm_io::LoraAdapterArtifact::load(path, Some(&self.base_manifest_sha256))
                        .map_err(|e| EngineError::BadRequest(format!("LoRA adapter error: {e}")))?;
                let (n_layer, n_embd) = self.inner.lora_dimensions()?;
                let uploaded = ncm_model::RuntimeLoraAdapter::from_artifact(
                    self.inner.backend(),
                    &artifact,
                    n_layer,
                    n_embd,
                )
                .map_err(|e| EngineError::Internal(format!("LoRA adapter upload failed: {e}")))?;
                self.adapters.insert(path.to_path_buf(), uploaded);
            }
            let cached = self.adapters[path].clone();
            self.inner.set_lora_adapter(Some(cached))?;
            self.active_adapter = Some(path.to_path_buf());
            Ok(())
        }
    }

    enum LoadedForBackend<B: GeneratorBackend> {
        Sudoku {
            backend: B,
            model: Generator<B>,
        },
        TinyStories {
            backend: B,
            model: TinyStoriesTeacher<B>,
            decoder: Decoder,
        },
        #[cfg(any(
            feature = "cuda-cudnn",
            all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
        ))]
        Xsum {
            backend: B,
            model: XsumTeacher<B>,
            decoder: Decoder,
            use_device_path: bool,
        },
        #[cfg(feature = "wmt14-experimental")]
        Wmt14 {
            backend: B,
            model: Wmt14Teacher<B>,
            decoder: Decoder,
            metal_optimized: bool,
        },
        #[cfg(any(
            all(feature = "lm1b-experimental", target_os = "linux"),
            all(
                feature = "lm1b-mlx-experimental",
                target_os = "macos",
                target_arch = "aarch64"
            )
        ))]
        Lm1b {
            backend: B,
            model: Lm1bSmallTeacher<B>,
            tokenizer: TextTokenizer,
            use_cuda_device: bool,
        },
        Local {
            backend: B,
            model: LocalNcmModel<B>,
            decoder: Option<Decoder>,
        },
    }

    impl<B: GeneratorBackend + Clone> LoadedForBackend<B> {
        fn backend(&self) -> &B {
            match self {
                Self::Sudoku { backend, .. }
                | Self::TinyStories { backend, .. }
                | Self::Local { backend, .. } => backend,
                #[cfg(any(
                    feature = "cuda-cudnn",
                    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
                ))]
                Self::Xsum { backend, .. } => backend,
                #[cfg(feature = "wmt14-experimental")]
                Self::Wmt14 { backend, .. } => backend,
                #[cfg(any(
                    all(feature = "lm1b-experimental", target_os = "linux"),
                    all(
                        feature = "lm1b-mlx-experimental",
                        target_os = "macos",
                        target_arch = "aarch64"
                    )
                ))]
                Self::Lm1b { backend, .. } => backend,
            }
        }

        fn lora_dimensions(&self) -> std::result::Result<(usize, usize), EngineError> {
            match self {
                Self::Sudoku { .. } => Err(EngineError::BadRequest(
                    "LoRA adapters are not supported on Sudoku inspection profiles".into(),
                )),
                Self::TinyStories { model, .. } => Ok(model.lora_dimensions()),
                #[cfg(any(
                    feature = "cuda-cudnn",
                    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
                ))]
                Self::Xsum { model, .. } => Ok(model.lora_dimensions()),
                #[cfg(feature = "wmt14-experimental")]
                Self::Wmt14 { model, .. } => Ok(model.lora_dimensions()),
                #[cfg(any(
                    all(feature = "lm1b-experimental", target_os = "linux"),
                    all(
                        feature = "lm1b-mlx-experimental",
                        target_os = "macos",
                        target_arch = "aarch64"
                    )
                ))]
                Self::Lm1b { model, .. } => Ok(model.lora_dimensions()),
                Self::Local { model, .. } => Ok(model.lora_dimensions()),
            }
        }

        fn set_lora_adapter(
            &mut self,
            adapter: Option<ncm_model::RuntimeLoraAdapter<B::Tensor>>,
        ) -> std::result::Result<(), EngineError> {
            match self {
                Self::Sudoku { .. } => {
                    if adapter.is_some() {
                        Err(EngineError::BadRequest(
                            "LoRA adapters are not supported on Sudoku inspection profiles".into(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Self::TinyStories { model, .. } => model
                    .set_lora_adapter(adapter)
                    .map_err(|e| EngineError::BadRequest(e.to_string())),
                #[cfg(any(
                    feature = "cuda-cudnn",
                    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
                ))]
                Self::Xsum { model, .. } => model
                    .set_lora_adapter(adapter)
                    .map_err(|e| EngineError::BadRequest(e.to_string())),
                #[cfg(feature = "wmt14-experimental")]
                Self::Wmt14 { model, .. } => model
                    .set_lora_adapter(adapter)
                    .map_err(|e| EngineError::BadRequest(e.to_string())),
                #[cfg(any(
                    all(feature = "lm1b-experimental", target_os = "linux"),
                    all(
                        feature = "lm1b-mlx-experimental",
                        target_os = "macos",
                        target_arch = "aarch64"
                    )
                ))]
                Self::Lm1b { model, .. } => model
                    .set_lora_adapter(adapter)
                    .map_err(|e| EngineError::BadRequest(e.to_string())),
                Self::Local { model, .. } => model
                    .set_lora_adapter(adapter)
                    .map_err(|e| EngineError::BadRequest(e.to_string())),
            }
        }
    }

    struct CacheEntry {
        model: LoadedModel,
        info: LoadedModelInfo,
    }

    struct LruCache {
        max_loaded: usize,
        order: VecDeque<String>,
        items: HashMap<String, CacheEntry>,
    }

    impl LruCache {
        fn new(max_loaded: usize) -> Self {
            Self {
                max_loaded,
                order: VecDeque::new(),
                items: HashMap::new(),
            }
        }
        fn touch(&mut self, key: &str) {
            self.order.retain(|k| k != key);
            self.order.push_back(key.to_owned());
        }
        fn contains(&self, key: &str) -> bool {
            self.items.contains_key(key)
        }
        fn get_mut(&mut self, key: &str) -> Option<&mut LoadedModel> {
            if self.items.contains_key(key) {
                self.touch(key);
                self.items.get_mut(key).map(|e| &mut e.model)
            } else {
                None
            }
        }
        fn record_use(&mut self, key: &str, count: u64) {
            if let Some(entry) = self.items.get_mut(key) {
                entry.info.last_used_unix = unix_now_secs();
                entry.info.requests_served = entry.info.requests_served.saturating_add(count);
            }
        }
        fn evict_if_full(&mut self) {
            while self.items.len() >= self.max_loaded {
                if let Some(oldest) = self.order.pop_front() {
                    self.items.remove(&oldest);
                } else {
                    break;
                }
            }
        }
        fn insert(&mut self, key: String, model: LoadedModel, info: LoadedModelInfo) {
            self.touch(&key);
            self.items.insert(key, CacheEntry { model, info });
        }
        fn publish(&self, sink: &Mutex<Vec<LoadedModelInfo>>) {
            let mut list = Vec::with_capacity(self.items.len());
            for key in self.order.iter().rev() {
                if let Some(entry) = self.items.get(key) {
                    list.push(entry.info.clone());
                }
            }
            if let Ok(mut guard) = sink.lock() {
                *guard = list;
            }
        }
    }

    fn build_loaded_info(entry: &ModelEntry, precision: ComputePrecision) -> LoadedModelInfo {
        let now = unix_now_secs();
        LoadedModelInfo {
            name: entry.name.clone(),
            profile: entry.profile.clone(),
            precision: precision.as_str().to_owned(),
            size_bytes: entry.size_bytes,
            vram_estimate_bytes: vram_estimate_bytes(entry.size_bytes, precision),
            digest: entry.digest.clone(),
            loaded_at_unix: now,
            last_used_unix: now,
            requests_served: 0,
        }
    }

    fn drain_matching_batch(
        pending: &mut VecDeque<WorkMsg>,
        registry: &ModelRegistry,
        canonical: &str,
        cfg_bits: u64,
        limit: usize,
        out: &mut Vec<(GenerateRequest, GenReply)>,
    ) {
        if out.len() >= limit {
            return;
        }
        let mut deferred = VecDeque::with_capacity(pending.len());
        while let Some(msg) = pending.pop_front() {
            if out.len() < limit
                && let WorkMsg::Generate { req, reply } = msg
            {
                let matches = req.seed.is_none()
                    && req.cfg_scale.unwrap_or(0.0).to_bits() == cfg_bits
                    && registry
                        .resolve(&req.model)
                        .is_some_and(|e| e.name == canonical);
                if matches {
                    out.push((req, reply));
                } else {
                    deferred.push_back(WorkMsg::Generate { req, reply });
                }
            } else {
                deferred.push_back(msg);
            }
        }
        *pending = deferred;
    }

    fn run_worker_loop(
        registry: Arc<ModelRegistry>,
        ps_snapshot: Arc<Mutex<Vec<LoadedModelInfo>>>,
        device: &str,
        precision: ComputePrecision,
        max_loaded: usize,
        max_batch: usize,
        rx: mpsc::Receiver<WorkMsg>,
    ) {
        let mut cache = LruCache::new(max_loaded);
        let auto_seed = AtomicU64::new(42);
        let mut pending: VecDeque<WorkMsg> = VecDeque::new();
        loop {
            if pending.is_empty() {
                match rx.recv() {
                    Ok(msg) => pending.push_back(msg),
                    Err(_) => break,
                }
            }
            while let Ok(msg) = rx.try_recv() {
                pending.push_back(msg);
            }
            let Some(first) = pending.pop_front() else {
                continue;
            };
            match first {
                WorkMsg::Preload { name, reply } => {
                    let res = match registry.resolve(&name) {
                        None => Err(EngineError::NotFound(format!("model `{name}` not found"))),
                        Some(entry) => {
                            let base_entry = entry
                                .base_model
                                .as_deref()
                                .and_then(|b| registry.resolve(b))
                                .unwrap_or(entry);
                            let cache_key = base_entry.name.clone();
                            if !cache.contains(&cache_key) {
                                cache.evict_if_full();
                                cache.publish(&ps_snapshot);
                                match load_model(base_entry, device, precision) {
                                    Ok(loaded) => {
                                        cache.insert(
                                            cache_key.clone(),
                                            loaded,
                                            build_loaded_info(base_entry, precision),
                                        );
                                        cache.publish(&ps_snapshot);
                                    }
                                    Err(e) => {
                                        let _ = reply.send(Err(e));
                                        continue;
                                    }
                                }
                            }
                            let select_res = match cache.get_mut(&cache_key) {
                                Some(loaded) => {
                                    loaded.select_adapter(entry.adapter_path.as_deref())
                                }
                                None => Err(EngineError::Internal("cache lookup failed".into())),
                            };
                            cache.publish(&ps_snapshot);
                            select_res
                        }
                    };
                    let _ = reply.send(res);
                }
                WorkMsg::Generate { req, reply } => {
                    let Some(entry) = registry.resolve(&req.model) else {
                        let _ = reply.send(Err(EngineError::NotFound(format!(
                            "model `{}` not found",
                            req.model
                        ))));
                        continue;
                    };
                    let canonical = entry.name.clone();
                    let base_entry = entry
                        .base_model
                        .as_deref()
                        .and_then(|b| registry.resolve(b))
                        .unwrap_or(entry);
                    let cache_key = base_entry.name.clone();
                    let adapter_path = entry.adapter_path.clone();
                    let cfg_bits = req.cfg_scale.unwrap_or(0.0).to_bits();
                    let allow_batch = max_batch > 1 && req.seed.is_none();

                    let mut batch_items =
                        Vec::with_capacity(if allow_batch { max_batch } else { 1 });
                    batch_items.push((req, reply));
                    if allow_batch {
                        drain_matching_batch(
                            &mut pending,
                            &registry,
                            &canonical,
                            cfg_bits,
                            max_batch,
                            &mut batch_items,
                        );
                    }

                    let mut load_duration_ns = 0u64;
                    if !cache.contains(&cache_key) {
                        cache.evict_if_full();
                        cache.publish(&ps_snapshot);
                        let t0 = Instant::now();
                        match load_model(base_entry, device, precision) {
                            Ok(loaded) => {
                                load_duration_ns = t0.elapsed().as_nanos() as u64;
                                cache.insert(
                                    cache_key.clone(),
                                    loaded,
                                    build_loaded_info(base_entry, precision),
                                );
                                cache.publish(&ps_snapshot);
                            }
                            Err(e) => {
                                for (_, rep) in batch_items {
                                    let _ = rep.send(Err(e.clone()));
                                }
                                continue;
                            }
                        }
                        // Coalesce any concurrent requests that arrived while the model loaded.
                        while let Ok(msg) = rx.try_recv() {
                            pending.push_back(msg);
                        }
                        if allow_batch {
                            drain_matching_batch(
                                &mut pending,
                                &registry,
                                &canonical,
                                cfg_bits,
                                max_batch,
                                &mut batch_items,
                            );
                        }
                    }

                    let req_refs: Vec<&GenerateRequest> =
                        batch_items.iter().map(|(r, _)| r).collect();
                    let t_eval = Instant::now();
                    let outputs = match cache.get_mut(&cache_key) {
                        Some(model) => match model.select_adapter(adapter_path.as_deref()) {
                            Ok(()) => run_loaded_batch(model, &req_refs, &auto_seed),
                            Err(e) => vec![Err(e); batch_items.len()],
                        },
                        None => vec![
                            Err(EngineError::Internal("cache lookup failed".into()));
                            batch_items.len()
                        ],
                    };
                    let eval_duration_ns = t_eval.elapsed().as_nanos() as u64;
                    cache.record_use(&cache_key, batch_items.len() as u64);
                    cache.publish(&ps_snapshot);

                    for ((_, rep), out) in batch_items.into_iter().zip(outputs) {
                        let resp =
                            out.map(|(text, prompt_tokens, completion_tokens, finish_reason)| {
                                GenerateResponse {
                                    model: canonical.clone(),
                                    text,
                                    prompt_tokens,
                                    completion_tokens,
                                    load_duration_ns,
                                    eval_duration_ns,
                                    finish_reason,
                                }
                            });
                        let _ = rep.send(resp);
                    }
                }
            }
        }
    }

    fn load_model(
        entry: &ModelEntry,
        device: &str,
        precision: ComputePrecision,
    ) -> std::result::Result<LoadedModel, EngineError> {
        let artifact = crate::quantize::load_artifact_for_device(&entry.path, device, precision)
            .map_err(|e| EngineError::Internal(format!("artifact load failed: {e}")))?;
        let base_sha = artifact.manifest_sha256().to_owned();
        match device {
            #[cfg(feature = "cpu")]
            "cpu" => {
                let backend = ncm_backend_cuda::CandleBackend::cpu()
                    .with_precision(precision)
                    .map_err(|e| EngineError::BadRequest(e.to_string()))?;
                load_for_candle(artifact, entry, backend, false, false)
                    .map(|inner| LoadedModel::Candle(LoadedSlot::new(inner, base_sha)))
            }
            #[cfg(feature = "cuda")]
            "cuda" => {
                let backend = ncm_backend_cuda::CandleBackend::cuda(0)
                    .and_then(|b| b.with_precision(precision))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                load_for_candle(artifact, entry, backend, false, false)
                    .map(|inner| LoadedModel::Candle(LoadedSlot::new(inner, base_sha)))
            }
            #[cfg(feature = "cuda-cudnn")]
            "cuda-cudnn" => {
                let is_lm1b = entry.profile.starts_with("lm1b-");
                let backend = if is_lm1b {
                    #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
                    {
                        ncm_backend_cuda::CandleBackend::cuda_lm1b_efficient_experimental(0)
                            .and_then(|b| b.with_precision(precision))
                            .map_err(|e| EngineError::Internal(e.to_string()))?
                    }
                    #[cfg(not(all(feature = "lm1b-experimental", target_os = "linux")))]
                    {
                        ncm_backend_cuda::CandleBackend::cuda_cudnn(0)
                            .and_then(|b| b.with_precision(precision))
                            .map_err(|e| EngineError::Internal(e.to_string()))?
                    }
                } else {
                    ncm_backend_cuda::CandleBackend::cuda_cudnn(0)
                        .and_then(|b| b.with_precision(precision))
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                };
                load_for_candle(artifact, entry, backend, true, true)
                    .map(|inner| LoadedModel::Candle(LoadedSlot::new(inner, base_sha)))
            }
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            "mlx" => {
                let backend = ncm_backend_mlx::MlxBackend::gpu()
                    .and_then(|b| b.with_precision(precision))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                load_for_mlx(artifact, entry, backend)
                    .map(|inner| LoadedModel::Mlx(LoadedSlot::new(inner, base_sha)))
            }
            other => Err(EngineError::BadRequest(format!(
                "unsupported device `{other}` in this build"
            ))),
        }
    }

    #[cfg(any(
        all(feature = "lm1b-experimental", target_os = "linux"),
        all(
            feature = "lm1b-mlx-experimental",
            target_os = "macos",
            target_arch = "aarch64"
        )
    ))]
    type RopeTables = (Vec<f32>, Vec<f32>);

    #[cfg(any(
        all(feature = "lm1b-experimental", target_os = "linux"),
        all(
            feature = "lm1b-mlx-experimental",
            target_os = "macos",
            target_arch = "aarch64"
        )
    ))]
    fn load_lm1b_external_rope(
        entry: &ModelEntry,
    ) -> std::result::Result<Option<RopeTables>, EngineError> {
        if entry.profile.ends_with("-v2") {
            return Ok(None);
        }
        let sidecar = entry
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("lm1b-reference-rope-v1.safetensors");
        let bytes = std::fs::read(&sidecar).map_err(|e| {
            EngineError::Internal(format!(
                "LM1B v1 model `{}` requires sibling rotary sidecar {}: {e}",
                entry.name,
                sidecar.display()
            ))
        })?;
        let rope = lm1b_small_reference_rope(&bytes)
            .map_err(|e| EngineError::Internal(format!("invalid LM1B rotary sidecar: {e}")))?;
        Ok(Some(rope))
    }

    #[cfg(feature = "cpu")]
    fn load_for_candle(
        artifact: Artifact,
        entry: &ModelEntry,
        backend: ncm_backend_cuda::CandleBackend,
        has_cudnn: bool,
        has_lm1b: bool,
    ) -> std::result::Result<LoadedForBackend<ncm_backend_cuda::CandleBackend>, EngineError> {
        let _ = (entry, has_cudnn, has_lm1b);
        match artifact.manifest().profile.as_str() {
            "sudoku-easy-inspection-v1"
            | "sudoku-medium-inspection-v1"
            | "sudoku-hard-inspection-v1" => {
                let model = Generator::from_artifact(backend.clone(), &artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok(LoadedForBackend::Sudoku { backend, model })
            }
            "tinystories-inference-v1" => {
                let decoder = Decoder::from_artifact(&artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok(LoadedForBackend::TinyStories {
                    backend,
                    model,
                    decoder,
                })
            }
            "xsum-inference-v1" | "xsum-inference-v2" => {
                #[cfg(feature = "cuda-cudnn")]
                if has_cudnn {
                    let decoder = Decoder::from_artifact(&artifact)
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                    let model = XsumTeacher::from_owned_artifact(backend.clone(), artifact)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                        .with_prefix_cache(true);
                    return Ok(LoadedForBackend::Xsum {
                        backend,
                        model,
                        decoder,
                        use_device_path: true,
                    });
                }
                Err(EngineError::BadRequest(
                    "XSum serving requires --device cuda-cudnn (or mlx on Apple Silicon)".into(),
                ))
            }
            "wmt14-inference-v1" => {
                #[cfg(feature = "wmt14-experimental")]
                {
                    let decoder = Decoder::from_artifact(&artifact)
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                    let model = Wmt14Teacher::from_owned_artifact(backend.clone(), artifact)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                        .with_prefix_cache(true);
                    Ok(LoadedForBackend::Wmt14 {
                        backend,
                        model,
                        decoder,
                        metal_optimized: false,
                    })
                }
                #[cfg(not(feature = "wmt14-experimental"))]
                Err(EngineError::BadRequest(
                    "WMT14 serving requires --features cuda or cuda-cudnn".into(),
                ))
            }
            p if p.starts_with("lm1b-") => {
                #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
                if has_lm1b {
                    let external_rope = load_lm1b_external_rope(entry)?;
                    let tokenizer = TextTokenizer::from_asset_bytes(
                        artifact
                            .file("tokenizer-manifest.json")
                            .map_err(|e| EngineError::Internal(e.to_string()))?,
                        artifact
                            .file("tokenizer.json")
                            .map_err(|e| EngineError::Internal(e.to_string()))?,
                        artifact
                            .file("tokenizer.LICENSE")
                            .map_err(|e| EngineError::Internal(e.to_string()))?,
                    )
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                    let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                    let model = if let Some((cos, sin)) = external_rope {
                        model
                            .with_reference_rope_diagnostic(&cos, &sin)
                            .map_err(|e| EngineError::Internal(e.to_string()))?
                    } else {
                        model
                    };
                    return Ok(LoadedForBackend::Lm1b {
                        backend,
                        model,
                        tokenizer,
                        use_cuda_device: true,
                    });
                }
                Err(EngineError::BadRequest(
                    "LM1B serving requires --device cuda-cudnn (or mlx on Apple Silicon)".into(),
                ))
            }
            "local-ncm-v1" => {
                let decoder = Decoder::from_artifact(&artifact).ok();
                let model = LocalNcmModel::from_artifact(backend.clone(), &artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?
                    .with_prefix_cache(true);
                Ok(LoadedForBackend::Local {
                    backend,
                    model,
                    decoder,
                })
            }
            other => Err(EngineError::BadRequest(format!(
                "unsupported profile `{other}`"
            ))),
        }
    }

    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    fn load_for_mlx(
        artifact: Artifact,
        entry: &ModelEntry,
        backend: ncm_backend_mlx::MlxBackend,
    ) -> std::result::Result<LoadedForBackend<ncm_backend_mlx::MlxBackend>, EngineError> {
        match artifact.manifest().profile.as_str() {
            "sudoku-easy-inspection-v1"
            | "sudoku-medium-inspection-v1"
            | "sudoku-hard-inspection-v1" => {
                let model = Generator::from_artifact(backend.clone(), &artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok(LoadedForBackend::Sudoku { backend, model })
            }
            "tinystories-inference-v1" => {
                let decoder = Decoder::from_artifact(&artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok(LoadedForBackend::TinyStories {
                    backend,
                    model,
                    decoder,
                })
            }
            "xsum-inference-v1" | "xsum-inference-v2" => {
                let decoder = Decoder::from_artifact(&artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let model = XsumTeacher::from_owned_artifact(backend.clone(), artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?
                    .with_prefix_cache(true);
                Ok(LoadedForBackend::Xsum {
                    backend,
                    model,
                    decoder,
                    use_device_path: false,
                })
            }
            "wmt14-inference-v1" => {
                let decoder = Decoder::from_artifact(&artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let model = Wmt14Teacher::from_owned_artifact(backend.clone(), artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?
                    .with_prefix_cache(true)
                    .with_deferred_head_check_experimental()
                    .with_level_materialization_experimental(5)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok(LoadedForBackend::Wmt14 {
                    backend,
                    model,
                    decoder,
                    metal_optimized: true,
                })
            }
            p if p.starts_with("lm1b-") => {
                let external_rope = load_lm1b_external_rope(entry)?;
                let tokenizer = TextTokenizer::from_asset_bytes(
                    artifact
                        .file("tokenizer-manifest.json")
                        .map_err(|e| EngineError::Internal(e.to_string()))?,
                    artifact
                        .file("tokenizer.json")
                        .map_err(|e| EngineError::Internal(e.to_string()))?,
                    artifact
                        .file("tokenizer.LICENSE")
                        .map_err(|e| EngineError::Internal(e.to_string()))?,
                )
                .map_err(|e| EngineError::Internal(e.to_string()))?;
                let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let model = if let Some((cos, sin)) = external_rope {
                    model
                        .with_reference_rope_diagnostic(&cos, &sin)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                } else {
                    model
                };
                Ok(LoadedForBackend::Lm1b {
                    backend,
                    model,
                    tokenizer,
                    use_cuda_device: false,
                })
            }
            "local-ncm-v1" => {
                let decoder = Decoder::from_artifact(&artifact).ok();
                let model = LocalNcmModel::from_artifact(backend.clone(), &artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?
                    .with_prefix_cache(true);
                Ok(LoadedForBackend::Local {
                    backend,
                    model,
                    decoder,
                })
            }
            other => Err(EngineError::BadRequest(format!(
                "unsupported profile `{other}`"
            ))),
        }
    }

    fn run_loaded_batch(
        loaded: &mut LoadedModel,
        reqs: &[&GenerateRequest],
        auto_seed: &AtomicU64,
    ) -> Vec<GenOutput> {
        match loaded {
            #[cfg(feature = "cpu")]
            LoadedModel::Candle(slot) => run_loaded_batch_backend(&mut slot.inner, reqs, auto_seed),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            LoadedModel::Mlx(slot) => run_loaded_batch_backend(&mut slot.inner, reqs, auto_seed),
        }
    }

    fn cap_ids(mut ids: Vec<u32>, max_tokens: Option<usize>) -> (Vec<u32>, String) {
        if let Some(limit) = max_tokens
            && limit < ids.len()
        {
            ids.truncate(limit);
            return (ids, "length".to_owned());
        }
        (ids, "stop".to_owned())
    }

    fn parse_sudoku_prompt(prompt: &str) -> std::result::Result<Vec<u32>, EngineError> {
        let trimmed = prompt.trim();
        if trimmed.starts_with('[')
            && let Ok(v) = serde_json::from_str::<Vec<u32>>(trimmed)
        {
            return Ok(v);
        }
        let nums: Vec<u32> = trimmed.chars().filter_map(|c| c.to_digit(10)).collect();
        if nums.len() == 81 || nums.len() == 89 {
            return Ok(nums);
        }
        Err(EngineError::BadRequest(
            "Sudoku prompt must be a JSON token-ID array or 81 digits (0-9)".into(),
        ))
    }

    fn split_batch_ids(
        raw_ids: Vec<u32>,
        batch: usize,
    ) -> std::result::Result<Vec<Vec<u32>>, EngineError> {
        if batch == 0 || raw_ids.is_empty() || !raw_ids.len().is_multiple_of(batch) {
            return Err(EngineError::Internal(
                "invalid batched backend output shape".into(),
            ));
        }
        let stride = raw_ids.len() / batch;
        Ok(raw_ids.chunks_exact(stride).map(<[u32]>::to_vec).collect())
    }

    fn run_loaded_batch_backend<B: GeneratorBackend>(
        loaded: &mut LoadedForBackend<B>,
        reqs: &[&GenerateRequest],
        auto_seed: &AtomicU64,
    ) -> Vec<GenOutput> {
        if reqs.is_empty() {
            return Vec::new();
        }
        let mut results: Vec<Option<GenOutput>> = vec![None; reqs.len()];
        match loaded {
            LoadedForBackend::Sudoku { backend, model } => {
                let mut valid: Vec<(usize, Vec<u32>)> = Vec::with_capacity(reqs.len());
                for (idx, req) in reqs.iter().enumerate() {
                    match parse_sudoku_prompt(&req.prompt) {
                        Ok(prefix) => valid.push((idx, prefix)),
                        Err(e) => results[idx] = Some(Err(e)),
                    }
                }
                if !valid.is_empty() {
                    let same_len = valid.iter().all(|(_, p)| p.len() == valid[0].1.len());
                    if same_len {
                        let batch = valid.len();
                        let flat: Vec<u32> =
                            valid.iter().flat_map(|(_, p)| p.iter().copied()).collect();
                        let run_res = model
                            .generate(&flat, batch)
                            .map_err(|e| EngineError::BadRequest(e.to_string()))
                            .and_then(|g| {
                                backend
                                    .read_indices(&g.ids)
                                    .map_err(|e| EngineError::Internal(e.to_string()))
                            })
                            .and_then(|raw| split_batch_ids(raw, batch));
                        match run_res {
                            Ok(rows) => {
                                for ((idx, prefix), ids) in valid.into_iter().zip(rows) {
                                    let completion_tokens = ids.len();
                                    let text = serde_json::to_string(&ids).unwrap_or_default();
                                    results[idx] = Some(Ok((
                                        text,
                                        prefix.len(),
                                        completion_tokens,
                                        "stop".to_owned(),
                                    )));
                                }
                            }
                            Err(e) => {
                                for (idx, _) in valid {
                                    results[idx] = Some(Err(e.clone()));
                                }
                            }
                        }
                    } else {
                        for (idx, prefix) in valid {
                            let res = model
                                .generate(&prefix, 1)
                                .map_err(|e| EngineError::BadRequest(e.to_string()))
                                .and_then(|g| {
                                    backend
                                        .read_indices(&g.ids)
                                        .map_err(|e| EngineError::Internal(e.to_string()))
                                })
                                .map(|ids| {
                                    let ct = ids.len();
                                    let text = serde_json::to_string(&ids).unwrap_or_default();
                                    (text, prefix.len(), ct, "stop".to_owned())
                                });
                            results[idx] = Some(res);
                        }
                    }
                }
            }
            LoadedForBackend::TinyStories {
                backend,
                model,
                decoder,
            } => {
                let mut valid: Vec<(usize, u32)> = Vec::with_capacity(reqs.len());
                for (idx, req) in reqs.iter().enumerate() {
                    let prompt_str = if req.prompt.trim().is_empty() {
                        "Once".to_owned()
                    } else {
                        req.prompt.clone()
                    };
                    match decoder.encode_texts(&[prompt_str]) {
                        Ok(enc) if !enc.is_empty() && !enc[0].is_empty() => {
                            valid.push((idx, enc[0][0]));
                        }
                        Ok(_) => {
                            results[idx] = Some(Err(EngineError::BadRequest(
                                "TinyStories prompt encoded to empty sequence".into(),
                            )));
                        }
                        Err(e) => results[idx] = Some(Err(EngineError::BadRequest(e.to_string()))),
                    }
                }
                if !valid.is_empty() {
                    let batch = valid.len();
                    let prefixes: Vec<u32> = valid.iter().map(|(_, tok)| *tok).collect();
                    let seed = if batch == 1 {
                        reqs[valid[0].0]
                            .seed
                            .unwrap_or_else(|| auto_seed.fetch_add(1, Ordering::Relaxed))
                    } else {
                        auto_seed.fetch_add(1, Ordering::Relaxed)
                    };
                    let run_res = model
                        .sample_seeded(&prefixes, batch, seed)
                        .map_err(|e| EngineError::Internal(e.to_string()))
                        .and_then(|g| {
                            backend
                                .read_indices(&g.ids)
                                .map_err(|e| EngineError::Internal(e.to_string()))
                        })
                        .and_then(|raw| split_batch_ids(raw, batch));
                    match run_res {
                        Ok(rows) => {
                            for ((idx, _), raw_ids) in valid.into_iter().zip(rows) {
                                let (ids, finish) = cap_ids(raw_ids, reqs[idx].max_tokens);
                                let ct = ids.len();
                                let decoded = decoder
                                    .decode_ids(&ids)
                                    .map_err(|e| EngineError::Internal(e.to_string()))
                                    .map(|text| (text, 1, ct, finish));
                                results[idx] = Some(decoded);
                            }
                        }
                        Err(e) => {
                            for (idx, _) in valid {
                                results[idx] = Some(Err(e.clone()));
                            }
                        }
                    }
                }
            }
            #[cfg(any(
                feature = "cuda-cudnn",
                all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
            ))]
            LoadedForBackend::Xsum {
                backend: _,
                model,
                decoder,
                use_device_path,
            } => {
                let mut valid: Vec<(usize, Vec<u32>, usize)> = Vec::with_capacity(reqs.len());
                for (idx, req) in reqs.iter().enumerate() {
                    let cfg_scale = req.cfg_scale.unwrap_or(0.0);
                    if !cfg_scale.is_finite() || cfg_scale < 0.0 {
                        results[idx] = Some(Err(EngineError::BadRequest(
                            "cfg_scale must be finite and nonnegative".into(),
                        )));
                        continue;
                    }
                    match decoder.encode_texts(std::slice::from_ref(&req.prompt)) {
                        Ok(mut enc) if !enc.is_empty() => {
                            let prefix = enc.remove(0);
                            let pt = prefix.iter().filter(|&&id| id != 50257).count().max(1);
                            valid.push((idx, prefix, pt));
                        }
                        Ok(_) => {
                            results[idx] = Some(Err(EngineError::BadRequest(
                                "XSum prompt encoded to empty sequence".into(),
                            )));
                        }
                        Err(e) => results[idx] = Some(Err(EngineError::BadRequest(e.to_string()))),
                    }
                }
                if !valid.is_empty() {
                    let batch = valid.len();
                    let cfg_scale = reqs[valid[0].0].cfg_scale.unwrap_or(0.0);
                    let single_seed = if batch == 1 {
                        reqs[valid[0].0].seed
                    } else {
                        None
                    };
                    let flat: Vec<u32> = valid
                        .iter()
                        .flat_map(|(_, p, _)| p.iter().copied())
                        .collect();
                    let run_res = if *use_device_path {
                        if let Some(seed) = single_seed {
                            model.sample_seeded_device(&flat, batch, cfg_scale, seed)
                        } else {
                            model.generate_top1_cfg_device(&flat, batch, cfg_scale)
                        }
                    } else if let Some(seed) = single_seed {
                        model.sample_seeded_cfg_cpu(&flat, batch, cfg_scale, seed)
                    } else {
                        model.generate_top1_cfg_cpu(&flat, batch, cfg_scale)
                    }
                    .map_err(|e| EngineError::Internal(e.to_string()))
                    .and_then(|g| split_batch_ids(g.ids, batch));
                    match run_res {
                        Ok(rows) => {
                            for ((idx, _, pt), raw_ids) in valid.into_iter().zip(rows) {
                                let (ids, finish) = cap_ids(raw_ids, reqs[idx].max_tokens);
                                let ct = ids.len();
                                let decoded = decoder
                                    .decode_ids(&ids)
                                    .map_err(|e| EngineError::Internal(e.to_string()))
                                    .map(|text| (text, pt, ct, finish));
                                results[idx] = Some(decoded);
                            }
                        }
                        Err(e) => {
                            for (idx, _, _) in valid {
                                results[idx] = Some(Err(e.clone()));
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "wmt14-experimental")]
            LoadedForBackend::Wmt14 {
                backend,
                model,
                decoder,
                metal_optimized,
            } => {
                let mut valid: Vec<(usize, Vec<u32>, usize)> = Vec::with_capacity(reqs.len());
                for (idx, req) in reqs.iter().enumerate() {
                    let cfg_scale = req.cfg_scale.unwrap_or(0.0);
                    if !cfg_scale.is_finite() || cfg_scale < 0.0 {
                        results[idx] = Some(Err(EngineError::BadRequest(
                            "cfg_scale must be finite and nonnegative".into(),
                        )));
                        continue;
                    }
                    match decoder.encode_texts(std::slice::from_ref(&req.prompt)) {
                        Ok(mut enc) if !enc.is_empty() => {
                            let prefix = enc.remove(0);
                            let pt = prefix.iter().filter(|&&id| id != 151643).count().max(1);
                            valid.push((idx, prefix, pt));
                        }
                        Ok(_) => {
                            results[idx] = Some(Err(EngineError::BadRequest(
                                "WMT14 prompt encoded to empty sequence".into(),
                            )));
                        }
                        Err(e) => results[idx] = Some(Err(EngineError::BadRequest(e.to_string()))),
                    }
                }
                if !valid.is_empty() {
                    let batch = valid.len();
                    let cfg_scale = reqs[valid[0].0].cfg_scale.unwrap_or(0.0);
                    let flat: Vec<u32> = valid
                        .iter()
                        .flat_map(|(_, p, _)| p.iter().copied())
                        .collect();
                    let run_res = if *metal_optimized && (cfg_scale == 0.0 || cfg_scale == 1.0) {
                        model
                            .generate_top1_with_tiled_decoder(&flat, batch, cfg_scale, 2)
                            .map_err(|e| EngineError::Internal(e.to_string()))
                            .map(|g| g.ids)
                    } else if cfg_scale == 0.0 {
                        model
                            .generate_top1(&flat, batch)
                            .map_err(|e| EngineError::Internal(e.to_string()))
                            .and_then(|g| {
                                backend
                                    .read_indices(&g.ids)
                                    .map_err(|e| EngineError::Internal(e.to_string()))
                            })
                    } else {
                        model
                            .generate_cfg_top1(&flat, batch, cfg_scale)
                            .map_err(|e| EngineError::Internal(e.to_string()))
                            .and_then(|g| {
                                backend
                                    .read_indices(&g.ids)
                                    .map_err(|e| EngineError::Internal(e.to_string()))
                            })
                    }
                    .and_then(|raw| split_batch_ids(raw, batch));
                    match run_res {
                        Ok(rows) => {
                            for ((idx, _, pt), raw_ids) in valid.into_iter().zip(rows) {
                                let (ids, finish) = cap_ids(raw_ids, reqs[idx].max_tokens);
                                let ct = ids.len();
                                let decoded = decoder
                                    .decode_ids(&ids)
                                    .map_err(|e| EngineError::Internal(e.to_string()))
                                    .map(|text| (text, pt, ct, finish));
                                results[idx] = Some(decoded);
                            }
                        }
                        Err(e) => {
                            for (idx, _, _) in valid {
                                results[idx] = Some(Err(e.clone()));
                            }
                        }
                    }
                }
            }
            #[cfg(any(
                all(feature = "lm1b-experimental", target_os = "linux"),
                all(
                    feature = "lm1b-mlx-experimental",
                    target_os = "macos",
                    target_arch = "aarch64"
                )
            ))]
            LoadedForBackend::Lm1b {
                backend: _,
                model,
                tokenizer,
                use_cuda_device,
            } => {
                // LM1B is unconditional (start token [101]).
                let batch = reqs.len();
                let prefixes = vec![101u32; batch];
                let seed = if batch == 1 {
                    reqs[0]
                        .seed
                        .unwrap_or_else(|| auto_seed.fetch_add(1, Ordering::Relaxed))
                } else {
                    auto_seed.fetch_add(1, Ordering::Relaxed)
                };
                let run_res = if *use_cuda_device {
                    #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
                    {
                        model.sample_seeded_fully_device(&prefixes, batch, seed)
                    }
                    #[cfg(not(all(feature = "lm1b-experimental", target_os = "linux")))]
                    {
                        model.sample_seeded(&prefixes, batch, seed)
                    }
                } else {
                    #[cfg(all(
                        feature = "lm1b-mlx-experimental",
                        target_os = "macos",
                        target_arch = "aarch64"
                    ))]
                    {
                        model.sample_seeded_device_fp32(&prefixes, batch, seed)
                    }
                    #[cfg(not(all(
                        feature = "lm1b-mlx-experimental",
                        target_os = "macos",
                        target_arch = "aarch64"
                    )))]
                    {
                        model.sample_seeded(&prefixes, batch, seed)
                    }
                }
                .map_err(|e| EngineError::Internal(e.to_string()))
                .and_then(|out| split_batch_ids(out.ids, batch));
                match run_res {
                    Ok(rows) => {
                        for (idx, raw_ids) in rows.into_iter().enumerate() {
                            let (ids, finish) = cap_ids(raw_ids, reqs[idx].max_tokens);
                            let ct = ids.len();
                            let decoded = tokenizer
                                .decode_lm1b_flm(&ids)
                                .map_err(|e| EngineError::Internal(e.to_string()))
                                .map(|text| (text, 1, ct, finish));
                            results[idx] = Some(decoded);
                        }
                    }
                    Err(e) => {
                        for slot in &mut results {
                            *slot = Some(Err(e.clone()));
                        }
                    }
                }
            }
            LoadedForBackend::Local {
                backend,
                model,
                decoder,
            } => {
                let prefix_len = model.config().prefix_len;
                let mut valid: Vec<(usize, Vec<u32>, usize)> = Vec::with_capacity(reqs.len());
                for (idx, req) in reqs.iter().enumerate() {
                    let parsed = if let Some(dec) = decoder {
                        dec.encode_texts(std::slice::from_ref(&req.prompt))
                            .map_err(|e| EngineError::BadRequest(e.to_string()))
                            .and_then(|mut enc| {
                                enc.pop().ok_or_else(|| {
                                    EngineError::BadRequest("empty encoded prefix".into())
                                })
                            })
                    } else {
                        serde_json::from_str::<Vec<u32>>(&req.prompt).map_err(|_| {
                            EngineError::BadRequest(
                                "this local-ncm-v1 model has no tokenizer.json; send a JSON token-ID array"
                                    .into(),
                            )
                        })
                    };
                    match parsed {
                        Ok(ids) if ids.len() == prefix_len => {
                            let pt = ids.len();
                            valid.push((idx, ids, pt));
                        }
                        Ok(ids) => {
                            results[idx] = Some(Err(EngineError::BadRequest(format!(
                                "local-ncm-v1 expects prefix length {prefix_len}, got {}",
                                ids.len()
                            ))));
                        }
                        Err(e) => results[idx] = Some(Err(e)),
                    }
                }
                if !valid.is_empty() {
                    let batch = valid.len();
                    let seed = if batch == 1 {
                        reqs[valid[0].0].seed
                    } else {
                        None
                    };
                    let flat: Vec<u32> = valid
                        .iter()
                        .flat_map(|(_, p, _)| p.iter().copied())
                        .collect();
                    let run_res = model
                        .generate(&flat, batch, seed)
                        .map_err(|e| EngineError::Internal(e.to_string()))
                        .and_then(|g| {
                            backend
                                .read_indices(&g.ids)
                                .map_err(|e| EngineError::Internal(e.to_string()))
                        })
                        .and_then(|raw| split_batch_ids(raw, batch));
                    match run_res {
                        Ok(rows) => {
                            for ((idx, _, pt), raw_ids) in valid.into_iter().zip(rows) {
                                let (ids, finish) = cap_ids(raw_ids, reqs[idx].max_tokens);
                                let ct = ids.len();
                                let decoded = if let Some(dec) = decoder {
                                    dec.decode_ids(&ids)
                                        .map_err(|e| EngineError::Internal(e.to_string()))
                                } else {
                                    Ok(serde_json::to_string(&ids).unwrap_or_default())
                                }
                                .map(|text| (text, pt, ct, finish));
                                results[idx] = Some(decoded);
                            }
                        }
                        Err(e) => {
                            for (idx, _, _) in valid {
                                results[idx] = Some(Err(e.clone()));
                            }
                        }
                    }
                }
            }
        }
        results
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    Err(EngineError::Internal("missing batch slot result".into()))
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_adapter_spec, parse_override, parse_pull_args, parse_run_args, parse_serve_args,
        vram_estimate_bytes,
    };
    use ncm_core::ComputePrecision;

    fn args(vals: &[&str]) -> Vec<String> {
        vals.iter().map(|s| (*s).into()).collect()
    }

    #[test]
    fn serve_arg_parser_accepts_defaults_and_options() {
        let cfg = parse_serve_args(&args(&[
            "ncm",
            "serve",
            "--models-dir",
            "/tmp/models",
            "--model",
            "custom=/tmp/my-model",
            "--adapter",
            "pirate=custom:/tmp/pirate-adapter",
            "--precision",
            "q8",
            "--host",
            "127.0.0.1",
            "--port",
            "18080",
            "--preload",
            "custom",
            "--max-loaded",
            "2",
            "--max-batch",
            "16",
        ]))
        .unwrap();
        assert_eq!(cfg.port, 18080);
        assert_eq!(cfg.precision, ComputePrecision::Q8);
        assert_eq!(cfg.max_loaded, 2);
        assert_eq!(cfg.max_batch, 16);
        assert_eq!(cfg.overrides.len(), 1);
        assert_eq!(cfg.overrides[0].0, "custom");
        assert_eq!(cfg.adapters.len(), 1);
        assert_eq!(cfg.adapters[0].0, "pirate");
        assert_eq!(cfg.adapters[0].1, "custom");
        assert_eq!(
            cfg.adapters[0].2,
            std::path::PathBuf::from("/tmp/pirate-adapter")
        );
        assert!(parse_adapter_spec("bad-spec").is_err());
        assert!(parse_adapter_spec("alias=missing-colon").is_err());
        assert!(parse_override("missing-equals").is_err());
        assert!(parse_serve_args(&args(&["ncm", "serve", "--max-loaded", "0"])).is_err());
        assert!(parse_serve_args(&args(&["ncm", "serve", "--max-batch", "0"])).is_err());
        assert!(parse_serve_args(&args(&["ncm", "serve", "--max-batch", "33"])).is_err());
    }

    #[test]
    fn run_arg_parser_supports_oneshot_and_repl() {
        let repl = parse_run_args(&args(&["ncm", "run", "wmt14", "--precision", "q4"])).unwrap();
        assert_eq!(repl.model, "wmt14");
        assert!(repl.prompt.is_none());
        assert_eq!(repl.precision, ComputePrecision::Q4);

        let oneshot = parse_run_args(&args(&[
            "ncm",
            "run",
            "wmt14",
            "Guten Morgen!",
            "--precision",
            "fp16",
            "--cfg",
            "1.0",
            "--seed",
            "7",
            "--max-tokens",
            "32",
        ]))
        .unwrap();
        assert_eq!(oneshot.prompt.as_deref(), Some("Guten Morgen!"));
        assert_eq!(oneshot.cfg_scale, Some(1.0));
        assert_eq!(oneshot.seed, Some(7));
        assert_eq!(oneshot.max_tokens, Some(32));
        assert!(parse_run_args(&args(&["ncm", "run"])).is_err());
        assert!(parse_run_args(&args(&["ncm", "run", "wmt14", "--max-tokens", "0"])).is_err());
    }

    #[test]
    fn pull_arg_parser_validates_list_and_model() {
        let list = parse_pull_args(&args(&["ncm", "pull", "--list"])).unwrap();
        assert!(list.list);
        assert!(list.model.is_none());

        let pull = parse_pull_args(&args(&[
            "ncm",
            "pull",
            "wmt14",
            "--models-dir",
            "/tmp/out",
            "--output",
            "/tmp/custom-wmt14",
            "--dry-run",
        ]))
        .unwrap();
        assert!(!pull.list);
        assert!(!pull.python_legacy);
        assert_eq!(pull.model.as_deref(), Some("wmt14"));
        assert_eq!(
            pull.output,
            Some(std::path::PathBuf::from("/tmp/custom-wmt14"))
        );
        assert!(pull.dry_run);
        let leg = parse_pull_args(&args(&["ncm", "pull", "wmt14", "--python-legacy"])).unwrap();
        assert!(leg.python_legacy);
        assert!(parse_pull_args(&args(&["ncm", "pull"])).is_err());
    }

    #[test]
    fn vram_estimate_scales_monotonically_with_precision() {
        let fp32 = vram_estimate_bytes(1_000_000_000, ComputePrecision::Fp32);
        let fp16 = vram_estimate_bytes(1_000_000_000, ComputePrecision::Fp16);
        let q8 = vram_estimate_bytes(1_000_000_000, ComputePrecision::Q8);
        let q4 = vram_estimate_bytes(1_000_000_000, ComputePrecision::Q4);
        assert!(fp32 > fp16 && fp16 > q8 && q8 > q4 && q4 > 0);
    }
}
