//! Local model listing (`ncm list`) and OpenAI/Ollama-compatible HTTP serving (`ncm serve`).
use crate::Result;
use ncm_core::ComputePrecision;
use ncm_serve::ModelRegistry;
use std::path::{Path, PathBuf};

const LIST_USAGE: &str = "usage: ncm list [--models-dir DIR] [--model NAME=PATH]... [--precision fp32|bf16|fp16] [--json]";

const SERVE_USAGE: &str = "\
usage: ncm serve [OPTIONS]

Options:
  --models-dir DIR           Directory containing NCM artifacts (default: artifacts/canonical or .)
  --model NAME=PATH          Register a model artifact under NAME (repeatable)
  --device DEV               Explicit device: cuda-cudnn, cuda, mlx, or cpu (default: compiled best)
  --precision fp32|bf16|fp16 Compute/weight precision (default: fp32)
  --host ADDR                Bind host (default: 127.0.0.1)
  --port PORT                Bind port (default: 11434)
  --preload NAME             Load and warm-cache one model before accepting requests
  --max-loaded N             Max simultaneously resident models in LRU cache (default: 1, max: 16)
";

pub(super) const LIST_HELP: &str = "\
List discovered local NCM artifacts.

Usage:
  ncm list [--models-dir DIR] [--model NAME=PATH]... [--json]

Scans --models-dir (default: artifacts/canonical if present, else .) for directories
containing a valid manifest.json with a servable NCM profile.
";

pub(super) const SERVE_HELP: &str = "\
Serve local NCM models over an OpenAI- and Ollama-compatible HTTP/1.1 API.

Usage:
  ncm serve [OPTIONS]

Options:
  --models-dir DIR           Directory containing NCM artifacts (default: artifacts/canonical or .)
  --model NAME=PATH          Register a model artifact under NAME (repeatable)
  --device DEV               Explicit device: cuda-cudnn, cuda, mlx, or cpu
  --precision fp32|bf16|fp16 Compute and weight precision (default: fp32)
  --host ADDR                Bind address (default: 127.0.0.1)
  --port PORT                Bind port (default: 11434)
  --preload NAME             Preload one model on startup
  --max-loaded N             Max resident models in LRU cache (default: 1, max: 16)

OpenAI endpoints:
  GET  /v1/models
  POST /v1/completions        (stream: false|true SSE)
  POST /v1/chat/completions   (stream: false|true SSE)

Ollama endpoints:
  GET  /  |  GET /health  |  GET /api/tags  |  POST /api/show
  POST /api/generate          (stream: true|false NDJSON)
  POST /api/chat              (stream: true|false NDJSON)
";

fn default_models_dir() -> Option<PathBuf> {
    let canonical = Path::new("artifacts/canonical");
    if canonical.is_dir() {
        Some(canonical.to_path_buf())
    } else {
        Some(PathBuf::from("."))
    }
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
    println!(
        "{:<24} {:<26} {:>10}  {:<5} PATH",
        "NAME", "PROFILE", "SIZE", "TEXT"
    );
    for m in &models {
        println!(
            "{:<24} {:<26} {:>10}  {:<5} {}",
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
    device: String,
    precision: ComputePrecision,
    host: String,
    port: u16,
    preload: Option<String>,
    max_loaded: usize,
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
        device: default_device().to_owned(),
        precision: ComputePrecision::Fp32,
        host: "127.0.0.1".to_owned(),
        port: 11434,
        preload: None,
        max_loaded: 1,
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
    let registry = ModelRegistry::discover(
        cfg.models_dir.as_deref(),
        &cfg.overrides,
        cfg.precision.as_str(),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

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

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
mod worker {
    use super::{ModelRegistry, Result, ServeConfig, format_size};
    use crate::decode::Decoder;
    use ncm_core::{ComputePrecision, GeneratorBackend};
    use ncm_io::Artifact;
    use ncm_model::{Generator, LocalNcmModel, TinyStoriesTeacher};
    use ncm_serve::ModelEntry;
    use ncm_serve::{EngineError, GenerateRequest, GenerateResponse, InferenceEngine, Server};
    use std::{
        collections::{HashMap, VecDeque},
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
            mpsc,
        },
        thread,
        time::Instant,
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

    enum WorkMsg {
        Generate {
            req: GenerateRequest,
            reply: mpsc::Sender<std::result::Result<GenerateResponse, EngineError>>,
        },
        Preload {
            name: String,
            reply: mpsc::Sender<std::result::Result<(), EngineError>>,
        },
    }

    struct LocalEngine {
        registry: Arc<ModelRegistry>,
        tx: mpsc::Sender<WorkMsg>,
    }

    impl InferenceEngine for LocalEngine {
        fn list_models(&self) -> Vec<ModelEntry> {
            self.registry.list()
        }
        fn get_model(&self, name: &str) -> Option<ModelEntry> {
            self.registry.resolve(name).cloned()
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

    pub(super) fn start_server(cfg: ServeConfig, registry: ModelRegistry) -> Result<()> {
        validate_device_precision(&cfg.device, cfg.precision)?;
        let registry = Arc::new(registry);
        let worker_registry = Arc::clone(&registry);
        let worker_device = cfg.device.clone();
        let worker_precision = cfg.precision;
        let max_loaded = cfg.max_loaded;
        let (tx, rx) = mpsc::channel::<WorkMsg>();

        let _worker = thread::spawn(move || {
            run_worker_loop(
                worker_registry,
                &worker_device,
                worker_precision,
                max_loaded,
                rx,
            );
        });

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

        let bind_addr = format!("{}:{}", cfg.host, cfg.port);
        let engine = LocalEngine {
            registry: Arc::clone(&registry),
            tx,
        };
        let server = Server::bind(&bind_addr, engine)?;
        let actual = server.local_addr()?;

        eprintln!("ncm serve listening on http://{actual}");
        eprintln!(
            "  device: {}  precision: {}  max-loaded: {}",
            cfg.device, cfg.precision, cfg.max_loaded
        );
        eprintln!(
            "  OpenAI API:  http://{actual}/v1/models  /v1/completions  /v1/chat/completions"
        );
        eprintln!("  Ollama API:  http://{actual}/api/tags   /api/show  /api/generate  /api/chat");
        eprintln!("  Models ({}):", registry.list().len());
        for m in registry.list() {
            eprintln!(
                "    - {:<20} ({}, {})",
                m.name,
                m.profile,
                format_size(m.size_bytes)
            );
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
        Candle(LoadedForBackend<ncm_backend_cuda::CandleBackend>),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        Mlx(LoadedForBackend<ncm_backend_mlx::MlxBackend>),
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

    struct LruCache {
        max_loaded: usize,
        order: VecDeque<String>,
        items: HashMap<String, LoadedModel>,
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
        fn get_mut(&mut self, key: &str) -> Option<&mut LoadedModel> {
            if self.items.contains_key(key) {
                self.touch(key);
                self.items.get_mut(key)
            } else {
                None
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
        fn insert(&mut self, key: String, val: LoadedModel) {
            self.touch(&key);
            self.items.insert(key, val);
        }
    }

    fn run_worker_loop(
        registry: Arc<ModelRegistry>,
        device: &str,
        precision: ComputePrecision,
        max_loaded: usize,
        rx: mpsc::Receiver<WorkMsg>,
    ) {
        let mut cache = LruCache::new(max_loaded);
        let auto_seed = AtomicU64::new(42);
        while let Ok(msg) = rx.recv() {
            match msg {
                WorkMsg::Preload { name, reply } => {
                    let res = match registry.resolve(&name) {
                        None => Err(EngineError::NotFound(format!("model `{name}` not found"))),
                        Some(entry) => {
                            if cache.get_mut(&entry.name).is_some() {
                                Ok(())
                            } else {
                                cache.evict_if_full();
                                match load_model(entry, device, precision) {
                                    Ok(loaded) => {
                                        cache.insert(entry.name.clone(), loaded);
                                        Ok(())
                                    }
                                    Err(e) => Err(e),
                                }
                            }
                        }
                    };
                    let _ = reply.send(res);
                }
                WorkMsg::Generate { req, reply } => {
                    let res =
                        handle_generate(&registry, &mut cache, device, precision, &auto_seed, req);
                    let _ = reply.send(res);
                }
            }
        }
    }

    fn handle_generate(
        registry: &ModelRegistry,
        cache: &mut LruCache,
        device: &str,
        precision: ComputePrecision,
        auto_seed: &AtomicU64,
        req: GenerateRequest,
    ) -> std::result::Result<GenerateResponse, EngineError> {
        let Some(entry) = registry.resolve(&req.model) else {
            return Err(EngineError::NotFound(format!(
                "model `{}` not found",
                req.model
            )));
        };
        let canonical = entry.name.clone();
        let mut load_duration_ns = 0u64;
        if cache.get_mut(&canonical).is_none() {
            // Evict before loading so GPU VRAM is freed prior to uploading the new artifact.
            cache.evict_if_full();
            let t0 = Instant::now();
            let loaded = load_model(entry, device, precision)?;
            load_duration_ns = t0.elapsed().as_nanos() as u64;
            cache.insert(canonical.clone(), loaded);
        }
        let model = cache
            .get_mut(&canonical)
            .ok_or_else(|| EngineError::Internal("cache lookup failed".into()))?;
        let t_eval = Instant::now();
        let (text, prompt_tokens, completion_tokens, finish_reason) =
            run_loaded(model, &req, auto_seed)?;
        let eval_duration_ns = t_eval.elapsed().as_nanos() as u64;
        Ok(GenerateResponse {
            model: canonical,
            text,
            prompt_tokens,
            completion_tokens,
            load_duration_ns,
            eval_duration_ns,
            finish_reason,
        })
    }

    fn load_model(
        entry: &ModelEntry,
        device: &str,
        precision: ComputePrecision,
    ) -> std::result::Result<LoadedModel, EngineError> {
        let artifact = Artifact::load_inference_streaming(&entry.path)
            .map_err(|e| EngineError::Internal(format!("artifact load failed: {e}")))?;
        match device {
            #[cfg(feature = "cpu")]
            "cpu" => {
                let backend = ncm_backend_cuda::CandleBackend::cpu()
                    .with_precision(precision)
                    .map_err(|e| EngineError::BadRequest(e.to_string()))?;
                load_for_candle(artifact, entry, backend, false, false).map(LoadedModel::Candle)
            }
            #[cfg(feature = "cuda")]
            "cuda" => {
                let backend = ncm_backend_cuda::CandleBackend::cuda(0)
                    .and_then(|b| b.with_precision(precision))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                load_for_candle(artifact, entry, backend, false, false).map(LoadedModel::Candle)
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
                load_for_candle(artifact, entry, backend, true, true).map(LoadedModel::Candle)
            }
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            "mlx" => {
                let backend = ncm_backend_mlx::MlxBackend::gpu()
                    .and_then(|b| b.with_precision(precision))
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                load_for_mlx(artifact, entry, backend).map(LoadedModel::Mlx)
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
                    let model = XsumTeacher::from_owned_artifact(backend, artifact)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                        .with_prefix_cache(true);
                    return Ok(LoadedForBackend::Xsum {
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
                    let model = Lm1bSmallTeacher::from_artifact(backend, &artifact)
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                    let model = if let Some((cos, sin)) = external_rope {
                        model
                            .with_reference_rope_diagnostic(&cos, &sin)
                            .map_err(|e| EngineError::Internal(e.to_string()))?
                    } else {
                        model
                    };
                    return Ok(LoadedForBackend::Lm1b {
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
                let model = XsumTeacher::from_owned_artifact(backend, artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?
                    .with_prefix_cache(true);
                Ok(LoadedForBackend::Xsum {
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
                let model = Lm1bSmallTeacher::from_artifact(backend, &artifact)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let model = if let Some((cos, sin)) = external_rope {
                    model
                        .with_reference_rope_diagnostic(&cos, &sin)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                } else {
                    model
                };
                Ok(LoadedForBackend::Lm1b {
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

    fn run_loaded(
        loaded: &mut LoadedModel,
        req: &GenerateRequest,
        auto_seed: &AtomicU64,
    ) -> std::result::Result<(String, usize, usize, String), EngineError> {
        match loaded {
            #[cfg(feature = "cpu")]
            LoadedModel::Candle(inner) => run_loaded_backend(inner, req, auto_seed),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            LoadedModel::Mlx(inner) => run_loaded_backend(inner, req, auto_seed),
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

    fn run_loaded_backend<B: GeneratorBackend>(
        loaded: &mut LoadedForBackend<B>,
        req: &GenerateRequest,
        auto_seed: &AtomicU64,
    ) -> std::result::Result<(String, usize, usize, String), EngineError> {
        match loaded {
            LoadedForBackend::Sudoku { backend, model } => {
                let prefix = parse_sudoku_prompt(&req.prompt)?;
                let prompt_tokens = prefix.len();
                let generated = model
                    .generate(&prefix, 1)
                    .map_err(|e| EngineError::BadRequest(e.to_string()))?;
                let ids = backend
                    .read_indices(&generated.ids)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let completion_tokens = ids.len();
                let text = serde_json::to_string(&ids).unwrap_or_default();
                Ok((text, prompt_tokens, completion_tokens, "stop".to_owned()))
            }
            LoadedForBackend::TinyStories {
                backend,
                model,
                decoder,
            } => {
                let prompt_str = if req.prompt.trim().is_empty() {
                    "Once".to_owned()
                } else {
                    req.prompt.clone()
                };
                let encoded = decoder
                    .encode_texts(&[prompt_str])
                    .map_err(|e| EngineError::BadRequest(e.to_string()))?;
                let seed = req
                    .seed
                    .unwrap_or_else(|| auto_seed.fetch_add(1, Ordering::Relaxed));
                let generated = model
                    .sample_seeded(&encoded[0], 1, seed)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let raw_ids = backend
                    .read_indices(&generated.ids)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let (ids, finish) = cap_ids(raw_ids, req.max_tokens);
                let completion_tokens = ids.len();
                let text = decoder
                    .decode_ids(&ids)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok((text, 1, completion_tokens, finish))
            }
            #[cfg(any(
                feature = "cuda-cudnn",
                all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
            ))]
            LoadedForBackend::Xsum {
                model,
                decoder,
                use_device_path,
            } => {
                let encoded = decoder
                    .encode_texts(std::slice::from_ref(&req.prompt))
                    .map_err(|e| EngineError::BadRequest(e.to_string()))?;
                let prefix = &encoded[0];
                let prompt_tokens = prefix.iter().filter(|&&id| id != 50257).count().max(1);
                let cfg_scale = req.cfg_scale.unwrap_or(0.0);
                if !cfg_scale.is_finite() || cfg_scale < 0.0 {
                    return Err(EngineError::BadRequest(
                        "cfg_scale must be finite and nonnegative".into(),
                    ));
                }
                let generated = if *use_device_path {
                    if let Some(seed) = req.seed {
                        model.sample_seeded_device(prefix, 1, cfg_scale, seed)
                    } else {
                        model.generate_top1_cfg_device(prefix, 1, cfg_scale)
                    }
                } else if let Some(seed) = req.seed {
                    model.sample_seeded_cfg_cpu(prefix, 1, cfg_scale, seed)
                } else {
                    model.generate_top1_cfg_cpu(prefix, 1, cfg_scale)
                }
                .map_err(|e| EngineError::Internal(e.to_string()))?;
                let (ids, finish) = cap_ids(generated.ids, req.max_tokens);
                let completion_tokens = ids.len();
                let text = decoder
                    .decode_ids(&ids)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok((text, prompt_tokens, completion_tokens, finish))
            }
            #[cfg(feature = "wmt14-experimental")]
            LoadedForBackend::Wmt14 {
                backend,
                model,
                decoder,
                metal_optimized,
            } => {
                let encoded = decoder
                    .encode_texts(std::slice::from_ref(&req.prompt))
                    .map_err(|e| EngineError::BadRequest(e.to_string()))?;
                let prefix = &encoded[0];
                let prompt_tokens = prefix.iter().filter(|&&id| id != 151643).count().max(1);
                let cfg_scale = req.cfg_scale.unwrap_or(0.0);
                if !cfg_scale.is_finite() || cfg_scale < 0.0 {
                    return Err(EngineError::BadRequest(
                        "cfg_scale must be finite and nonnegative".into(),
                    ));
                }
                let raw_ids = if *metal_optimized && (cfg_scale == 0.0 || cfg_scale == 1.0) {
                    model
                        .generate_top1_with_tiled_decoder(prefix, 1, cfg_scale, 2)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                        .ids
                } else if cfg_scale == 0.0 {
                    let g = model
                        .generate_top1(prefix, 1)
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                    backend
                        .read_indices(&g.ids)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                } else {
                    let g = model
                        .generate_cfg_top1(prefix, 1, cfg_scale)
                        .map_err(|e| EngineError::Internal(e.to_string()))?;
                    backend
                        .read_indices(&g.ids)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                };
                let (ids, finish) = cap_ids(raw_ids, req.max_tokens);
                let completion_tokens = ids.len();
                let text = decoder
                    .decode_ids(&ids)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok((text, prompt_tokens, completion_tokens, finish))
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
                model,
                tokenizer,
                use_cuda_device,
            } => {
                // LM1B is unconditional (start token [101]).
                let seed = req
                    .seed
                    .unwrap_or_else(|| auto_seed.fetch_add(1, Ordering::Relaxed));
                let output = if *use_cuda_device {
                    #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
                    {
                        model.sample_seeded_fully_device(&[101], 1, seed)
                    }
                    #[cfg(not(all(feature = "lm1b-experimental", target_os = "linux")))]
                    {
                        model.sample_seeded(&[101], 1, seed)
                    }
                } else {
                    #[cfg(all(
                        feature = "lm1b-mlx-experimental",
                        target_os = "macos",
                        target_arch = "aarch64"
                    ))]
                    {
                        model.sample_seeded_device_fp32(&[101], 1, seed)
                    }
                    #[cfg(not(all(
                        feature = "lm1b-mlx-experimental",
                        target_os = "macos",
                        target_arch = "aarch64"
                    )))]
                    {
                        model.sample_seeded(&[101], 1, seed)
                    }
                }
                .map_err(|e| EngineError::Internal(e.to_string()))?;
                let (ids, finish) = cap_ids(output.ids, req.max_tokens);
                let completion_tokens = ids.len();
                let text = tokenizer
                    .decode_lm1b_flm(&ids)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                Ok((text, 1, completion_tokens, finish))
            }
            LoadedForBackend::Local {
                backend,
                model,
                decoder,
            } => {
                let (prefix, prompt_tokens) = if let Some(dec) = decoder {
                    let encoded = dec
                        .encode_texts(std::slice::from_ref(&req.prompt))
                        .map_err(|e| EngineError::BadRequest(e.to_string()))?;
                    let len = encoded[0].len();
                    (encoded.into_iter().next().unwrap(), len)
                } else {
                    let ids: Vec<u32> = serde_json::from_str(&req.prompt).map_err(|_| {
                        EngineError::BadRequest(
                            "this local-ncm-v1 model has no tokenizer.json; send a JSON token-ID array"
                                .into(),
                        )
                    })?;
                    let len = ids.len();
                    (ids, len)
                };
                let generated = model
                    .generate(&prefix, 1, req.seed)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let raw_ids = backend
                    .read_indices(&generated.ids)
                    .map_err(|e| EngineError::Internal(e.to_string()))?;
                let (ids, finish) = cap_ids(raw_ids, req.max_tokens);
                let completion_tokens = ids.len();
                let text = if let Some(dec) = decoder {
                    dec.decode_ids(&ids)
                        .map_err(|e| EngineError::Internal(e.to_string()))?
                } else {
                    serde_json::to_string(&ids).unwrap_or_default()
                };
                Ok((text, prompt_tokens, completion_tokens, finish))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_override, parse_serve_args};

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
            "--precision",
            "fp16",
            "--host",
            "127.0.0.1",
            "--port",
            "18080",
            "--preload",
            "custom",
            "--max-loaded",
            "2",
        ]))
        .unwrap();
        assert_eq!(cfg.port, 18080);
        assert_eq!(cfg.precision, ncm_core::ComputePrecision::Fp16);
        assert_eq!(cfg.max_loaded, 2);
        assert_eq!(cfg.overrides.len(), 1);
        assert_eq!(cfg.overrides[0].0, "custom");
        assert!(parse_override("missing-equals").is_err());
        assert!(parse_serve_args(&args(&["ncm", "serve", "--max-loaded", "0"])).is_err());
    }
}
