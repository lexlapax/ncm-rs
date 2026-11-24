//! Native Rust training and fine-tuning for `local-ncm-v1` codecs (RVQ + EMA) and generators.
//!
//! Supports reverse-mode autodiff on CPU and NVIDIA CUDA (via Candle) and Apple Silicon Metal
//! (via MLX `value_and_grad`), deterministic checkpoint save/resume, and direct verified
//! `local-ncm-v1` artifact export loadable by `ncm inspect`, `ncm sample`, `ncm run`, and
//! `ncm serve`.

pub mod artifact;
#[cfg(feature = "cpu")]
pub mod candle_backend;
pub mod data;
pub mod ema;
pub mod lora;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx_backend;
pub mod schedule;

use artifact::{export_local_ncm_artifact, load_checkpoint, load_local_artifact_weights};
use data::TokenDataset;
use ncm_core::LocalNcmConfig;
use ncm_tokenizers::LocalTokenizer;
use schedule::{init_codec_weights, init_generator_weights};
use serde::{Deserialize, Serialize};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

/// Training error variants.
#[derive(Debug, Clone)]
pub enum TrainError {
    Config(String),
    Data(String),
    Artifact(String),
    Backend(String),
    Io(String),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(m) => write!(f, "config error: {m}"),
            Self::Data(m) => write!(f, "data error: {m}"),
            Self::Artifact(m) => write!(f, "artifact error: {m}"),
            Self::Backend(m) => write!(f, "backend error: {m}"),
            Self::Io(m) => write!(f, "I/O error: {m}"),
        }
    }
}

impl std::error::Error for TrainError {}

/// Convenience result alias for `ncm-train`.
pub type Result<T> = std::result::Result<T, TrainError>;

/// Which training stage(s) to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainStage {
    /// Train the multiscale RVQ codec (straight-through + EMA codebooks + dead-code restarts).
    Codec,
    /// Train or fine-tune the multiscale generator against a frozen codec.
    Generator,
    /// Train the codec first, then train the generator against the resulting frozen codec.
    All,
}

impl TrainStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Codec => "codec",
            Self::Generator => "generator",
            Self::All => "all",
        }
    }
}

impl FromStr for TrainStage {
    type Err = TrainError;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "codec" => Ok(Self::Codec),
            "generator" => Ok(Self::Generator),
            "all" => Ok(Self::All),
            other => Err(TrainError::Config(format!(
                "unsupported stage `{other}` (expected codec, generator, or all)"
            ))),
        }
    }
}

/// Hyperparameters controlling optimization, EMA codebooks, and checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub stage: TrainStage,
    pub steps: usize,
    pub codec_steps: Option<usize>,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    pub commitment_weight: f64,
    pub ema_decay: f64,
    pub ema_eps: f64,
    pub dead_code_threshold: f64,
    pub seed: u64,
    pub checkpoint_every: usize,
    /// LoRA adapter rank for generator fine-tuning (`0` = full fine-tuning).
    #[serde(default)]
    pub lora_rank: usize,
    /// LoRA scaling alpha (`0.0` = default `2.0 * lora_rank` when `lora_rank > 0`).
    #[serde(default)]
    pub lora_alpha: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            stage: TrainStage::All,
            steps: 100,
            codec_steps: None,
            batch_size: 8,
            learning_rate: 1e-3,
            min_lr: 1e-5,
            warmup_steps: 10,
            weight_decay: 0.01,
            grad_clip_norm: 1.0,
            commitment_weight: 0.25,
            ema_decay: 0.99,
            ema_eps: 1e-5,
            dead_code_threshold: 0.05,
            seed: 42,
            checkpoint_every: 0,
            lora_rank: 0,
            lora_alpha: 0.0,
        }
    }
}

impl TrainConfig {
    /// Resolved LoRA alpha (`2.0 * lora_rank` when `lora_alpha == 0.0`).
    pub fn effective_lora_alpha(&self) -> f64 {
        if self.lora_rank == 0 {
            0.0
        } else if self.lora_alpha > 0.0 {
            self.lora_alpha
        } else {
            (self.lora_rank as f64) * 2.0
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.steps == 0 {
            return Err(TrainError::Config("--steps must be >= 1".into()));
        }
        if self.batch_size == 0 || self.batch_size > 256 {
            return Err(TrainError::Config("--batch must be 1..=256".into()));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(TrainError::Config("--lr must be finite and > 0".into()));
        }
        if !self.min_lr.is_finite() || self.min_lr < 0.0 || self.min_lr > self.learning_rate {
            return Err(TrainError::Config("min_lr must be in [0, lr]".into()));
        }
        if !(0.0..1.0).contains(&self.ema_decay) {
            return Err(TrainError::Config("ema_decay must be in [0, 1)".into()));
        }
        if self.lora_rank > 0 {
            if self.stage != TrainStage::Generator {
                return Err(TrainError::Config(
                    "--lora-rank is only supported for --stage generator".into(),
                ));
            }
            if self.lora_rank > 256 {
                return Err(TrainError::Config("--lora-rank must be 1..=256".into()));
            }
            let alpha = self.effective_lora_alpha();
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(TrainError::Config(
                    "--lora-alpha must be finite and > 0".into(),
                ));
            }
        } else if self.lora_alpha < 0.0 || !self.lora_alpha.is_finite() {
            return Err(TrainError::Config(
                "--lora-alpha must be finite and >= 0".into(),
            ));
        }
        Ok(())
    }
}

/// Metrics recorded for each completed optimization step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub stage: TrainStage,
    pub step: usize,
    pub loss: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recon_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_loss: Option<f64>,
    pub grad_norm: f64,
    pub learning_rate: f64,
    pub restarted_codes: usize,
    pub step_ms: f64,
}

/// Full summary returned after training and verified `local-ncm-v1` export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainReport {
    pub schema: String,
    pub device: String,
    pub stage: TrainStage,
    pub output_artifact: String,
    pub dataset_samples: usize,
    pub total_seconds: f64,
    pub codec_initial_loss: Option<f64>,
    pub codec_final_loss: Option<f64>,
    pub generator_initial_loss: Option<f64>,
    pub generator_final_loss: Option<f64>,
    pub steps_completed: usize,
    pub mean_step_ms: f64,
    #[serde(default)]
    pub lora_rank: usize,
    #[serde(default)]
    pub trainable_params: usize,
    #[serde(default)]
    pub total_params: usize,
    pub history: Vec<StepMetrics>,
}

/// Complete job description for [`run_training_job`].
#[derive(Debug, Clone)]
pub struct TrainJob {
    pub config_path: Option<PathBuf>,
    pub init_from: Option<PathBuf>,
    pub data_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub checkpoint_dir: Option<PathBuf>,
    pub resume_dir: Option<PathBuf>,
    pub device: String,
    pub overwrite: bool,
    pub train_cfg: TrainConfig,
}

/// Execute an end-to-end training or fine-tuning job and export a verified `local-ncm-v1` artifact.
pub fn run_training_job(job: &TrainJob) -> Result<TrainReport> {
    job.train_cfg.validate()?;
    let t_all = Instant::now();

    // Load config + optional initial weights from --init-from or --config.
    let (config_bytes, config, mut codec_weights, mut gen_weights, mut tokenizer_bytes) =
        if let Some(ref init_dir) = job.init_from {
            let loaded = load_local_artifact_weights(init_dir)?;
            (
                loaded.config_bytes,
                loaded.config,
                loaded.codec_weights,
                loaded.generator_weights,
                loaded.tokenizer_bytes,
            )
        } else {
            let cfg_path = job.config_path.as_ref().ok_or_else(|| {
                TrainError::Config("pass either --config FILE or --init-from ARTIFACT".into())
            })?;
            let bytes = fs::read(cfg_path)
                .map_err(|e| TrainError::Io(format!("read config {}: {e}", cfg_path.display())))?;
            let cfg =
                LocalNcmConfig::from_json(&bytes).map_err(|e| TrainError::Config(e.to_string()))?;
            let cw = init_codec_weights(cfg.codec(), job.train_cfg.seed);
            let gw = init_generator_weights(cfg.codec(), cfg.model(), job.train_cfg.seed);
            (bytes, cfg, cw, gw, None)
        };

    if let Some(ref tok_path) = job.tokenizer_path {
        tokenizer_bytes =
            Some(fs::read(tok_path).map_err(|e| {
                TrainError::Io(format!("read tokenizer {}: {e}", tok_path.display()))
            })?);
    }

    let local_tok = match &tokenizer_bytes {
        Some(bytes) => Some(
            LocalTokenizer::from_json_bytes(bytes)
                .map_err(|e| TrainError::Data(format!("invalid tokenizer.json: {e}")))?,
        ),
        None => None,
    };

    let dataset = TokenDataset::load(&job.data_path, &config, local_tok.as_ref())?;
    let resume_ckpt = match &job.resume_dir {
        Some(dir) => Some(load_checkpoint(dir, &config)?),
        None => None,
    };

    let mut all_history = Vec::new();
    let mut codec_initial = None;
    let mut codec_final = None;
    let mut gen_initial = None;
    let mut gen_final = None;

    if matches!(job.train_cfg.stage, TrainStage::Codec | TrainStage::All) {
        let mut stage_cfg = job.train_cfg.clone();
        if let Some(cs) = job.train_cfg.codec_steps {
            stage_cfg.steps = cs;
        }
        let stage_resume = resume_ckpt
            .as_ref()
            .filter(|c| c.meta.stage == TrainStage::Codec);
        let (trained_codec, hist) = run_codec_stage(
            &config,
            &dataset,
            &codec_weights,
            stage_resume,
            &stage_cfg,
            &job.device,
            job.checkpoint_dir.as_deref(),
        )?;
        codec_weights = trained_codec;
        codec_initial = hist.first().map(|m| m.loss);
        codec_final = hist.last().map(|m| m.loss);
        all_history.extend(hist);
    }

    if matches!(job.train_cfg.stage, TrainStage::Generator | TrainStage::All) {
        let stage_resume = resume_ckpt
            .as_ref()
            .filter(|c| c.meta.stage == TrainStage::Generator);
        let (trained_gen, hist) = run_generator_stage(
            &config,
            &dataset,
            &codec_weights,
            &gen_weights,
            stage_resume,
            &job.train_cfg,
            &job.device,
            job.checkpoint_dir.as_deref(),
        )?;
        gen_weights = trained_gen;
        gen_initial = hist.first().map(|m| m.loss);
        gen_final = hist.last().map(|m| m.loss);
        all_history.extend(hist);
    }

    let total_params = match job.train_cfg.stage {
        TrainStage::Codec => lora::count_params(&codec_weights),
        TrainStage::Generator => lora::count_params(&gen_weights),
        TrainStage::All => lora::count_params(&codec_weights) + lora::count_params(&gen_weights),
    };
    let trainable_params = if job.train_cfg.lora_rank > 0 {
        let adapters = lora::init_lora_adapters(&gen_weights, job.train_cfg.lora_rank, 0)?;
        lora::count_params(&adapters)
    } else {
        total_params
    };

    let provenance = serde_json::json!({
        "trainer": "ncm-train",
        "profile": "local-ncm-v1",
        "device": job.device,
        "stage": job.train_cfg.stage.as_str(),
        "steps_completed": all_history.len(),
        "seed": job.train_cfg.seed,
        "lora_rank": job.train_cfg.lora_rank,
        "lora_alpha": job.train_cfg.effective_lora_alpha(),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "codec_final_loss": codec_final,
        "generator_final_loss": gen_final,
    });

    export_local_ncm_artifact(
        &job.output_dir,
        &config_bytes,
        &config,
        &codec_weights,
        &gen_weights,
        tokenizer_bytes.as_deref(),
        &provenance,
        job.overwrite,
    )?;

    let total_seconds = t_all.elapsed().as_secs_f64();
    let mean_step_ms = if all_history.is_empty() {
        0.0
    } else {
        all_history.iter().map(|m| m.step_ms).sum::<f64>() / all_history.len() as f64
    };

    Ok(TrainReport {
        schema: "ncm-train-report-v1".into(),
        device: job.device.clone(),
        stage: job.train_cfg.stage,
        output_artifact: job.output_dir.display().to_string(),
        dataset_samples: dataset.len(),
        total_seconds,
        codec_initial_loss: codec_initial,
        codec_final_loss: codec_final,
        generator_initial_loss: gen_initial,
        generator_final_loss: gen_final,
        steps_completed: all_history.len(),
        mean_step_ms,
        lora_rank: job.train_cfg.lora_rank,
        trainable_params,
        total_params,
        history: all_history,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_codec_stage(
    config: &LocalNcmConfig,
    dataset: &TokenDataset,
    init_weights: &schedule::TensorMap,
    resume: Option<&artifact::LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    device: &str,
    checkpoint_dir: Option<&Path>,
) -> Result<(schedule::TensorMap, Vec<StepMetrics>)> {
    let _ = (
        config,
        dataset,
        init_weights,
        resume,
        train_cfg,
        checkpoint_dir,
    );
    match device {
        #[cfg(feature = "cpu")]
        "cpu" | "cuda" | "cuda-cudnn" => candle_backend::train_codec(
            config,
            dataset,
            init_weights,
            resume,
            train_cfg,
            device,
            checkpoint_dir,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => mlx_backend::train_codec(
            config,
            dataset,
            init_weights,
            resume,
            train_cfg,
            checkpoint_dir,
        ),
        other => Err(TrainError::Backend(format!(
            "training device `{other}` is not enabled in this build"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_generator_stage(
    config: &LocalNcmConfig,
    dataset: &TokenDataset,
    frozen_codec_weights: &schedule::TensorMap,
    init_gen_weights: &schedule::TensorMap,
    resume: Option<&artifact::LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    device: &str,
    checkpoint_dir: Option<&Path>,
) -> Result<(schedule::TensorMap, Vec<StepMetrics>)> {
    let _ = (
        config,
        dataset,
        frozen_codec_weights,
        init_gen_weights,
        resume,
        train_cfg,
        checkpoint_dir,
    );
    match device {
        #[cfg(feature = "cpu")]
        "cpu" | "cuda" | "cuda-cudnn" => candle_backend::train_generator(
            config,
            dataset,
            frozen_codec_weights,
            init_gen_weights,
            resume,
            train_cfg,
            device,
            checkpoint_dir,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => mlx_backend::train_generator(
            config,
            dataset,
            frozen_codec_weights,
            init_gen_weights,
            resume,
            train_cfg,
            checkpoint_dir,
        ),
        other => Err(TrainError::Backend(format!(
            "training device `{other}` is not enabled in this build"
        ))),
    }
}
