//! Verification suite for `ncm-train`: EMA updates, finite-difference gradient checks,
//! deterministic checkpoint resume, tiny-overfit convergence, fine-tuning, and verified
//! `local-ncm-v1` inference roundtrips across CPU, CUDA, and MLX.

use ncm_core::LocalNcmConfig;
#[cfg(feature = "cpu")]
use ncm_train::schedule::init_generator_weights;
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
use ncm_train::{TrainConfig, TrainJob, TrainStage};
use ncm_train::{
    data::{TokenDataset, TrainingSample},
    ema::EmaCodebookState,
    schedule::{init_codec_weights, lr_at_step},
};
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
struct TempDir(PathBuf);
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("ncm-train-test-{tag}-{}-{id}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
#[cfg(any(
    feature = "cpu",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn tiny_config_json() -> String {
    r#"{
      "tokenizer": {
        "checkpoint_path": "local-codec.pt",
        "in_vocab_size": 16,
        "context_length": 8,
        "embed_dim": 16,
        "vqvae_vocab_size": [8, 8, 8],
        "n_layers": 2,
        "compression_factor": 1,
        "dropout": 0.0,
        "pre_quant_dropout": 0.0,
        "pre_quant_groupnorm": 4,
        "pre_quant_norm_type": "group",
        "use_qk_norm": true,
        "dec_layers": 1,
        "enc_positional": true,
        "dec_positional": false,
        "vector_quantizer_config": {
          "clss": "multiscale_residual_vector_quantizer",
          "commitment_cost": 0.25,
          "decay": 0.99,
          "epsilon": 1e-5,
          "learned_all_sampling": false,
          "learned_l1_sampling": true,
          "learned_l1_cascade": true,
          "learned_l1_cascade_base": 2,
          "learned_l1_cascade_norm": false,
          "learned_l1_cascade_norm_final": false,
          "levels": {
            "use_manual_levels": true,
            "manual_levels": [2, 4, 8]
          },
          "quant_resi": {
            "enabled": true,
            "learnable_ratio": false,
            "ratio": 0.5,
            "share_mode": 0
          },
          "aux": {
            "fine_drop": {
              "min_keep": 1,
              "prob": 0.0
            }
          }
        }
      },
      "model": {
        "attn_pattern": "block_diagonal",
        "bias": true,
        "context_length": 18,
        "dropout": 0.0,
        "n_embd": 16,
        "n_head": 2,
        "n_layer": 1,
        "prefix_len": 4,
        "query_level0": true,
        "raw_token_prefix": true,
        "rope_base": 10000,
        "tie_heads_to_codebook": false,
        "use_level_encoding": true,
        "use_positional_encoding": true,
        "use_qk_norm": true,
        "use_rope": true,
        "vocab_size": [8, 8, 8],
        "temperature": 1.0,
        "head_kind": "untied_affine"
      }
    }"#
    .to_owned()
}

fn tiny_samples() -> Vec<TrainingSample> {
    vec![
        TrainingSample {
            prefix: vec![1, 2, 3, 4],
            target: vec![1, 2, 3, 4, 5, 6, 7, 8],
        },
        TrainingSample {
            prefix: vec![9, 10, 11, 12],
            target: vec![9, 10, 11, 12, 13, 14, 15, 1],
        },
    ]
}

#[test]
fn lr_schedule_warmup_and_cosine_bounds() {
    let l0 = lr_at_step(0, 100, 1e-3, 1e-5, 10);
    let l9 = lr_at_step(9, 100, 1e-3, 1e-5, 10);
    let l50 = lr_at_step(50, 100, 1e-3, 1e-5, 10);
    let l100 = lr_at_step(100, 100, 1e-3, 1e-5, 10);
    assert!((l0 - 1e-4).abs() < 1e-9);
    assert!((l9 - 1e-3).abs() < 1e-9);
    assert!(l50 < l9 && l50 > l100);
    assert!((l100 - 1e-5).abs() < 1e-9);
}

#[test]
fn ema_codebook_updates_and_restarts_dead_codes() {
    let cfg = LocalNcmConfig::from_json(tiny_config_json().as_bytes()).unwrap();
    let weights = init_codec_weights(cfg.codec(), 7);
    let mut ema = EmaCodebookState::from_codebooks(cfg.codec(), &weights);
    let d = cfg.codec().raw().embed_dim;
    let mut book = weights["vq.embeddings.0.weight"].1.clone();

    // Assign all 4 vectors to code 0 with constant value 3.0; codes 1..7 are unused.
    let flat_ze = vec![3.0f32; 4 * d];
    let assigned = vec![0u32; 4];
    let mut rng = 123u64;
    let mut restarted_total = 0;
    for _ in 0..30 {
        let stats = ema.update_level(0, &flat_ze, &assigned, &mut book, 0.8, 1e-5, 0.2, &mut rng);
        assert_eq!(stats.active_codes, 1);
        restarted_total += stats.restarted_codes;
    }
    // Code 0 must converge close to 3.0, and unused codes must trigger dead-code restarts.
    assert!((book[0] - 3.0).abs() < 0.05, "book[0]={}", book[0]);
    assert!(restarted_total > 0, "expected dead-code restarts");
}

#[test]
fn dataset_validation_and_deterministic_batches() {
    let cfg = LocalNcmConfig::from_json(tiny_config_json().as_bytes()).unwrap();
    let ds = TokenDataset::from_samples(&cfg, tiny_samples()).unwrap();
    assert_eq!(ds.len(), 2);
    let b1 = ds.sample_batch(4, 3, 42);
    let b2 = ds.sample_batch(4, 3, 42);
    assert_eq!(b1, b2);

    // Out-of-range token rejected.
    let bad = vec![TrainingSample {
        prefix: vec![1, 2, 3, 99],
        target: vec![1, 2, 3, 4, 5, 6, 7, 8],
    }];
    assert!(TokenDataset::from_samples(&cfg, bad).is_err());
}

#[cfg(feature = "cpu")]
#[test]
fn finite_difference_matches_analytic_generator_gradients() {
    let cfg = LocalNcmConfig::from_json(tiny_config_json().as_bytes()).unwrap();
    let codec_w = init_codec_weights(cfg.codec(), 11);
    let gen_w = init_generator_weights(cfg.codec(), cfg.model(), 19);
    let prefixes = [1u32, 2, 3, 4, 9, 10, 11, 12];
    let targets = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1];

    let mut failures = Vec::new();
    for (param, idx) in [
        ("transformer.wte.weight", 3),
        ("l0_queries", 5),
        ("transformer.blocks.0.attn.c_attn.weight", 17),
        ("transformer.blocks.0.attn.q_norm.weight", 1),
        ("transformer.blocks.0.mlp.c_fc.weight", 7),
        ("lm_heads.0.weight", 2),
        ("lm_heads.2.bias", 1),
    ] {
        let (analytic, numeric, diff) = ncm_train::candle_backend::finite_diff_check_generator(
            &cfg, &codec_w, &gen_w, &prefixes, &targets, 2, param, idx, 1e-3,
        )
        .unwrap();
        let rel = diff / numeric.abs().max(analytic.abs()).max(1e-3);
        eprintln!(
            "{param}[{idx}]: analytic={analytic:.6} numeric={numeric:.6} diff={diff:.6} rel={rel:.4}"
        );
        if !(diff < 2e-3 || rel < 2e-2) {
            failures.push(format!(
                "{param}[{idx}]: analytic={analytic:.6} numeric={numeric:.6}"
            ));
        }
    }
    assert!(failures.is_empty(), "grad mismatches: {failures:?}");
}

#[cfg(feature = "cpu")]
#[test]
fn checkpoint_save_and_resume_is_deterministic() {
    let cfg = LocalNcmConfig::from_json(tiny_config_json().as_bytes()).unwrap();
    let ds = TokenDataset::from_samples(&cfg, tiny_samples()).unwrap();
    let codec_w = init_codec_weights(cfg.codec(), 42);
    let gen_w = init_generator_weights(cfg.codec(), cfg.model(), 42);

    let full_cfg = TrainConfig {
        stage: TrainStage::Codec,
        steps: 10,
        batch_size: 2,
        learning_rate: 5e-3,
        min_lr: 1e-4,
        warmup_steps: 2,
        seed: 42,
        checkpoint_every: 5,
        ..TrainConfig::default()
    };

    // Codec: 10 uninterrupted vs 5 + resume 5
    let ckpt_dir = TempDir::new("ckpt-codec");
    let (w_full, h_full) =
        ncm_train::candle_backend::train_codec(&cfg, &ds, &codec_w, None, &full_cfg, "cpu", None)
            .unwrap();

    // Verify resume with constant LR so 5+5 sees an identical LR schedule.
    let const_full = TrainConfig {
        warmup_steps: 0,
        min_lr: 5e-3,
        ..full_cfg.clone()
    };
    let (w_const_full, h_const_full) =
        ncm_train::candle_backend::train_codec(&cfg, &ds, &codec_w, None, &const_full, "cpu", None)
            .unwrap();
    let mut const_half = const_full.clone();
    const_half.steps = 5;
    let _ = ncm_train::candle_backend::train_codec(
        &cfg,
        &ds,
        &codec_w,
        None,
        &const_half,
        "cpu",
        Some(&ckpt_dir.0),
    )
    .unwrap();
    let loaded = ncm_train::artifact::load_checkpoint(&ckpt_dir.0, &cfg).unwrap();
    assert_eq!(loaded.meta.step, 5);
    let (w_resumed, h_resumed) = ncm_train::candle_backend::train_codec(
        &cfg,
        &ds,
        &codec_w,
        Some(&loaded),
        &const_full,
        "cpu",
        None,
    )
    .unwrap();
    assert_eq!(h_resumed.len(), 5);
    let loss_diff = (h_const_full.last().unwrap().loss - h_resumed.last().unwrap().loss).abs();
    assert!(loss_diff < 1e-5, "codec resume loss diff={loss_diff}");
    for (k, (_, v1)) in &w_const_full {
        let (_, v2) = &w_resumed[k];
        let max_d = v1
            .iter()
            .zip(v2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-5, "codec weight `{k}` resume max_diff={max_d}");
    }

    // Generator: 10 uninterrupted vs 5 + resume 5
    let ckpt_gen = TempDir::new("ckpt-gen");
    let gen_full = TrainConfig {
        stage: TrainStage::Generator,
        steps: 10,
        batch_size: 2,
        learning_rate: 5e-3,
        min_lr: 5e-3,
        warmup_steps: 0,
        seed: 42,
        checkpoint_every: 5,
        ..TrainConfig::default()
    };
    let (gw_full, gh_full) = ncm_train::candle_backend::train_generator(
        &cfg, &ds, &w_full, &gen_w, None, &gen_full, "cpu", None,
    )
    .unwrap();
    let mut gen_half = gen_full.clone();
    gen_half.steps = 5;
    let _ = ncm_train::candle_backend::train_generator(
        &cfg,
        &ds,
        &w_full,
        &gen_w,
        None,
        &gen_half,
        "cpu",
        Some(&ckpt_gen.0),
    )
    .unwrap();
    let loaded_gen = ncm_train::artifact::load_checkpoint(&ckpt_gen.0, &cfg).unwrap();
    assert_eq!(loaded_gen.meta.step, 5);
    let (gw_resumed, gh_resumed) = ncm_train::candle_backend::train_generator(
        &cfg,
        &ds,
        &w_full,
        &gen_w,
        Some(&loaded_gen),
        &gen_full,
        "cpu",
        None,
    )
    .unwrap();
    assert_eq!(gh_resumed.len(), 5);
    assert!((gh_full.last().unwrap().loss - gh_resumed.last().unwrap().loss).abs() < 1e-5);
    for (k, (_, v1)) in &gw_full {
        let (_, v2) = &gw_resumed[k];
        let max_d = v1
            .iter()
            .zip(v2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_d < 1e-5,
            "generator weight `{k}` resume max_diff={max_d}"
        );
    }
    let _ = h_full;
}

#[cfg(feature = "cpu")]
#[test]
fn codec_and_generator_overfit_and_roundtrip_to_local_ncm_inference() {
    use ncm_backend_cuda::CandleBackend;
    use ncm_core::CodecBackend;
    use ncm_io::Artifact;
    use ncm_model::LocalNcmModel;

    let tmp = TempDir::new("overfit-cpu");
    let cfg_path = tmp.0.join("config.json");
    let data_path = tmp.0.join("data.json");
    let out_dir = tmp.0.join("trained-model");
    let ft_dir = tmp.0.join("finetuned-model");

    fs::write(&cfg_path, tiny_config_json()).unwrap();
    fs::write(
        &data_path,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [1, 2, 3, 4, 5, 6, 7, 8]},
          {"prefix": [9, 10, 11, 12], "target": [9, 10, 11, 12, 13, 14, 15, 1]}
        ]"#,
    )
    .unwrap();

    let job = TrainJob {
        config_path: Some(cfg_path),
        init_from: None,
        data_path: data_path.clone(),
        tokenizer_path: None,
        output_dir: out_dir.clone(),
        checkpoint_dir: None,
        resume_dir: None,
        device: "cpu".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::All,
            steps: 60,
            codec_steps: Some(60),
            batch_size: 2,
            learning_rate: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 5,
            weight_decay: 0.0,
            grad_clip_norm: 1.0,
            commitment_weight: 0.25,
            ema_decay: 0.9,
            ema_eps: 1e-5,
            dead_code_threshold: 0.1,
            seed: 42,
            checkpoint_every: 0,
        },
    };
    let report = ncm_train::run_training_job(&job).unwrap();
    let c0 = report.codec_initial_loss.unwrap();
    let c1 = report.codec_final_loss.unwrap();
    let g0 = report.generator_initial_loss.unwrap();
    let g1 = report.generator_final_loss.unwrap();
    assert!(c1 < c0 * 0.5, "codec did not converge: {c0:.4} -> {c1:.4}");
    assert!(
        g1 < g0 * 0.3,
        "generator did not converge: {g0:.4} -> {g1:.4}"
    );

    // Verify exported local-ncm-v1 loads through strict streaming verification + LocalNcmModel.
    let artifact = Artifact::load_inference_streaming(&out_dir).unwrap();
    assert_eq!(artifact.manifest().profile, "local-ncm-v1");
    let backend = CandleBackend::cpu();
    let model = LocalNcmModel::from_artifact(backend.clone(), &artifact).unwrap();
    let generated = model.generate(&[1, 2, 3, 4], 1, None).unwrap();
    let ids = backend.read_indices(&generated.ids).unwrap();
    assert_eq!(ids.len(), 8);

    // Fine-tune generator from the exported artifact (--init-from) and verify re-export.
    let ft_job = TrainJob {
        config_path: None,
        init_from: Some(out_dir),
        data_path,
        tokenizer_path: None,
        output_dir: ft_dir.clone(),
        checkpoint_dir: None,
        resume_dir: None,
        device: "cpu".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::Generator,
            steps: 20,
            batch_size: 2,
            learning_rate: 3e-3,
            min_lr: 5e-4,
            warmup_steps: 2,
            weight_decay: 0.0,
            ..TrainConfig::default()
        },
    };
    let ft_report = ncm_train::run_training_job(&ft_job).unwrap();
    assert!(
        ft_report.generator_final_loss.unwrap() <= ft_report.generator_initial_loss.unwrap() + 0.05
    );
    let ft_art = Artifact::load_inference_streaming(&ft_dir).unwrap();
    let _ft_model = LocalNcmModel::from_artifact(CandleBackend::cpu(), &ft_art).unwrap();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_full_train_and_inference_roundtrip() {
    use ncm_backend_cuda::CandleBackend;
    use ncm_core::CodecBackend;
    use ncm_io::Artifact;
    use ncm_model::LocalNcmModel;

    let tmp = TempDir::new("overfit-cuda");
    let cfg_path = tmp.0.join("config.json");
    let data_path = tmp.0.join("data.json");
    let out_dir = tmp.0.join("cuda-model");
    fs::write(&cfg_path, tiny_config_json()).unwrap();
    fs::write(
        &data_path,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [1, 2, 3, 4, 5, 6, 7, 8]},
          {"prefix": [9, 10, 11, 12], "target": [9, 10, 11, 12, 13, 14, 15, 1]}
        ]"#,
    )
    .unwrap();

    let job = TrainJob {
        config_path: Some(cfg_path),
        init_from: None,
        data_path,
        tokenizer_path: None,
        output_dir: out_dir.clone(),
        checkpoint_dir: None,
        resume_dir: None,
        device: "cuda".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::All,
            steps: 60,
            codec_steps: Some(60),
            batch_size: 2,
            learning_rate: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 5,
            weight_decay: 0.0,
            ema_decay: 0.9,
            dead_code_threshold: 0.1,
            seed: 42,
            ..TrainConfig::default()
        },
    };
    let report = ncm_train::run_training_job(&job).unwrap();
    assert!(report.codec_final_loss.unwrap() < report.codec_initial_loss.unwrap() * 0.5);
    assert!(report.generator_final_loss.unwrap() < report.generator_initial_loss.unwrap() * 0.3);

    let artifact = Artifact::load_inference_streaming(&out_dir).unwrap();
    let backend = CandleBackend::cpu();
    let model = LocalNcmModel::from_artifact(backend.clone(), &artifact).unwrap();
    let generated = model.generate(&[1, 2, 3, 4], 1, None).unwrap();
    let ids = backend.read_indices(&generated.ids).unwrap();
    assert_eq!(ids.len(), 8);
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mlx_full_train_and_inference_roundtrip() {
    use ncm_io::Artifact;

    let tmp = TempDir::new("overfit-mlx");
    let cfg_path = tmp.0.join("config.json");
    let data_path = tmp.0.join("data.json");
    let out_dir = tmp.0.join("mlx-model");
    fs::write(&cfg_path, tiny_config_json()).unwrap();
    fs::write(
        &data_path,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [1, 2, 3, 4, 5, 6, 7, 8]},
          {"prefix": [9, 10, 11, 12], "target": [9, 10, 11, 12, 13, 14, 15, 1]}
        ]"#,
    )
    .unwrap();

    let job = TrainJob {
        config_path: Some(cfg_path),
        init_from: None,
        data_path,
        tokenizer_path: None,
        output_dir: out_dir.clone(),
        checkpoint_dir: None,
        resume_dir: None,
        device: "mlx".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::All,
            steps: 60,
            codec_steps: Some(60),
            batch_size: 2,
            learning_rate: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 5,
            weight_decay: 0.0,
            ema_decay: 0.9,
            dead_code_threshold: 0.1,
            seed: 42,
            ..TrainConfig::default()
        },
    };
    let report = ncm_train::run_training_job(&job).unwrap();
    assert!(report.codec_final_loss.unwrap() < report.codec_initial_loss.unwrap() * 0.5);
    assert!(report.generator_final_loss.unwrap() < report.generator_initial_loss.unwrap() * 0.3);

    let artifact = Artifact::load_inference_streaming(&out_dir).unwrap();
    assert_eq!(artifact.manifest().profile, "local-ncm-v1");
}
