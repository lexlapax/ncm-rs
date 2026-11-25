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
use ncm_train::{TrainConfig, TrainJob, TrainPrecision, TrainStage};
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
    let (w_full, h_full) = ncm_train::candle_backend::train_codec(
        &cfg, &ds, None, &codec_w, None, &full_cfg, "cpu", None,
    )
    .unwrap();

    // Verify resume with constant LR so 5+5 sees an identical LR schedule.
    let const_full = TrainConfig {
        warmup_steps: 0,
        min_lr: 5e-3,
        ..full_cfg.clone()
    };
    let (w_const_full, h_const_full) = ncm_train::candle_backend::train_codec(
        &cfg,
        &ds,
        None,
        &codec_w,
        None,
        &const_full,
        "cpu",
        None,
    )
    .unwrap();
    let mut const_half = const_full.clone();
    const_half.steps = 5;
    let _ = ncm_train::candle_backend::train_codec(
        &cfg,
        &ds,
        None,
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
        None,
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
        &cfg, &ds, None, &w_full, &gen_w, None, &gen_full, "cpu", None,
    )
    .unwrap();
    let mut gen_half = gen_full.clone();
    gen_half.steps = 5;
    let _ = ncm_train::candle_backend::train_generator(
        &cfg,
        &ds,
        None,
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
        None,
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
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(out_dir.clone()),
        save_adapter: None,
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
            ..TrainConfig::default()
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
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(ft_dir.clone()),
        save_adapter: None,
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
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(out_dir.clone()),
        save_adapter: None,
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
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(out_dir.clone()),
        save_adapter: None,
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

#[cfg(feature = "cpu")]
#[test]
fn lora_finite_diff_merged_equivalence_and_checkpoint_resume() {
    use ncm_train::lora::{init_lora_adapters, lora_a_name, lora_b_name, merge_lora_weights};

    let cfg = LocalNcmConfig::from_json(tiny_config_json().as_bytes()).unwrap();
    let ds = TokenDataset::from_samples(&cfg, tiny_samples()).unwrap();
    let codec_w = init_codec_weights(cfg.codec(), 11);
    let gen_w = init_generator_weights(cfg.codec(), cfg.model(), 19);
    let prefixes = [1u32, 2, 3, 4, 9, 10, 11, 12];
    let targets = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1];

    // Give lora_b non-zero values so BOTH lora_a and lora_b have non-trivial gradients.
    let rank = 4;
    let alpha = 8.0;
    let mut adapters = init_lora_adapters(&gen_w, rank, 77).unwrap();
    for (name, (_, vals)) in adapters.iter_mut() {
        if name.ends_with(".lora_b") {
            for (i, v) in vals.iter_mut().enumerate() {
                *v = (((i * 19 + 7) % 23) as f32 - 11.0) * 0.02;
            }
        }
    }

    // 1. Central finite-difference check on LoRA A and B (< 2e-4 abs diff).
    for (base, idx_a, idx_b) in [
        ("transformer.blocks.0.attn.c_attn.weight", 5, 9),
        ("transformer.blocks.0.attn.c_proj.weight", 3, 7),
        ("transformer.blocks.0.mlp.c_fc.weight", 11, 13),
        ("transformer.blocks.0.mlp.c_proj.weight", 4, 6),
    ] {
        for (name, idx) in [(lora_a_name(base), idx_a), (lora_b_name(base), idx_b)] {
            let (analytic, numeric, diff) =
                ncm_train::candle_backend::finite_diff_check_generator_lora(
                    &cfg, &codec_w, &gen_w, &adapters, rank, alpha, &prefixes, &targets, 2, &name,
                    idx, 1e-3,
                )
                .unwrap();
            assert!(
                diff < 2e-4,
                "LoRA grad mismatch on {name}[{idx}]: analytic={analytic:.6} numeric={numeric:.6} diff={diff:.6}"
            );
        }
    }

    // 2. Exact LoRA-vs-merged forward loss equivalence (< 1e-5).
    let loss_lora = ncm_train::candle_backend::eval_generator_loss(
        &cfg,
        &codec_w,
        &gen_w,
        Some((&adapters, rank, alpha)),
        &prefixes,
        &targets,
        2,
        "cpu",
    )
    .unwrap();
    let merged = merge_lora_weights(&gen_w, &adapters, rank, alpha).unwrap();
    let loss_merged = ncm_train::candle_backend::eval_generator_loss(
        &cfg, &codec_w, &merged, None, &prefixes, &targets, 2, "cpu",
    )
    .unwrap();
    let eq_diff = (loss_lora - loss_merged).abs();
    assert!(
        eq_diff < 1e-5,
        "LoRA vs merged loss mismatch: lora={loss_lora:.8} merged={loss_merged:.8} diff={eq_diff:.8}"
    );

    // 3. Deterministic LoRA checkpoint save/resume (10 uninterrupted vs 5 + resume 5).
    let ckpt_dir = TempDir::new("ckpt-lora");
    let full_cfg = TrainConfig {
        stage: TrainStage::Generator,
        steps: 10,
        batch_size: 2,
        learning_rate: 5e-3,
        min_lr: 5e-3,
        warmup_steps: 0,
        weight_decay: 0.01,
        seed: 42,
        checkpoint_every: 5,
        lora_rank: rank,
        lora_alpha: alpha,
        ..TrainConfig::default()
    };
    let (w_full, h_full) = ncm_train::candle_backend::train_generator(
        &cfg, &ds, None, &codec_w, &gen_w, None, &full_cfg, "cpu", None,
    )
    .unwrap();

    let mut half_cfg = full_cfg.clone();
    half_cfg.steps = 5;
    let _ = ncm_train::candle_backend::train_generator(
        &cfg,
        &ds,
        None,
        &codec_w,
        &gen_w,
        None,
        &half_cfg,
        "cpu",
        Some(&ckpt_dir.0),
    )
    .unwrap();
    let loaded = ncm_train::artifact::load_checkpoint(&ckpt_dir.0, &cfg).unwrap();
    assert_eq!(loaded.meta.step, 5);
    assert_eq!(loaded.meta.lora_rank, rank);

    // Mismatched LoRA rank on resume must fail explicitly.
    let mut bad_cfg = full_cfg.clone();
    bad_cfg.lora_rank = 2;
    assert!(
        ncm_train::candle_backend::train_generator(
            &cfg,
            &ds,
            None,
            &codec_w,
            &gen_w,
            Some(&loaded),
            &bad_cfg,
            "cpu",
            None,
        )
        .is_err()
    );

    let (w_resumed, h_resumed) = ncm_train::candle_backend::train_generator(
        &cfg,
        &ds,
        None,
        &codec_w,
        &gen_w,
        Some(&loaded),
        &full_cfg,
        "cpu",
        None,
    )
    .unwrap();
    assert_eq!(h_resumed.len(), 5);
    let loss_diff = (h_full.last().unwrap().loss - h_resumed.last().unwrap().loss).abs();
    assert!(loss_diff < 1e-5, "LoRA resume loss diff={loss_diff}");
    for (k, (_, v1)) in &w_full {
        let (_, v2) = &w_resumed[k];
        let max_d = v1
            .iter()
            .zip(v2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-5, "LoRA weight `{k}` resume max_diff={max_d}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lora_finetune_converges_and_merges_for_inference() {
    use ncm_backend_cuda::CandleBackend;
    use ncm_core::CodecBackend;
    use ncm_io::Artifact;
    use ncm_model::LocalNcmModel;

    let tmp = TempDir::new("lora-cuda");
    let cfg_path = tmp.0.join("config.json");
    let base_data = tmp.0.join("base-data.json");
    let ft_data = tmp.0.join("ft-data.json");
    let base_dir = tmp.0.join("base");
    let lora_dir = tmp.0.join("lora-merged");
    fs::write(&cfg_path, tiny_config_json()).unwrap();
    fs::write(
        &base_data,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [1, 2, 3, 4, 5, 6, 7, 8]},
          {"prefix": [9, 10, 11, 12], "target": [9, 10, 11, 12, 13, 14, 15, 1]}
        ]"#,
    )
    .unwrap();
    // Domain-shifted target (swap prefix->target mapping) so base model has high initial loss
    // while keeping trained embeddings/heads.
    fs::write(
        &ft_data,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [9, 10, 11, 12, 13, 14, 15, 1]},
          {"prefix": [9, 10, 11, 12], "target": [1, 2, 3, 4, 5, 6, 7, 8]}
        ]"#,
    )
    .unwrap();

    // Create a trained base artifact first.
    let base_job = TrainJob {
        config_path: Some(cfg_path),
        init_from: None,
        data_path: base_data,
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(base_dir.clone()),
        save_adapter: None,
        checkpoint_dir: None,
        resume_dir: None,
        device: "cuda".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::All,
            steps: 50,
            codec_steps: Some(40),
            batch_size: 2,
            learning_rate: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 5,
            weight_decay: 0.0,
            ..TrainConfig::default()
        },
    };
    ncm_train::run_training_job(&base_job).unwrap();

    // Fine-tune with LoRA rank 4 on CUDA.
    let lora_job = TrainJob {
        config_path: None,
        init_from: Some(base_dir),
        data_path: ft_data,
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(lora_dir.clone()),
        save_adapter: None,
        checkpoint_dir: None,
        resume_dir: None,
        device: "cuda".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::Generator,
            steps: 60,
            batch_size: 2,
            learning_rate: 5e-3,
            min_lr: 5e-4,
            warmup_steps: 5,
            weight_decay: 0.0,
            lora_rank: 4,
            lora_alpha: 8.0,
            ..TrainConfig::default()
        },
    };
    let report = ncm_train::run_training_job(&lora_job).unwrap();
    assert_eq!(report.lora_rank, 4);
    assert!(report.trainable_params < report.total_params);
    assert!(
        report.generator_final_loss.unwrap() < report.generator_initial_loss.unwrap() * 0.5,
        "LoRA loss did not decrease enough: {:?} -> {:?}",
        report.generator_initial_loss,
        report.generator_final_loss
    );

    // Merged local-ncm-v1 artifact must load and generate cleanly.
    let artifact = Artifact::load_inference_streaming(&lora_dir).unwrap();
    let backend = CandleBackend::cpu();
    let model = LocalNcmModel::from_artifact(backend.clone(), &artifact).unwrap();
    let generated = model.generate(&[1, 2, 3, 4], 1, None).unwrap();
    assert_eq!(backend.read_indices(&generated.ids).unwrap().len(), 8);
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mlx_lora_finetune_converges_and_resumes_deterministically() {
    use ncm_io::Artifact;

    let tmp = TempDir::new("lora-mlx");
    let cfg_path = tmp.0.join("config.json");
    let base_data = tmp.0.join("base-data.json");
    let ft_data = tmp.0.join("ft-data.json");
    let base_dir = tmp.0.join("base");
    let lora_dir = tmp.0.join("lora-merged");
    let ckpt_dir = tmp.0.join("ckpt-lora");
    fs::write(&cfg_path, tiny_config_json()).unwrap();
    fs::write(
        &base_data,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [1, 2, 3, 4, 5, 6, 7, 8]},
          {"prefix": [9, 10, 11, 12], "target": [9, 10, 11, 12, 13, 14, 15, 1]}
        ]"#,
    )
    .unwrap();
    fs::write(
        &ft_data,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [9, 10, 11, 12, 13, 14, 15, 1]},
          {"prefix": [9, 10, 11, 12], "target": [1, 2, 3, 4, 5, 6, 7, 8]}
        ]"#,
    )
    .unwrap();

    let base_job = TrainJob {
        config_path: Some(cfg_path),
        init_from: None,
        data_path: base_data,
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(base_dir.clone()),
        save_adapter: None,
        checkpoint_dir: None,
        resume_dir: None,
        device: "mlx".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::All,
            steps: 50,
            codec_steps: Some(40),
            batch_size: 2,
            learning_rate: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 5,
            weight_decay: 0.0,
            ..TrainConfig::default()
        },
    };
    ncm_train::run_training_job(&base_job).unwrap();

    let lora_job = TrainJob {
        config_path: None,
        init_from: Some(base_dir.clone()),
        data_path: ft_data,
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(lora_dir.clone()),
        save_adapter: None,
        checkpoint_dir: Some(ckpt_dir.clone()),
        resume_dir: None,
        device: "mlx".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::Generator,
            steps: 60,
            batch_size: 2,
            learning_rate: 5e-3,
            min_lr: 5e-4,
            warmup_steps: 5,
            weight_decay: 0.0,
            checkpoint_every: 30,
            lora_rank: 4,
            lora_alpha: 8.0,
            ..TrainConfig::default()
        },
    };
    let report = ncm_train::run_training_job(&lora_job).unwrap();
    assert_eq!(report.lora_rank, 4);
    assert!(report.trainable_params < report.total_params);
    assert!(
        report.generator_final_loss.unwrap() < report.generator_initial_loss.unwrap() * 0.5,
        "MLX LoRA loss did not decrease enough: {:?} -> {:?}",
        report.generator_initial_loss,
        report.generator_final_loss
    );
    let artifact = Artifact::load_inference_streaming(&lora_dir).unwrap();
    assert_eq!(artifact.manifest().profile, "local-ncm-v1");
}

#[cfg(feature = "cpu")]
#[test]
fn jsonl_dataset_loads_all_schemas_and_chunks_long_texts() {
    use ncm_tokenizers::LocalTokenizer;

    let cfg = LocalNcmConfig::from_json(tiny_config_json().as_bytes()).unwrap();
    let tok_json = r###"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "WhitespaceSplit"},
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordPiece",
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
        "vocab": {
          "[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7,
          "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15
        }
      }
    }"###;
    let tok = LocalTokenizer::from_json_bytes(tok_json.as_bytes()).unwrap();

    let tmp = TempDir::new("jsonl-loader");
    let jsonl_path = tmp.0.join("mixed.jsonl");
    // context_len=8, prefix_len=4:
    // - Line 1: {"text": "a b c d e f g h i j k l"} (12 tokens -> 2 overlapping 8-token chunks)
    // - Line 2: {"prompt": "a b c d", "completion": "e f g h"} (1 sample)
    // - Line 3: {"prefix": [9, 10, 11, 12], "target": [9, 10, 11, 12, 13, 14, 15, 1]} (1 sample)
    fs::write(
        &jsonl_path,
        concat!(
            "{\"text\": \"a b c d e f g h i j k l\"}\n",
            "\n",
            "{\"prompt\": \"a b c d\", \"completion\": \"e f g h\"}\n",
            "{\"prefix\": [9, 10, 11, 12], \"target\": [9, 10, 11, 12, 13, 14, 15, 1]}\n",
        ),
    )
    .unwrap();

    let ds = TokenDataset::load(&jsonl_path, &cfg, Some(&tok)).unwrap();
    assert_eq!(ds.len(), 4);
    assert_eq!(ds.samples()[0].prefix, vec![1, 2, 3, 4]);
    assert_eq!(ds.samples()[0].target, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(ds.samples()[1].prefix, vec![5, 6, 7, 8]);
    assert_eq!(ds.samples()[1].target, vec![5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(ds.samples()[2].prefix, vec![1, 2, 3, 4]);
    assert_eq!(ds.samples()[2].target, vec![5, 6, 7, 8, 0, 0, 0, 0]);
    assert_eq!(ds.samples()[3].prefix, vec![9, 10, 11, 12]);
    assert_eq!(ds.samples()[3].target, vec![9, 10, 11, 12, 13, 14, 15, 1]);

    // Sequential eval_batches partitions all 4 samples into two batches of 2.
    let eval_b = ds.eval_batches(2, 64);
    assert_eq!(eval_b.len(), 2);
    assert_eq!(eval_b[0].2, 2);
    assert_eq!(eval_b[1].2, 2);
}

#[cfg(feature = "cpu")]
#[test]
fn grad_accum_matches_full_batch_and_val_eval_records_loss() {
    let cfg = LocalNcmConfig::from_json(tiny_config_json().as_bytes()).unwrap();
    let samples = vec![
        TrainingSample {
            prefix: vec![1, 2, 3, 4],
            target: vec![1, 2, 3, 4, 5, 6, 7, 8],
        },
        TrainingSample {
            prefix: vec![9, 10, 11, 12],
            target: vec![9, 10, 11, 12, 13, 14, 15, 1],
        },
        TrainingSample {
            prefix: vec![2, 3, 4, 5],
            target: vec![2, 3, 4, 5, 6, 7, 8, 9],
        },
        TrainingSample {
            prefix: vec![8, 7, 6, 5],
            target: vec![8, 7, 6, 5, 4, 3, 2, 1],
        },
    ];
    let ds = TokenDataset::from_samples(&cfg, samples).unwrap();
    let codec_w = init_codec_weights(cfg.codec(), 11);
    let gen_w = init_generator_weights(cfg.codec(), cfg.model(), 19);

    // 1. Codec gradient accumulation: batch=4, accum=1 vs batch=2, accum=2.
    let c_full_cfg = TrainConfig {
        stage: TrainStage::Codec,
        steps: 6,
        batch_size: 4,
        grad_accum: 1,
        learning_rate: 5e-3,
        min_lr: 5e-3,
        warmup_steps: 0,
        seed: 42,
        ..TrainConfig::default()
    };
    let c_accum_cfg = TrainConfig {
        batch_size: 2,
        grad_accum: 2,
        ..c_full_cfg.clone()
    };
    let (cw_full, ch_full) = ncm_train::candle_backend::train_codec(
        &cfg,
        &ds,
        Some(&ds),
        &codec_w,
        None,
        &c_full_cfg,
        "cpu",
        None,
    )
    .unwrap();
    let (cw_acc, ch_acc) = ncm_train::candle_backend::train_codec(
        &cfg,
        &ds,
        Some(&ds),
        &codec_w,
        None,
        &c_accum_cfg,
        "cpu",
        None,
    )
    .unwrap();
    assert!(
        (ch_full.last().unwrap().loss - ch_acc.last().unwrap().loss).abs() < 1e-5,
        "codec grad_accum loss mismatch: {} vs {}",
        ch_full.last().unwrap().loss,
        ch_acc.last().unwrap().loss
    );
    for (k, (_, v1)) in &cw_full {
        let (_, v2) = &cw_acc[k];
        let max_d = v1
            .iter()
            .zip(v2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-5, "codec `{k}` grad_accum max_diff={max_d}");
    }

    // 2. Generator gradient accumulation + eval_every validation loss.
    let g_full_cfg = TrainConfig {
        stage: TrainStage::Generator,
        steps: 6,
        batch_size: 4,
        grad_accum: 1,
        learning_rate: 5e-3,
        min_lr: 5e-3,
        warmup_steps: 0,
        eval_every: 3,
        seed: 42,
        ..TrainConfig::default()
    };
    let g_accum_cfg = TrainConfig {
        batch_size: 2,
        grad_accum: 2,
        ..g_full_cfg.clone()
    };
    let (gw_full, gh_full) = ncm_train::candle_backend::train_generator(
        &cfg,
        &ds,
        Some(&ds),
        &cw_full,
        &gen_w,
        None,
        &g_full_cfg,
        "cpu",
        None,
    )
    .unwrap();
    let (gw_acc, gh_acc) = ncm_train::candle_backend::train_generator(
        &cfg,
        &ds,
        Some(&ds),
        &cw_full,
        &gen_w,
        None,
        &g_accum_cfg,
        "cpu",
        None,
    )
    .unwrap();
    assert!(
        (gh_full.last().unwrap().loss - gh_acc.last().unwrap().loss).abs() < 1e-5,
        "generator grad_accum loss mismatch: {} vs {}",
        gh_full.last().unwrap().loss,
        gh_acc.last().unwrap().loss
    );
    for (k, (_, v1)) in &gw_full {
        let (_, v2) = &gw_acc[k];
        let max_d = v1
            .iter()
            .zip(v2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-5, "generator `{k}` grad_accum max_diff={max_d}");
    }
    // eval_every=3 on 6 steps -> val_loss populated on steps 3 and 6 only.
    assert!(gh_acc[0].val_loss.is_none());
    assert!(gh_acc[1].val_loss.is_none());
    assert!(gh_acc[2].val_loss.is_some());
    assert!(gh_acc[5].val_loss.is_some());

    // 3. CPU BF16 must fail explicitly.
    let bad_bf16 = TrainConfig {
        train_precision: TrainPrecision::Bf16,
        ..g_full_cfg
    };
    assert!(
        ncm_train::candle_backend::train_generator(
            &cfg, &ds, None, &cw_full, &gen_w, None, &bad_bf16, "cpu", None,
        )
        .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_bf16_grad_accum_and_val_eval() {
    use ncm_backend_cuda::CandleBackend;
    use ncm_core::CodecBackend;
    use ncm_io::Artifact;
    use ncm_model::LocalNcmModel;

    let tmp = TempDir::new("bf16-cuda");
    let cfg_path = tmp.0.join("config.json");
    let base_data = tmp.0.join("base.json");
    let ft_data = tmp.0.join("ft.jsonl");
    let val_data = tmp.0.join("val.jsonl");
    let base_dir = tmp.0.join("base");
    let bf16_dir = tmp.0.join("bf16-lora");
    fs::write(&cfg_path, tiny_config_json()).unwrap();
    fs::write(
        &base_data,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [1, 2, 3, 4, 5, 6, 7, 8]},
          {"prefix": [9, 10, 11, 12], "target": [9, 10, 11, 12, 13, 14, 15, 1]}
        ]"#,
    )
    .unwrap();
    fs::write(
        &ft_data,
        concat!(
            "{\"prefix\": [1, 2, 3, 4], \"target\": [9, 10, 11, 12, 13, 14, 15, 1]}\n",
            "{\"prefix\": [9, 10, 11, 12], \"target\": [1, 2, 3, 4, 5, 6, 7, 8]}\n",
        ),
    )
    .unwrap();
    fs::write(
        &val_data,
        "{\"prefix\": [1, 2, 3, 4], \"target\": [9, 10, 11, 12, 13, 14, 15, 1]}\n",
    )
    .unwrap();

    let base_job = TrainJob {
        config_path: Some(cfg_path),
        init_from: None,
        data_path: base_data,
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(base_dir.clone()),
        save_adapter: None,
        checkpoint_dir: None,
        resume_dir: None,
        device: "cuda".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::All,
            steps: 50,
            codec_steps: Some(40),
            batch_size: 2,
            learning_rate: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 5,
            weight_decay: 0.0,
            ..TrainConfig::default()
        },
    };
    ncm_train::run_training_job(&base_job).unwrap();

    let bf16_job = TrainJob {
        config_path: None,
        init_from: Some(base_dir),
        data_path: ft_data,
        val_data_path: Some(val_data),
        tokenizer_path: None,
        output_dir: Some(bf16_dir.clone()),
        save_adapter: None,
        checkpoint_dir: None,
        resume_dir: None,
        device: "cuda".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::Generator,
            steps: 60,
            batch_size: 2,
            grad_accum: 2,
            train_precision: TrainPrecision::Bf16,
            eval_every: 20,
            learning_rate: 5e-3,
            min_lr: 5e-4,
            warmup_steps: 5,
            weight_decay: 0.0,
            lora_rank: 4,
            lora_alpha: 8.0,
            ..TrainConfig::default()
        },
    };
    let report = ncm_train::run_training_job(&bf16_job).unwrap();
    assert_eq!(report.train_precision, TrainPrecision::Bf16);
    assert_eq!(report.grad_accum, 2);
    assert!(report.final_val_loss.is_some());
    assert!(
        report.generator_final_loss.unwrap() < report.generator_initial_loss.unwrap() * 0.5,
        "CUDA BF16 LoRA loss did not converge: {:?} -> {:?}",
        report.generator_initial_loss,
        report.generator_final_loss
    );
    let artifact = Artifact::load_inference_streaming(&bf16_dir).unwrap();
    let backend = CandleBackend::cpu();
    let model = LocalNcmModel::from_artifact(backend.clone(), &artifact).unwrap();
    let generated = model.generate(&[1, 2, 3, 4], 1, None).unwrap();
    assert_eq!(backend.read_indices(&generated.ids).unwrap().len(), 8);
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mlx_bf16_grad_accum_and_val_eval() {
    use ncm_io::Artifact;

    let tmp = TempDir::new("bf16-mlx");
    let cfg_path = tmp.0.join("config.json");
    let base_data = tmp.0.join("base.json");
    let ft_data = tmp.0.join("ft.jsonl");
    let val_data = tmp.0.join("val.jsonl");
    let base_dir = tmp.0.join("base");
    let bf16_dir = tmp.0.join("bf16-lora");
    fs::write(&cfg_path, tiny_config_json()).unwrap();
    fs::write(
        &base_data,
        r#"[
          {"prefix": [1, 2, 3, 4], "target": [1, 2, 3, 4, 5, 6, 7, 8]},
          {"prefix": [9, 10, 11, 12], "target": [9, 10, 11, 12, 13, 14, 15, 1]}
        ]"#,
    )
    .unwrap();
    fs::write(
        &ft_data,
        concat!(
            "{\"prefix\": [1, 2, 3, 4], \"target\": [9, 10, 11, 12, 13, 14, 15, 1]}\n",
            "{\"prefix\": [9, 10, 11, 12], \"target\": [1, 2, 3, 4, 5, 6, 7, 8]}\n",
        ),
    )
    .unwrap();
    fs::write(
        &val_data,
        "{\"prefix\": [1, 2, 3, 4], \"target\": [9, 10, 11, 12, 13, 14, 15, 1]}\n",
    )
    .unwrap();

    let base_job = TrainJob {
        config_path: Some(cfg_path),
        init_from: None,
        data_path: base_data,
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(base_dir.clone()),
        save_adapter: None,
        checkpoint_dir: None,
        resume_dir: None,
        device: "mlx".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::All,
            steps: 50,
            codec_steps: Some(40),
            batch_size: 2,
            learning_rate: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 5,
            weight_decay: 0.0,
            ..TrainConfig::default()
        },
    };
    ncm_train::run_training_job(&base_job).unwrap();

    let bf16_job = TrainJob {
        config_path: None,
        init_from: Some(base_dir),
        data_path: ft_data,
        val_data_path: Some(val_data),
        tokenizer_path: None,
        output_dir: Some(bf16_dir.clone()),
        save_adapter: None,
        checkpoint_dir: None,
        resume_dir: None,
        device: "mlx".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::Generator,
            steps: 60,
            batch_size: 2,
            grad_accum: 2,
            train_precision: TrainPrecision::Bf16,
            eval_every: 20,
            learning_rate: 5e-3,
            min_lr: 5e-4,
            warmup_steps: 5,
            weight_decay: 0.0,
            lora_rank: 4,
            lora_alpha: 8.0,
            ..TrainConfig::default()
        },
    };
    let report = ncm_train::run_training_job(&bf16_job).unwrap();
    assert_eq!(report.train_precision, TrainPrecision::Bf16);
    assert_eq!(report.grad_accum, 2);
    assert!(report.final_val_loss.is_some());
    assert!(
        report.generator_final_loss.unwrap() < report.generator_initial_loss.unwrap() * 0.5,
        "MLX BF16 LoRA loss did not converge: {:?} -> {:?}",
        report.generator_initial_loss,
        report.generator_final_loss
    );
    let artifact = Artifact::load_inference_streaming(&bf16_dir).unwrap();
    assert_eq!(artifact.manifest().profile, "local-ncm-v1");
}

#[cfg(feature = "cpu")]
#[test]
fn lora_unmerged_adapter_matches_merged_inference_and_rejects_tamper() {
    use ncm_backend_cuda::CandleBackend;
    use ncm_core::CodecBackend;
    use ncm_io::{Artifact, LoraAdapterArtifact};
    use ncm_model::{LocalNcmModel, RuntimeLoraAdapter};

    let tmp = TempDir::new("lora-adapter-equiv");
    let cfg_path = tmp.0.join("config.json");
    let base_data = tmp.0.join("base.json");
    let ft_data = tmp.0.join("ft.json");
    let base_dir = tmp.0.join("base-model");
    let merged_dir = tmp.0.join("merged-model");
    let adapter_dir = tmp.0.join("lora-adapter");
    let adapter_only_dir = tmp.0.join("lora-adapter-only");

    fs::write(&cfg_path, tiny_config_json()).unwrap();
    fs::write(
        &base_data,
        serde_json::to_string(&vec![
            serde_json::json!({"prefix": [1, 2, 3, 4], "target": [5, 6, 7, 8, 9, 10, 11, 12]}),
            serde_json::json!({"prefix": [4, 3, 2, 1], "target": [12, 11, 10, 9, 8, 7, 6, 5]}),
        ])
        .unwrap(),
    )
    .unwrap();
    fs::write(
        &ft_data,
        serde_json::to_string(&vec![
            serde_json::json!({"prefix": [1, 2, 3, 4], "target": [15, 14, 13, 12, 11, 10, 9, 8]}),
            serde_json::json!({"prefix": [4, 3, 2, 1], "target": [8, 9, 10, 11, 12, 13, 14, 15]}),
        ])
        .unwrap(),
    )
    .unwrap();

    // 1. Train base model.
    let base_job = TrainJob {
        config_path: Some(cfg_path),
        init_from: None,
        data_path: base_data,
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(base_dir.clone()),
        save_adapter: None,
        checkpoint_dir: None,
        resume_dir: None,
        device: "cpu".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::All,
            steps: 25,
            codec_steps: Some(20),
            batch_size: 2,
            learning_rate: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 3,
            weight_decay: 0.0,
            ..TrainConfig::default()
        },
    };
    ncm_train::run_training_job(&base_job).unwrap();

    // 2. Fine-tune with LoRA exporting BOTH merged artifact and unmerged adapter.
    let ft_job = TrainJob {
        config_path: None,
        init_from: Some(base_dir.clone()),
        data_path: ft_data.clone(),
        val_data_path: None,
        tokenizer_path: None,
        output_dir: Some(merged_dir.clone()),
        save_adapter: Some(adapter_dir.clone()),
        checkpoint_dir: None,
        resume_dir: None,
        device: "cpu".into(),
        overwrite: true,
        train_cfg: TrainConfig {
            stage: TrainStage::Generator,
            steps: 30,
            batch_size: 2,
            learning_rate: 5e-3,
            min_lr: 5e-4,
            warmup_steps: 3,
            weight_decay: 0.0,
            lora_rank: 4,
            lora_alpha: 8.0,
            ..TrainConfig::default()
        },
    };
    let report = ncm_train::run_training_job(&ft_job).unwrap();
    assert!(report.output_adapter.is_some());

    // 3. Also verify adapter-only export (no --output) works cleanly.
    let adapter_only_job = TrainJob {
        output_dir: None,
        save_adapter: Some(adapter_only_dir.clone()),
        ..ft_job
    };
    let rep_only = ncm_train::run_training_job(&adapter_only_job).unwrap();
    assert!(rep_only.output_artifact.is_empty());
    assert!(rep_only.output_adapter.is_some());

    // 4. Compare unmerged adapter attached to base model vs merged model in FP32.
    let base_art = Artifact::load_inference_streaming(&base_dir).unwrap();
    let merged_art = Artifact::load_inference_streaming(&merged_dir).unwrap();
    let base_sha = base_art.manifest_sha256().to_owned();
    let merged_sha = merged_art.manifest_sha256().to_owned();
    assert_ne!(base_sha, merged_sha);

    // Adapter must reject mismatched base manifest SHA-256.
    assert!(LoraAdapterArtifact::load(&adapter_dir, Some(&merged_sha)).is_err());

    let adapter_art = LoraAdapterArtifact::load(&adapter_dir, Some(&base_sha)).unwrap();
    let adapter_only_art = LoraAdapterArtifact::load(&adapter_only_dir, Some(&base_sha)).unwrap();
    assert_eq!(adapter_art.tensors, adapter_only_art.tensors);

    let backend = CandleBackend::cpu();
    let mut base_model = LocalNcmModel::from_artifact(backend.clone(), &base_art).unwrap();
    let merged_model = LocalNcmModel::from_artifact(backend.clone(), &merged_art).unwrap();

    let prefix = [1u32, 2, 3, 4];
    let base_before = base_model.generate(&prefix, 1, None).unwrap();
    let base_ids_before = backend.read_indices(&base_before.ids).unwrap();

    let (n_layer, n_embd) = base_model.lora_dimensions();
    let runtime_adapter =
        RuntimeLoraAdapter::from_artifact(&backend, &adapter_art, n_layer, n_embd).unwrap();
    base_model.set_lora_adapter(Some(runtime_adapter)).unwrap();

    let adapted_out = base_model.generate(&prefix, 1, None).unwrap();
    let merged_out = merged_model.generate(&prefix, 1, None).unwrap();

    let adapted_codes = backend.read_indices(&adapted_out.codes).unwrap();
    let merged_codes = backend.read_indices(&merged_out.codes).unwrap();
    let adapted_ids = backend.read_indices(&adapted_out.ids).unwrap();
    let merged_ids = backend.read_indices(&merged_out.ids).unwrap();
    assert_eq!(
        adapted_codes, merged_codes,
        "unmerged adapter codes must match merged model exactly"
    );
    assert_eq!(
        adapted_ids, merged_ids,
        "unmerged adapter decoded IDs must match merged model exactly"
    );

    let adapted_logits = backend.read_f32(&adapted_out.logits).unwrap();
    let merged_logits = backend.read_f32(&merged_out.logits).unwrap();
    let max_diff = adapted_logits
        .iter()
        .zip(&merged_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "unmerged adapter vs merged logits max_diff={max_diff} >= 1e-4"
    );

    // 5. Detach adapter -> restores exact base model output (proves clean hot-swap!).
    base_model.set_lora_adapter(None).unwrap();
    let base_after = base_model.generate(&prefix, 1, None).unwrap();
    assert_eq!(
        backend.read_indices(&base_after.ids).unwrap(),
        base_ids_before
    );

    // 6. Tamper 1 byte in adapter.safetensors -> must fail closed.
    let st_path = adapter_dir.join("adapter.safetensors");
    let mut bytes = fs::read(&st_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    fs::write(&st_path, &bytes).unwrap();
    assert!(LoraAdapterArtifact::load(&adapter_dir, Some(&base_sha)).is_err());
}
