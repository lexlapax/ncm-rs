//! Native Rust training & fine-tuning CLI (`ncm train`).
use crate::Result;
use ncm_train::{TrainConfig, TrainJob, TrainStage, run_training_job};
use std::path::PathBuf;

const TRAIN_USAGE: &str = "\
usage: ncm train (--config FILE | --init-from ARTIFACT) --data FILE (--output DIR | --save-adapter DIR) --device DEV [OPTIONS]";

pub(super) const TRAIN_HELP: &str = "\
Train or fine-tune a local-ncm-v1 codec (RVQ + EMA) and/or generator natively in Rust.

Usage:
  ncm train (--config FILE | --init-from ARTIFACT) --data FILE (--output DIR | --save-adapter DIR) --device DEV [OPTIONS]

Required:
  --config FILE          Local NCM config.json (for training from scratch)
  --init-from ARTIFACT   Existing verified local-ncm-v1, tinystories, xsum, wmt14, or lm1b artifact
  --data FILE            Training dataset (.jsonl, .json, or .txt)
  --output DIR           Output directory for the exported verified merged local-ncm-v1 artifact
  --save-adapter DIR     Output directory for a verified unmerged ncm-lora-adapter-v1 adapter
  --device DEV           Explicit training device: cuda, cuda-cudnn, mlx, or cpu (no fallback)

Options:
  --stage STAGE          codec | generator | all (default: all, or generator when --init-from is set)
  --tokenizer FILE       Optional tokenizer.json to bundle & encode text datasets
  --val-data FILE        Optional validation dataset (.jsonl, .json, or .txt) for no-grad eval
  --eval-every N         Evaluate validation loss every N steps (default: 0 = final step only)
  --lora-rank R          Enable LoRA generator fine-tuning with rank R (default: 0 = full fine-tuning)
  --lora-alpha A         LoRA scaling alpha (default: 2 * R when --lora-rank > 0)
  --train-precision P    Compute precision: fp32 (default) | bf16 (GPU only; FP32 master weights)
  --grad-accum N         Micro-batch gradient accumulation steps (default: 1)
  --steps N              Optimization steps (default: 100)
  --codec-steps N        Separate step count for codec stage when --stage all
  --batch N              Minibatch size 1..=256 (default: 8)
  --lr FLOAT             Base AdamW learning rate (default: 1e-3)
  --min-lr FLOAT         Cosine floor learning rate (default: 1e-5)
  --warmup N             Linear LR warmup steps (default: 10)
  --weight-decay FLOAT   Decoupled AdamW weight decay (default: 0.01)
  --grad-clip FLOAT      Global L2 gradient clipping norm, 0 disables (default: 1.0)
  --commitment FLOAT     VQ commitment loss weight (default: 0.25)
  --ema-decay FLOAT      EMA codebook decay (default: 0.99)
  --dead-code FLOAT      Dead-code restart cluster threshold (default: 0.05)
  --seed N               Deterministic RNG seed (default: 42)
  --checkpoint-dir DIR   Directory for periodic training checkpoints
  --checkpoint-every N   Save checkpoint every N steps (default: 0 = disabled)
  --resume DIR           Resume training from a saved checkpoint directory
  --report FILE          Optional JSON file path to write the full training report
  --overwrite            Replace --output if it already exists
";

pub(super) fn parse_train_args(args: &[String]) -> Result<(TrainJob, Option<PathBuf>)> {
    let mut config_path: Option<PathBuf> = None;
    let mut init_from: Option<PathBuf> = None;
    let mut data_path: Option<PathBuf> = None;
    let mut val_data_path: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut save_adapter: Option<PathBuf> = None;
    let mut checkpoint_dir: Option<PathBuf> = None;
    let mut resume_dir: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut device: Option<String> = None;
    let mut overwrite = false;
    let mut explicit_stage: Option<TrainStage> = None;
    let mut cfg = TrainConfig::default();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                config_path = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--init-from" => {
                init_from = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--data" => {
                data_path = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--val-data" => {
                val_data_path = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--eval-every" => {
                cfg.eval_every = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--grad-accum" => {
                cfg.grad_accum = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--train-precision" => {
                cfg.train_precision = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--tokenizer" => {
                tokenizer_path = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--output" => {
                output_dir = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--save-adapter" => {
                save_adapter = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--device" => {
                device = Some(args.get(i + 1).ok_or(TRAIN_USAGE)?.clone());
                i += 2;
            }
            "--stage" => {
                explicit_stage = Some(args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?);
                i += 2;
            }
            "--lora-rank" => {
                cfg.lora_rank = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--lora-alpha" => {
                cfg.lora_alpha = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--steps" => {
                cfg.steps = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--codec-steps" => {
                cfg.codec_steps = Some(args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?);
                i += 2;
            }
            "--batch" => {
                cfg.batch_size = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--lr" => {
                cfg.learning_rate = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--min-lr" => {
                cfg.min_lr = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--warmup" => {
                cfg.warmup_steps = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--weight-decay" => {
                cfg.weight_decay = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--grad-clip" => {
                cfg.grad_clip_norm = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--commitment" => {
                cfg.commitment_weight = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--ema-decay" => {
                cfg.ema_decay = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--dead-code" => {
                cfg.dead_code_threshold = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--seed" => {
                cfg.seed = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--checkpoint-dir" => {
                checkpoint_dir = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--checkpoint-every" => {
                cfg.checkpoint_every = args.get(i + 1).ok_or(TRAIN_USAGE)?.parse()?;
                i += 2;
            }
            "--resume" => {
                resume_dir = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--report" => {
                report_path = Some(PathBuf::from(args.get(i + 1).ok_or(TRAIN_USAGE)?));
                i += 2;
            }
            "--overwrite" => {
                overwrite = true;
                i += 1;
            }
            _ => return Err(TRAIN_USAGE.into()),
        }
    }

    if config_path.is_none() && init_from.is_none() {
        return Err("ncm train requires either --config FILE or --init-from ARTIFACT".into());
    }
    if config_path.is_some() && init_from.is_some() {
        return Err("pass either --config or --init-from, not both".into());
    }
    let data_path = data_path.ok_or("ncm train requires --data FILE")?;
    if output_dir.is_none() && save_adapter.is_none() {
        return Err("ncm train requires --output DIR and/or --save-adapter DIR".into());
    }
    let device = device.ok_or("ncm train requires explicit --device (cuda, mlx, or cpu)")?;

    cfg.stage = explicit_stage.unwrap_or(if init_from.is_some() {
        TrainStage::Generator
    } else {
        TrainStage::All
    });
    if cfg.min_lr > cfg.learning_rate {
        cfg.min_lr = cfg.learning_rate * 0.1;
    }

    Ok((
        TrainJob {
            config_path,
            init_from,
            data_path,
            val_data_path,
            tokenizer_path,
            output_dir,
            save_adapter,
            checkpoint_dir,
            resume_dir,
            device,
            overwrite,
            train_cfg: cfg,
        },
        report_path,
    ))
}

pub(super) fn run_train(args: &[String]) -> Result<()> {
    let (job, report_path) = parse_train_args(args)?;
    let report = run_training_job(&job)?;
    if let Some(path) = report_path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
    }
    let summary = serde_json::json!({
        "schema": report.schema,
        "device": report.device,
        "stage": report.stage,
        "output_artifact": report.output_artifact,
        "output_adapter": report.output_adapter,
        "dataset_samples": report.dataset_samples,
        "steps_completed": report.steps_completed,
        "total_seconds": (report.total_seconds * 1000.0).round() / 1000.0,
        "mean_step_ms": (report.mean_step_ms * 100.0).round() / 100.0,
        "train_precision": report.train_precision.as_str(),
        "grad_accum": report.grad_accum,
        "lora_rank": report.lora_rank,
        "trainable_params": report.trainable_params,
        "total_params": report.total_params,
        "codec_initial_loss": report.codec_initial_loss,
        "codec_final_loss": report.codec_final_loss,
        "generator_initial_loss": report.generator_initial_loss,
        "generator_final_loss": report.generator_final_loss,
        "final_val_loss": report.final_val_loss,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_train_args;
    use ncm_train::{TrainPrecision, TrainStage};
    use std::path::PathBuf;

    fn args(vals: &[&str]) -> Vec<String> {
        vals.iter().map(|s| (*s).into()).collect()
    }

    #[test]
    fn train_arg_parser_validates_scratch_and_finetune_modes() {
        let (job, rep) = parse_train_args(&args(&[
            "ncm",
            "train",
            "--config",
            "cfg.json",
            "--data",
            "train.json",
            "--output",
            "out",
            "--device",
            "cpu",
            "--steps",
            "50",
            "--batch",
            "4",
            "--lr",
            "0.005",
            "--report",
            "rep.json",
            "--overwrite",
        ]))
        .unwrap();
        assert_eq!(job.train_cfg.stage, TrainStage::All);
        assert_eq!(job.train_cfg.steps, 50);
        assert_eq!(job.train_cfg.batch_size, 4);
        assert!(job.overwrite);
        assert!(rep.is_some());

        let (ft, _) = parse_train_args(&args(&[
            "ncm",
            "train",
            "--init-from",
            "base-model",
            "--data",
            "train.jsonl",
            "--val-data",
            "val.jsonl",
            "--eval-every",
            "10",
            "--output",
            "out-ft",
            "--device",
            "cuda",
            "--grad-accum",
            "4",
            "--train-precision",
            "bf16",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--save-adapter",
            "out-adapter",
        ]))
        .unwrap();
        assert_eq!(ft.train_cfg.stage, TrainStage::Generator);
        assert_eq!(ft.train_cfg.lora_rank, 8);
        assert_eq!(ft.train_cfg.lora_alpha, 16.0);
        assert_eq!(ft.train_cfg.grad_accum, 4);
        assert_eq!(ft.train_cfg.train_precision, TrainPrecision::Bf16);
        assert_eq!(ft.train_cfg.eval_every, 10);
        assert_eq!(ft.val_data_path, Some(PathBuf::from("val.jsonl")));
        assert_eq!(ft.save_adapter, Some(PathBuf::from("out-adapter")));

        // Adapter-only mode without --output is accepted.
        let (adapter_only, _) = parse_train_args(&args(&[
            "ncm",
            "train",
            "--init-from",
            "base-model",
            "--data",
            "train.jsonl",
            "--save-adapter",
            "adapter-only",
            "--lora-rank",
            "8",
            "--device",
            "cuda",
        ]))
        .unwrap();
        assert!(adapter_only.output_dir.is_none());
        assert_eq!(
            adapter_only.save_adapter,
            Some(PathBuf::from("adapter-only"))
        );

        // Missing device or conflicting --config + --init-from rejected.
        assert!(
            parse_train_args(&args(&[
                "ncm", "train", "--config", "c", "--data", "d", "--output", "o"
            ]))
            .is_err()
        );
        assert!(
            parse_train_args(&args(&[
                "ncm",
                "train",
                "--config",
                "c",
                "--init-from",
                "m",
                "--data",
                "d",
                "--output",
                "o",
                "--device",
                "cpu"
            ]))
            .is_err()
        );
    }
}
