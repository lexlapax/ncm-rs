//! Help for the explicit-device CLI. Qualified and experimental paths are separate.

pub(super) const GENERAL: &str = "\
NCM inference (explicit device; no fallback)

Usage:
  ncm --version
  ncm doctor [--models-dir DIR] [--self-test] [--json]
  ncm inspect ARTIFACT
  ncm list [--models-dir DIR] [--model NAME=PATH]... [--json]
  ncm pull (--list | MODEL) [OPTIONS]
  ncm quantize ARTIFACT --precision q8|q4 --device cuda|mlx [--force]
  ncm run MODEL [PROMPT] [OPTIONS]
  ncm serve [OPTIONS]
  ncm train (--config FILE | --init-from ARTIFACT) --data FILE (--output DIR | --save-adapter DIR) --device DEV [OPTIONS]
  ncm sample ARTIFACT DEVICE (INPUT_JSON | --prompt TEXT) [OPTIONS]
  ncm generate|reconstruct ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]
  ncm sample-tinystories|sample-xsum|sample-wmt14|sample-lm1b ...

Run `ncm <command> --help` (`doctor`, `sample`, `serve`, `run`, `pull`, `quantize`, `train`, `list`) for details.
Profile-specific sample commands remain available. The XSum alias requires explicit
experimental flags for positive CFG and stochastic sampling. WMT14 CFG scales other
than 0 or 1 require --experimental-cfg-wmt14 on both commands.
";

pub(super) const SAMPLE: &str = "\
Usage: ncm sample ARTIFACT DEVICE (INPUT_JSON | --prompt TEXT) [OPTIONS]

The artifact manifest selects Sudoku, TinyStories, XSum, WMT14, LM1B or local-ncm-v1.
INPUT_JSON contains token-ID rows; with --text it contains an array of strings.
--prompt TEXT passes one UTF-8 prompt string inline (implies --text and --decode).
--text supports TinyStories (one non-padding token), XSum, WMT14 and local-ncm-v1
(when tokenizer.json is bundled). LM1B accepts only empty strings, mapped to [101];
nonempty prompts are unsupported.
LM1B v1 requires the pinned rotary export named lm1b-reference-rope-v1.safetensors
beside the artifact directory, or an explicit --reference-rope FILE. LM1B v2 bundles
the verified rotary tables inside inference.safetensors.

Options:
  --prompt TEXT                  Pass one text prompt inline (implies --text --decode).
  --seed N                       Seed TinyStories, LM1B (default 0) or local-ncm-v1.
  --cfg S                        Classifier-free guidance scale (default 0).
  --precision fp32|bf16|fp16|q8|q4 Compute and weight precision (default fp32).
  --adapter DIR                  Apply a verified unmerged ncm-lora-adapter-v1 directory.
  --decode                       Include evaluator-decoded text for text profiles.
  --plain                        Print decoded text lines directly instead of JSON (implies --decode).
  --text                         Encode text input; LM1B accepts only empty strings.
  --reference-rope FILE          Override default LM1B rotary table path.
  --host-fp64                    LM1B Metal host FP64 sampling.
  --top-p-fp32                   Experimental LM1B Metal top-p sampler.
  --experimental-stochastic-xsum Experimental XSum sampling; requires --seed.
  --experimental-positive-cfg-xsum Experimental positive-CFG XSum; requires --cfg > 0.
  --experimental-cfg-wmt14       Experimental WMT14 CFG other than 0 or 1.

Devices are explicit; availability depends on the build and profile. WMT14's
mlx-retry-padded device is experimental. OWT inference is unsupported.
Legacy sample-xsum positional CFG/seed also require the corresponding experimental flags.
No device or sampling fallback is performed.
";

pub(super) fn requested(args: &[String]) -> Option<&'static str> {
    match args {
        [_, flag] if flag == "--help" || flag == "-h" => Some(GENERAL),
        [_, command, flag] if command == "doctor" && (flag == "--help" || flag == "-h") => {
            Some(crate::doctor::DOCTOR_HELP)
        }
        [_, command, flag] if command == "sample" && (flag == "--help" || flag == "-h") => {
            Some(SAMPLE)
        }
        [_, command, flag] if command == "serve" && (flag == "--help" || flag == "-h") => {
            Some(crate::serve::SERVE_HELP)
        }
        [_, command, flag] if command == "list" && (flag == "--help" || flag == "-h") => {
            Some(crate::serve::LIST_HELP)
        }
        [_, command, flag] if command == "run" && (flag == "--help" || flag == "-h") => {
            Some(crate::serve::RUN_HELP)
        }
        [_, command, flag] if command == "pull" && (flag == "--help" || flag == "-h") => {
            Some(crate::serve::PULL_HELP)
        }
        [_, command, flag] if command == "quantize" && (flag == "--help" || flag == "-h") => {
            Some(crate::quantize::QUANTIZE_HELP)
        }
        [_, command, flag] if command == "train" && (flag == "--help" || flag == "-h") => {
            Some(crate::train::TRAIN_HELP)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{GENERAL, SAMPLE, requested};

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }

    #[test]
    fn only_explicit_help_invocations_bypass_inference() {
        assert_eq!(requested(&args(&["ncm", "--help"])), Some(GENERAL));
        assert_eq!(requested(&args(&["ncm", "sample", "-h"])), Some(SAMPLE));
        assert_eq!(
            requested(&args(&["ncm", "sample", "--help", "extra"])),
            None
        );
        assert_eq!(requested(&args(&["ncm", "inspect", "--help"])), None);
        for option in [
            "--top-p-fp32",
            "--experimental-stochastic-xsum",
            "--experimental-positive-cfg-xsum",
            "--experimental-cfg-wmt14",
            "mlx-retry-padded",
        ] {
            let line = SAMPLE.lines().find(|line| line.contains(option)).unwrap();
            assert!(line.to_ascii_lowercase().contains("experimental"), "{line}");
        }
    }
}
