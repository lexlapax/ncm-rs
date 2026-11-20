//! Help for the explicit-device CLI. Qualified and experimental paths are separate.

pub(super) const GENERAL: &str = "\
NCM inference (explicit device; no fallback)

Usage:
  ncm inspect ARTIFACT
  ncm sample ARTIFACT DEVICE INPUT_JSON [OPTIONS]
  ncm generate|reconstruct ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]
  ncm sample-tinystories|sample-xsum|sample-wmt14|sample-lm1b ...

Run `ncm sample --help` for profiles, options and experimental paths.
Profile-specific sample commands remain available. The XSum alias requires explicit
experimental flags for positive CFG and stochastic sampling.
";

pub(super) const SAMPLE: &str = "\
Usage: ncm sample ARTIFACT DEVICE INPUT_JSON [OPTIONS]

The artifact manifest selects Sudoku, TinyStories, XSum, WMT14 or LM1B.
INPUT_JSON contains token-ID rows; with --text it contains an array of strings.
--text supports TinyStories (one non-padding token), XSum and WMT14. LM1B
accepts only empty strings, mapped to [101]; nonempty prompts are unsupported.
LM1B requires the pinned rotary export named
lm1b-reference-rope-v1.safetensors beside the artifact directory,
or an explicit --reference-rope FILE.

Options:
  --seed N                       Seed TinyStories or LM1B (default 0).
  --cfg S                        Classifier-free guidance scale (default 0).
  --decode                       Include evaluator-decoded text for text profiles.
  --text                         Encode text input; LM1B accepts only empty strings.
  --reference-rope FILE          Override default LM1B rotary table path.
  --host-fp64                    LM1B Metal host FP64 sampling.
  --top-p-fp32                   Experimental LM1B Metal top-p sampler.
  --experimental-stochastic-xsum Experimental XSum sampling; requires --seed.
  --experimental-positive-cfg-xsum Experimental positive-CFG XSum; requires --cfg > 0.

Devices are explicit; availability depends on the build and profile. WMT14's
mlx-retry-padded device is experimental. OWT inference is unsupported.
Legacy sample-xsum positional CFG/seed also require the corresponding experimental flags.
No device or sampling fallback is performed.
";

pub(super) fn requested(args: &[String]) -> Option<&'static str> {
    match args {
        [_, flag] if flag == "--help" || flag == "-h" => Some(GENERAL),
        [_, command, flag] if command == "sample" && (flag == "--help" || flag == "-h") => {
            Some(SAMPLE)
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
            "mlx-retry-padded",
        ] {
            let line = SAMPLE.lines().find(|line| line.contains(option)).unwrap();
            assert!(line.to_ascii_lowercase().contains("experimental"), "{line}");
        }
    }
}
