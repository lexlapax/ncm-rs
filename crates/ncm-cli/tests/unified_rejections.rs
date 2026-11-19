//! End-to-end rejection checks; synthetic manifests never reach artifact loading or inference.
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        fs::create_dir_all(&base).unwrap();
        let path = base.join(format!("unified-rejections-{}", std::process::id()));
        fs::create_dir(&path).expect("test scratch directory must be new");
        Self(path)
    }

    fn artifact(&self, name: &str, profile: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir(&path).unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join("manifest.json"))
            .unwrap();
        write!(file, "{{\"profile\":\"{profile}\"}}").unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn unified_command_rejects_unsupported_profiles_options_and_devices() {
    let scratch = Scratch::new();
    let cases: &[(&str, &str, &str, &[&str], &str)] = &[
        (
            "sudoku",
            "sudoku-easy-inspection-v1",
            "cuda",
            &["--decode"],
            "unsupported option for Sudoku",
        ),
        (
            "tiny",
            "tinystories-inference-v1",
            "cuda",
            &["--cfg", "1"],
            "unsupported option for TinyStories",
        ),
        (
            "xsum-seed",
            "xsum-inference-v2",
            "cuda-cudnn",
            &["--seed", "7"],
            "stochastic sampling requires both",
        ),
        (
            "xsum-cfg",
            "xsum-inference-v2",
            "cuda-cudnn",
            &["--cfg", "1"],
            "positive XSum CFG requires",
        ),
        (
            "wmt",
            "wmt14-inference-v1",
            "cuda",
            &["--seed", "7"],
            "unsupported option for WMT14",
        ),
        (
            "lm1b-text",
            "lm1b-small-inference-v1",
            "cuda-cudnn",
            &["--text"],
            "unsupported option for LM1B",
        ),
        (
            "owt",
            "owt-inference-v1",
            "cuda",
            &[],
            "unsupported artifact profile: owt-inference-v1",
        ),
        (
            "unknown",
            "unknown-inference-v1",
            "cuda",
            &[],
            "unsupported artifact profile: unknown-inference-v1",
        ),
        (
            "lm1b-device",
            "lm1b-small-inference-v1",
            "cpu",
            &[],
            "requires an explicit cuda-cudnn or mlx device",
        ),
        (
            "lm1b-top-p",
            "lm1b-small-inference-v1",
            "cuda-cudnn",
            &["--top-p-fp32"],
            "require explicit mlx device",
        ),
    ];
    for (name, profile, device, options, message) in cases {
        let artifact = scratch.artifact(name, profile);
        let output = Command::new(env!("CARGO_BIN_EXE_ncm"))
            .arg("sample")
            .arg(artifact)
            .arg(device)
            .arg(scratch.0.join("missing-input.json"))
            .args(*options)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{name}");
        assert!(output.stdout.is_empty(), "{name}: unexpected stdout");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(message), "{name}: {stderr}");
    }
}
