//! Compile optional CUDA-only layout kernels; ordinary CPU builds need no SDK.
use std::{env, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=src/generator_layout.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let output =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR")).join("generator_layout.ptx");
    let compiler = env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let status = Command::new(compiler)
        .args([
            "--ptx",
            "--gpu-architecture=compute_75",
            "-O3",
            "src/generator_layout.cu",
            "-o",
        ])
        .arg(output)
        .status()
        .expect("failed to start nvcc for CUDA layout kernels");
    assert!(
        status.success(),
        "nvcc failed compiling CUDA layout kernels"
    );
}
