# Experimental generator Q/K RMS and rotary fusion

This isolated experiment builds on the hybrid-normalization and compiled exact-GELU candidates. It combines the existing weighted RMS normalization and split-half rotary equations for FP32 width-64 Q/K rows on MLX at batch 16 or larger. The shared default retains the existing separate operations, including on CUDA. No model parameters, upstream pin, acceptance limits, or production code are changed.

The Metal reduction assumes a 32-lane SIMD group; this is an unqualified experimental assumption, not a portability guarantee. Floating-point rounding, output identity, full gates, speed, memory and tail behavior require measurement. The compiled GELU dependency in this experiment is the unguarded candidate; its separate initialization error limitation remains open.

Linux first: `cargo fmt --all`, `RUSTC_WRAPPER= cargo check -p ncm-model -p ncm-core`, replay of the three patches into a clean archive with byte comparisons, and the release CUDA pinned synthetic generator golden plus strict-cuDNN retained TinyStories batch passed. The CUDA log is `reports/phase4-qk-rope-cuda-v1.log` (SHA-256 `1d3d021f88f1cb4ce0868821eca4b5e596e972769849e86fa7f156d60a6b31b7`). These focused checks are not a full CUDA gate.

Next: run the focused Metal operator comparison and, if successful, screen seed identity and old/new performance on the available Mac. A positive screen would still require full Linux then Mac gates and the existing quality, numerical and repeated Python comparisons.
