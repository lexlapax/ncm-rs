# XSum scale-one Metal guidance probe

The official XSum evaluation uses CFG zero. The Metal CLI still accepts only
that scale. This probe tests the separate positive-guidance path using eight
already observed real prefixes from the [CUDA scale-one comparison](cuda-cfg-real-v1.md).

At local commit `48d841d`, the strict CUDA CLI reproduced all 1,600
intermediate codes and 384 raw IDs in the retained pinned Python CUDA trace.
The input SHA-256 is
`beeb2c1fb6d46b0cdf3cfad61c3f9a6cd32a26db5099b6fe6342862edffdef36`.
The reference trace SHA-256 is
`fe20849f9da1d5c679a4cb5ad36e69f504c9eaa26a2981ec9b93177acec4186b`.

The adapted Python MLX model on Mac Studio produced the same outputs with
scale one. It paired the document and null prefixes, applied the existing
level-dependent scale, and fed the selected codes back to both halves. The
first attempt accidentally coerced an MLX value to NumPy through a scalar
multiplication and stopped at argmax. The corrected run uses explicit MLX
FP32 scalar arrays for both multiplications. The existing CPU FP64 selector
handled final tokens. The Mac service remained running. The result receipt
SHA-256 is
`90b8643863479afad6e92d9202d2851c500bdb8f7af0973fed612eef2c82a927`.
Source hashes, scripts, and both outputs are under ignored
`reports/phase4-xsum-cfg-cross-device-v1.F1qEar/`.

The Rust diagnostic exposes this schedule with explicit CPU FP64 final
selection. It shares the existing guided top-1 implementation and keeps
the request lock through final selection. The previous scale-zero and
device-final paths keep their existing behavior. The optional
`CFG_SCALE` argument on `xsum_top1_free_experimental` is for diagnostic
runs. The normal Metal CLI is unchanged.

The real-CUDA regression loads the same pinned trace and compares both
Rust final selectors against its codes and IDs. Access to the reference
weights and trace is explicit:

```sh
NCM_XSUM_CFG_ARTIFACT="$PWD/artifacts/canonical/xsum-inference-rope-v2-final" \
NCM_XSUM_CFG_TRACE="$PWD/reports/phase4-xsum-cfg-real-reference-v1/trace.safetensors" \
cargo test --locked --release -p ncm-backend-cuda --features cuda-cudnn \
  --test xsum_cfg -- --ignored
```

The strict CUDA regression passed using absolute fixture paths and the
local cuDNN library. Its cuDNN-only Clippy check passed with warnings denied. Its first run used relative paths from the Cargo
test working directory and could not find the assets. The full Linux
suite hit the temporary filesystem quota during an unrelated I/O test.
The next attempt supplied an on-disk temporary path, but hermetic test
children still used the system temporary directory. Moving the idle
benchmark build cache to ignored disk storage freed 5.5 GB and retained
the original path by symlink. The unchanged Linux suite then passed,
including 524 report and data tests and all 18 CUDA operator cases.
The failed logs are retained. The passing log SHA-256 is
`0606ec2e4498052e53767d100be021b17b9f35a4d6069d2dafb235fd2fb81bdc`.

After Linux validation, Mac Studio fast-forwarded to
`8c36edce0407ca9b96a5aeaec83616d41f0dbadf`. The first local wrapper used
a reserved zsh variable and stopped before starting the remote probe.
The corrected wrapper reused the clean checkout and existing inputs.
The focused Rust Metal build and Clippy check passed. The diagnostic
matched all 1,600 codes and all 384 raw IDs from the same eight real
prefixes at scale one. The binary SHA-256 is
`d83bfc8896a0e7734651252da0637cd8b37465408850782729d220817d14d5da`.
The comparison SHA-256 is
`6c167cb94c04e516205af7546b5b9cd1ab3973fc5912ffd19db01ff3c43c64d3`.

The full Mac suite `tools/check.sh --mlx` passed on that commit,
including all 18 real Metal operator cases. Its retained log SHA-256 is
`a8d70ce410342158b0382437c7270b8e07ff5515139cc730d1b1e3844e053a98`.
Afterward, both the rebuilt strict CUDA CLI and the same Rust Metal
diagnostic matched each of the pinned two-row traces at scales 0, 1
and 2. The Metal diagnostic rejected negative and NaN scales explicitly.
Service snapshots before and after the Metal probes match. The
three-scale receipt SHA-256 is
`796201a4d9ac42f630a6082ad9a4ace31f0c9343b2066a9407ea4a1df75c8f`.

The input and output identities, build logs and reproducer scripts are
in the Linux evidence bundle named above. These are retained-fixture
comparisons. Full positive-CFG quality and performance are unmeasured.
The production Metal CLI still accepts only CFG zero, and its full-corpus
exact gate still fails.
