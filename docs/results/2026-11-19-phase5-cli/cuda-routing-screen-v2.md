# Phase 5 CLI routing: reproducible CUDA screen

2026-11-19. The tracked `tools/bench/phase5_cli_equivalence.py` compares baseline, current alias and unified-command stdout in separate processes, validates output keys and dimensions, binds the pinned artifact-manifest hashes, pinned LM1B rotary table, requested binary hashes and input hashes, and checks those files plus the runner remain unchanged. It requires a new output directory and writes `result.json` only after all cases pass. It does not establish that a supplied binary was built from claimed source: that provenance must be established separately. It does not prove quality, performance, all inputs, text encoding or Metal correctness.

On CUDA the runner passed **nine profiles**, one fixed batch each: Sudoku easy/medium/hard, TinyStories, XSum, WMT14 and LM1B small/medium/large. The baseline binary was the `702168c` build from [the first screen](cuda-routing-screen-v1.md); candidate was the `bafcd6e` build from [the rotary screen](cuda-lm1b-rope-v1.md). The two newly covered Sudoku cases produced these common stdout SHA-256s:

- Medium: `b5a22993f2df60ece95b6b188e009a55cf10ad59a2591e69e478a025125d104d`.
- Hard: `655dbfdb50d1210923246a730a08332156df64418365328d55a2f20af8b7caf0`.

The seven other hashes match the first screen. The full local result is `reports/phase5-cli-equivalence-runner-v2/result.json`, SHA-256 `46a10d77cc877368e272bb31b5eddee478db0a08d901ee2d2c1efb0bb7451263`; runner SHA-256 `8318ab6d09c332fdd7f9e0558383d2d9a29c631a3742f25b7fac2f1035a5050f`. The following exact command produced it; the baseline and candidate must already be built and the inputs prepared as in the earlier screen:

```sh
setpriv --inh-caps=-all env LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" OMP_NUM_THREADS=1 \
  python3 tools/bench/phase5_cli_equivalence.py \
  --baseline reports/phase5-cli-equivalence/baseline-ncm \
  --baseline-sha256 f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098 \
  --candidate target/phase5-candidate/release/ncm \
  --candidate-sha256 4b52d2fb565b7196b4027b70ad6389006a5fa571531c285c7225159bb43c98ae \
  --artifacts artifacts/canonical --inputs artifacts/canonical/sudoku-data-phase3 \
  --xsum-input reports/phase5-cli-equivalence/xsum.json --wmt14-input reports/phase5-cli-equivalence/wmt14.json \
  --rope reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors \
  --platform cuda --output reports/phase5-cli-equivalence-runner-v2
```

Use a fresh output directory for a rerun. The runner uses explicit `--reference-rope`; the default-sidecar route was separately screened. Automated negative controls reject malformed output, byte differences, failed processes and identical baseline/candidate binaries. The earlier routing, text and decode screens retain their separate scopes and limitations. Mac execution, user decisions, fresh-clone setup and the release gate remain pending.

After that screen, the runner's output-path preflight was corrected to reject live and dangling
symlinks before resolving the destination. The updated runner SHA-256 is
`2af9fdd181a6ab05ce8c1f4bfddbbd99c4eaf81985fa564943f20e6178cad81a`; five offline tests and
`tools/check.sh` passed, including symlink negative controls. The nine-profile CUDA screen above
used the earlier runner hash. Output parent directories must remain trusted and stable.

### Updated-runner CUDA rerun

The updated runner was subsequently run on CUDA against the same nine fixed requests, using the
same baseline and the newer candidate binary SHA-256
`cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`. It reported all nine
baseline/alias/unified byte comparisons passing. Every output hash and row count matched the
prior screen. Its ignored `reports/phase5-cli-equivalence-runner-v3/result.json` SHA-256 is
`4ec4b555db3bafe437eb0a582b726d9be7518449f512186a0cbef7ea7322ad8b`. The command above was used
with that candidate hash and `--output reports/phase5-cli-equivalence-runner-v3`; all other inputs,
flags and environment were unchanged. This rerun tests the hardened runner on the same cases, not
additional corpus, quality, performance or Metal coverage.

## CLI rejection integration checks

`crates/ncm-cli/tests/unified_rejections.rs` invokes the actual CLI with ten synthetic manifests
and a nonexistent input. It checks exit status 1, empty stdout and the expected clear diagnostic
for unsupported Sudoku/TinyStories/XSum/WMT14/LM1B options, OWT and an unknown profile,
LM1B's unsupported CPU device, and the Metal-only LM1B top-p flag on CUDA. These cases all fail
before full artifact loading or inference; they are not a complete profile/device matrix or GPU
validation. The test passed with both default features and `cuda-cudnn`, and CLI all-target Clippy
with `cuda-cudnn` and default `tools/check.sh` passed. For the CUDA-feature test and Clippy, the
local cuDNN link/runtime paths and `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86` were set as
in the existing CUDA CLI screens.
