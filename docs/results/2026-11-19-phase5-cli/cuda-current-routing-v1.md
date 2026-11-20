# Current CLI CUDA routing and XSum flag screens

2026-11-20. On a shared desktop GPU, the [tracked equivalence runner](../../../tools/bench/phase5_cli_equivalence.py)
compared the retained pre-change baseline `reports/phase5-cli-equivalence/baseline-ncm` SHA-256
`f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098` with the CUDA-feature
debug CLI SHA-256 `7509d9dd32f6e27b3a5f8231d500aef8e3da2691a1a0ce7c30c997739cfeca3b`.
The candidate was built from the code at `d74e0b4` (subsequent commits before this run were
documentation only). For all nine existing fixed-profile requests, baseline alias, candidate alias
and unified stdout were byte-identical, with the same hashes as the previous
[routing screen](cuda-routing-screen-v2.md). The ignored result file
`reports/phase5-cli-equivalence-current-v1/result.json` SHA-256 is
`5cc84361c32d0890b80cdf15c297f8c01afc268b4a8f256a440c80e45b214b92`. The runner binds input,
manifest and binary hashes, checks output shape and detects post-run input changes. It does not
attest binary source. Its command was the earlier report's command with candidate `target/debug/ncm`,
the candidate hash above, and output `reports/phase5-cli-equivalence-current-v1`. The first attempt
correctly rejected a different debug-binary hash before creating output; rebuilding with
`cargo build -p ncm-cli --features cuda-cudnn` restored the candidate identity above.

Separately, three fixed XSum requests compared old unflagged alias, new explicitly flagged alias,
and new unified command on `artifacts/canonical/xsum-inference-rope-v2-final` (manifest
`8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4`) and
`reports/phase5-cli-equivalence/xsum.json` (SHA-256
`74703e40c8fd26fff5a6d919c6d40550a39a33b94183455739093947b7ef2489`). All three outputs in each
case were byte-identical:

| Case | Legacy positional values | Required alias flags | Output SHA-256 |
| --- | --- | --- | --- |
| Positive CFG | `1` | `--experimental-positive-cfg-xsum` | `75aeac4fe53bb7416fbd2d8a75814671ca84d5721eb30b8699d09b7ecd2b57ed` |
| Stochastic | `0 42` | `--experimental-stochastic-xsum` | `801defdf5ed80c6a0bf17c75c2a426789293b169b827d722aee0a11cc713a215` |
| Both | `1 42` | both flags | `afe30c1c93dcd35c9f0308b4b8d8a48ef8cfa7f2b12d6e42dc8dfaeb340604b3` |

The unified calls used the corresponding `--cfg 1` and/or `--seed 42` plus the same flags. All
runs used `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1` and the reference cuDNN library path.
Raw captures remain ignored in `reports/phase5-xsum-alias-flags-cuda-v1/`. Host tests separately
check rejection of unflagged experimental aliases. These are fixed-input screens, not full corpus,
quality, performance, alternate Sudoku artifact qualification or Metal evidence.

## Post-WMT14 boundary nine-profile rerun

On 2026-11-20, the same nine fixed requests were rerun on CUDA with the pre-change baseline and a release candidate built from `d5b2931`, which includes the WMT14 CFG boundary. All baseline, alias and unified stdout streams were byte-identical per profile; all nine hashes matched the previous screen. The WMT14 request here uses default CFG=0; the separate [CFG boundary screen](cuda-wmt14-cfg-boundary-v1.md) exercises scales 0/1/2 and the [CFG=1 corpus comparison](cuda-wmt14-cfg1-quality-retest-v1.md#legacy-and-pre-change-aliases) covers all three routes.

Candidate SHA-256 `83f9a9a243ac83a7b8f6cf888ddebcdafa9fd49f7dbf45b44df7d1a3429bd8ce`; baseline `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`; runner `2af9fdd181a6ab05ce8c1f4bfddbbd99c4eaf81985fa564943f20e6178cad81a`. Ignored result `reports/phase5-cli-equivalence-current-v2/result.json` SHA-256 `3bf0a020cac4cb3cbdbfb3ce431293decd7e9c0da474fd14a04c28a98b34e016`. The runner checked input and binary hashes before and after; an independent replay checked saved stdout equality and hashes against the previous nine-profile screen. Five focused runner tests and `setpriv --inh-caps=-all tools/check.sh` passed. This is fixed-input CUDA equivalence only, not an exhaustive option matrix, model-quality or performance gate, or Metal evidence; raw outputs remain local.

Reproduce with the [documented routing command](cuda-routing-screen-v2.md), substituting `target/release/ncm`, the candidate hash above and a fresh output directory. Build with `cargo build -p ncm-cli --features cuda-cudnn --release --locked` and the same CUDA/cuDNN environment; the runner verifies the provided binary hash but cannot prove its source.
