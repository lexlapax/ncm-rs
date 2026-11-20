# WMT14 CFG boundary: fixed-request CUDA screen

2026-11-20. The user selected option 1 in the [policy audit](wmt14-cfg-policy-audit-v1.md): CFG 0 and 1 stay unflagged; other positive scales require `--experimental-cfg-wmt14` on unified and legacy commands. This is an interface decision, not a new quality exception. The flag is rejected when unnecessary and on other profiles. Existing non-0/1 legacy calls must add it.

A shared-desktop CUDA fixed-input screen compared the pre-change binary's positional alias with the candidate's alias and unified command. The latter two used the flag only for scale 2. All three stdout streams were byte-identical per scale:

| CFG | stdout SHA-256 |
| --- | --- |
| 0 | `17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a` |
| 1 | `374b657b197af78011c8f532aed3c5b1e868631e263562d2ac906839f0159fcb` |
| 2 (experimental) | `4d22adc794c2c8418345f94588af8764eb0570a10f9047845465804b526e0a0e` |

Both candidate routes rejected unflagged scale 2 with empty stdout and the expected flag error. Baseline binary SHA-256 `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`; candidate debug binary SHA-256 `d9a4e2ac4e147aaa6f33ceae808afc9f05da145230439f7653abb5d3c9bb9ffc`; artifact manifest `01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81`; input SHA-256 `8b0fd6d29cacfd8acf1e033a3664aec21c2a28976af9f9fc1ad77d62b1c8a825`. Candidate built from this change before the final test-only integration additions. Captures remain local under ignored `reports/phase5-wmt14-cfg-boundary-v1/`.

Build used `setpriv --inh-caps=-all cargo build -p ncm-cli --features cuda-cudnn --locked`, `LIBRARY_PATH=$PWD/.reference/phase4-cudnn-probe-lib`, the reference cuDNN `LD_LIBRARY_PATH`, `NVCC_CCBIN=/usr/bin/g++-15`, `CUDA_COMPUTE_CAP=86`, `CARGO_BUILD_JOBS=4` and `OMP_NUM_THREADS=1`. Each inference used `setpriv --inh-caps=-all`, `artifacts/canonical/wmt14-inference-v1-final`, device `cuda` and `reports/phase5-cli-equivalence/wmt14.json`:

```sh
OLD=reports/phase5-cli-equivalence/baseline-ncm
NEW=target/debug/ncm
ART=artifacts/canonical/wmt14-inference-v1-final
INPUT=reports/phase5-cli-equivalence/wmt14.json
# Repeat with S=0 and S=1 with no FLAG; S=2 uses FLAG below.
S=2
setpriv --inh-caps=-all "$OLD" sample-wmt14 "$ART" cuda "$INPUT" "$S" > old.json
setpriv --inh-caps=-all "$NEW" sample-wmt14 "$ART" cuda "$INPUT" "$S" --experimental-cfg-wmt14 > alias.json
setpriv --inh-caps=-all "$NEW" sample "$ART" cuda "$INPUT" --cfg "$S" --experimental-cfg-wmt14 > unified.json
cmp old.json alias.json && cmp old.json unified.json
```

The focused CLI tests and `setpriv --inh-caps=-all tools/check.sh` passed; 12 artifact-dependent CLI tests were ignored. These are parser and one fixed CUDA request per scale, not full-corpus quality, performance, Metal, or release qualification. No frozen thresholds, artifact hashes or previous failures changed.
