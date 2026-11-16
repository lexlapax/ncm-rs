# Experimental XSum rotary artifact profile

The opt-in `xsum-inference-v2` profile adds `rope_cos` and `rope_sin` to
`inference.safetensors`; the existing `xsum-inference-v1` profile remains
available. The two added FP32 tensors contain 512 positions by 32 dimensions
and must match the exact hashes recorded in the separately prepared
[upstream rotary reference](rotary-reference-preparation-v1.md). The host
preflight rejects missing, incorrectly shaped or noncanonical rotary data.
The XSum generator builds its positional buffers from those tensors when the
new profile is selected; production inference does not invoke Python.

This profile is an intermediate step, not a correctness or performance gate.
On the retained 32-row Python shard-1 batch 0, both profiles produced identical
6,400 codes and 1,536 raw IDs and matched Python. On the previously failed
batch 77, the table-only v2 path completed but differed from Python on three
codes and five raw IDs. The other numerical changes found in the
[near-tie investigation](cuda-near-tie-investigation-v1.md) are not integrated.
The full XSum CUDA quality run and MLX runtime remain pending.
The later [native CUDA path](cuda-reference-native-v1.md) matched this failing
batch and additional samples; full quality and performance gates remain open.

A short exploratory comparison used the same batch-0 inputs and the same
newly built Rust binary, with two warmups and five measured trials per profile
on the RTX 3060 Ti. Median generation plus selection and host readback was
4.398 seconds for v1 and 4.420 seconds for v2; peak process RSS was 1,085,124
and 1,087,528 KiB, respectively. The CUDA default pool reported the same
used/reserved peak for both (2,308,162,448 / 3,053,453,312 bytes), excluding
other allocations. These sequential single-process screens did not collect
activity telemetry or a matched Python timing; they establish neither a
performance gate nor equivalence across inputs.

The final local export is `artifacts/canonical/xsum-inference-rope-v2-final/`,
with manifest SHA-256
`8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4`.
The earlier diagnostic export used for the CUDA screens differs from the final
export only in provenance metadata; all payload file hashes match. The first
export attempt used an unqualified tokenizer and correctly failed its hash
check. Two initial ignored preflight invocations used paths relative to the
crate directory and failed to find the artifact; the reruns with absolute
paths passed for both v1 and v2. Raw screen results and diagnostic binaries
are under `reports/phase4-xsum-tie-probe-v1/` and are not checked into Git.

The artifact can be reproduced into a new directory with:

```sh
.reference/venv/bin/python tools/reference/xsum_artifact.py --source artifacts/upstream/xsum-phase4 --tokenizer-asset artifacts/canonical/tokenizers-qualified/gpt2 --rope-reference reports/phase4-xsum-rope-reference-v5 --output artifacts/canonical/xsum-inference-rope-v2-final --report reports/phase4-xsum-rope-artifact-export-v2.json
NCM_XSUM_ARTIFACT="$PWD/artifacts/canonical/xsum-inference-rope-v2-final" RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 cargo test -p ncm-model --test loading real_xsum_generator_inventory_and_allowance_preflight -- --ignored
```

The exporter refuses an existing output or report. Targeted Rust and Python
tests include invalid profile data and invalid rotary tensor checks. The full
local `tools/check.sh --cuda` passed, including 18 real-device operator cases;
its log is `reports/phase4-xsum-rope-profile-cuda-check-v1.log`. The script
used `TMPDIR=reports/phase4-xsum-check-tmp`, which was removed after it became
empty. This validates the implementation checks, not the failed model-quality
gate.
