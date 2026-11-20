# TinyStories CUDA five-seed routing comparison

2026-11-20. For each frozen official seed `[42, 0, 1, 2, 3]`, a 128-row batch of `[0]`
prefixes was generated using the pre-change `sample-tinystories` command, the current alias and
the current unified `sample` command, on `cuda-cudnn`. All three stdout files matched byte-for-byte
for every seed. Each parsed output had the expected seed, 128 rows, 641 codes and 128 IDs per row,
with all values within the checked vocabulary bound. These are 640 generated rows, not the full
five-seed 256-sample-per-seed quality gate.

The shared artifact `artifacts/canonical/tinystories-phase4-v1` manifest SHA-256 was
`0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1`; the compact input JSON
SHA-256 was `f3c19d788476477d6f2a944175166f50b4de81663d587a127d9a650d25e5c691`. The
pre-change CLI SHA-256 was `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`;
the current release CLI SHA-256 was `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`.
All four identities matched before and after. Each invocation used `setpriv --inh-caps=-all`,
`OMP_NUM_THREADS=1` and the reference cuDNN library path; the GPU was shared with the desktop.

| Seed | Common output SHA-256 |
| --- | --- |
| 42 | `e1b503119e5c394d1d8c5fe3273f45079a0a77f2f3e77c2e0e4fd72176660ffa` |
| 0 | `83a84375ccf36bebc47aa6dfbe746727d14e614be13e157097a379c8609d3bce` |
| 1 | `156c476d05869d30e090d9b48dc576c130cc1664e2d095c1dcf911f8069c252d` |
| 2 | `32fac0f1f9de1a44c583e3aa8e94e5923d2c437b021a995eece5ef65b3400c71` |
| 3 | `9e67116cb59a4da6671399f68818ed473c1ddea0e936851017ccc0d79c0430b9` |

The ignored input and output captures remain at `reports/phase5-tinystories-five-seeds-cuda-v1/`.
This compares Rust CLI routing, not the Python RNG stream, evaluator quality, performance or Metal.
The frozen [quality protocol](../2026-11-15-phase4-tinystories/development-quality-v1.md) requires
256 samples per seed and separate evaluator scoring.
