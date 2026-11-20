# LM1B empty text CUDA screen

2026-11-20. The user approved accepting only empty strings for LM1B `--text`, mapped to the fixed
`[101]` start token, and rejecting nonempty prompts. The implementation accepts 1–32 rows and
uses the same bounded input reader and model path as ID inputs; it does not tokenize arbitrary text.

A CUDA-feature debug CLI built from the working tree with `cargo build -p ncm-cli --features
cuda-cudnn` had SHA-256 `7509d9dd32f6e27b3a5f8231d500aef8e3da2691a1a0ce7c30c997739cfeca3b`.
With `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1`, the reference cuDNN library path, retained
`artifacts/canonical/phase5-guide-lm1b-small-replay`, default pinned rotary sidecar, and seed 42,
`ncm sample ... --text --decode` on `[""]` and the same command without `--text` on `[[101]]`
produced byte-identical stdout SHA-256 `cd87171d7f866ad664cedf252c8eee86e857d5c66a00855468da73e009a423c6`.
For `["hello"]`, the text command exited 1, produced empty stdout, and reported that prompts are
unsupported. Inputs and captures are in ignored `reports/phase5-lm1b-empty-text-v1/`.

This initial screen is one small-model request on a shared desktop GPU, not full profile equivalence,
quality, performance, release qualification or Metal evidence. Unit tests also cover empty/oversize
batches, nonempty and whitespace strings, non-string JSON and two empty rows.

## All three sizes

2026-11-20. The CUDA-feature release CLI SHA-256
`6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278` ran small, medium and large
against `artifacts/canonical/lm1b-{small,medium,large}-inference-v1` with the pinned sibling rotary
sidecar, using seed 42 and `--decode`. For each size, `[""] --text` and `[[101]]` produced
byte-identical stdout; each output has one row, 641 codes, 128 IDs and a decoded string:

| Size | Manifest SHA-256 | Both outputs SHA-256 |
| --- | --- | --- |
| Small | `58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76` | `cd87171d7f866ad664cedf252c8eee86e857d5c66a00855468da73e009a423c6` |
| Medium | `2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c` | `c2995c6aaaa74f764f6602d6105aa0f51c121a2f752371d5cd6653e36f5a143e` |
| Large | `86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a` | `398b182c08dea307d3518c2382fd958d9e5b9bddf3870da33996e455e3bb44d1` |

The text input SHA-256 was `7c077ca2308714493d07163e1033c4282bd869ff6d477b3e77408587f95e2930`,
ID input `a283936d3db1e156d26a0ace722f476f1895782f23729587a45c4f838168e7dc`, and sidecar
`9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc`; these, the binary and
manifest hashes matched before and after the runs. Each request used `setpriv --inh-caps=-all`,
`OMP_NUM_THREADS=1`, reference cuDNN `LD_LIBRARY_PATH`, and the shared desktop GPU. Inputs and
outputs remain ignored under `reports/phase5-lm1b-empty-all-cuda-v1/`. This covers one row and
seed per size, not other batch sizes, quality, performance or Metal.
