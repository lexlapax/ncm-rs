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

This is one small-model request on a shared desktop GPU, not full profile equivalence, quality,
performance, release qualification or Metal evidence. Unit tests also cover empty/oversize batches,
nonempty and whitespace strings, non-string JSON and two empty rows.
