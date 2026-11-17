# LM1B-small CUDA fixed-history generator reference

Two previously inspected seed-11 development-code rows were used with BERT
CLS prefix 101. The pinned upstream small generator receives each row's first
code explicitly, matching the evaluator's unigram override of query-mode
level zero. For the remaining eleven levels, an instrumented sampling hook
returns the retained codes while recording the unmodified head logits. Hooks
also record the eleven post-upsampling convolution inputs and outputs. The
trace includes all twelve reconstruction levels, final decoder logits, greedy
IDs and IDs from the pinned upstream frequency/deficit/adjacency selection.
This is a fixed-history reference, not a test of stochastic sampling.

Two fresh CUDA processes with deterministic cuDNN, TF32 disabled and math SDPA
produced identical 51-tensor payloads and reports. Their report SHA-256 is
`00f20a4a67127293d5d01d3c146e0e3cf3cc9a2aa1992570e36901599b886d0c`;
the safetensors SHA-256 is
`fd815da45bc20fde58aa44e030aeb94e032a42ba71baa1f355f09fc91fd5223b`.
The trace's forced codes and penalty-adjusted IDs exactly match the two source
development rows. The reference source SHA-256 is
`19da672743305be6fdc7fa8e4780d3a32de073154a02090521759a736077cc37`.
The pinned checkpoints, config, evaluator, upstream checkout and watched source
are checked before and after. Earlier v1/v2 traces omitted the penalty-adjusted
IDs and are retained; v3/v4 include them. All raw evidence remains in ignored
`reports/phase4-lm1b-small-teacher-cuda-reference-v{1,2,3,4}/`.

The command was `setpriv --inh-caps=-all env OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false .reference/venv/bin/python
tools/reference/lm1b_small_teacher_trace.py --upstream .reference/upstream-ncm
--source artifacts/upstream/lm1b-phase4 --prefixes
reports/phase4-lm1b-small-cuda-development-v1.json --output NEW_REPORT_DIR`.
The focused input test validates metadata, full row counts, widths and code
ranges. This is two inspected histories, not a Rust comparison, held-out
quality result, performance result or Metal gate.

The repository standard `tools/check.sh` passed; log SHA-256
`5a36e0bbd935138e77451882ba2ab6af1f78598d9bea84b52e1ad62d8402ab70`.
