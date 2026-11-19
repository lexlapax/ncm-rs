# Phase 5 text decode: CUDA screen

2026-11-19. On the real CUDA device, the unified command with `--decode` was run on the fixed requests in [the routing screen](cuda-routing-screen-v1.md). For each request, removing `texts` from the result reproduced the baseline codes, IDs and other fields exactly. The text for each of the six text profiles matched its pinned evaluator decoding on the generated IDs: the upstream TinyStories `BPETokenizer`, Transformers GPT-2 and Qwen3 for XSum and WMT14, and Transformers BERT for all three LM1B sizes. This is one output row per profile, **not** corpus-wide or Metal evidence.

| Profile | Decoded CLI stdout SHA-256 |
| --- | --- |
| TinyStories | `5dcc981e5e8bfa63e1d9c17f097e2213a1449be0ec9ac5d16618e7f96054cc4f` |
| XSum | `6db35c54fece2e27ea0e96e6f4e29cebba23a42bd20e899401218658cd2a0328` |
| WMT14 | `fd68c625bc98b4e7fccfde240db198c67ae8251e8607110b5319e8a617098d32` |
| LM1B small | `cd87171d7f866ad664cedf252c8eee86e857d5c66a00855468da73e009a423c6` |
| LM1B medium | `c2995c6aaaa74f764f6602d6105aa0f51c121a2f752371d5cd6653e36f5a143e` |
| LM1B large | `398b182c08dea307d3518c2382fd958d9e5b9bddf3870da33996e455e3bb44d1` |

The candidate binary SHA-256 was `12eb0b9fc45518cf4d01cab73b7316eec147f78159552466ec1f7a9153d37074`, built `--release --locked -p ncm-cli --features cuda-cudnn` with the build environment from the routing screen. The retained inputs, baseline and decoded JSON are in ignored `reports/phase5-cli-equivalence/`. Run with `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1` and the reference environment's cuDNN `LD_LIBRARY_PATH`.

For TinyStories, XSum and WMT14, the reference stops at the first EOS, removes padding, and decodes with special tokens skipped. LM1B uses the existing FLM behavior: decode the full row with BERT specials retained. GPT-2 and Qwen3 reference decoders return an empty string for their model-only padded-vocabulary IDs; the CLI filters those IDs within the model vocabulary before decoding. Unit tests exercise EOS, padding, model-only IDs and out-of-range rejection. The reference comparison used the pinned assets under `artifacts/upstream/tokenizers-v2/` and `.reference/upstream-ncm`, with `.reference/tokenizers-venv`.

Still pending: `--text`, corpus-exact encoding, no-extra-argument LM1B rotary setup, broader decode checks, Metal validation and performance evidence. No release gate is claimed.
