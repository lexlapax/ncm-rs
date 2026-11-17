# LM1B checkpoint inventory preparation

On 2026-11-16, `.reference/venv/bin/python tools/reference/lm1b_inventory.py
artifacts/upstream/lm1b-phase4 > reports/phase4-lm1b-inventory-prep-v1.json`
verified the four checkpoint sizes and SHA-256 values already documented in
[the Phase 4 contract](../../PHASE4-TEXT.md), before and after CPU-only
`torch.load(weights_only=True, mmap=True)` inspection. The inventory JSON has
SHA-256 `058724305bdda8f08c09d5ee5e9514aa44e43ab05e96e9a2a9eee5d30b4a1814`.
It is ignored local evidence, not a model export or inference validation.

| Checkpoint | Model tensors | Model tensor bytes | Other checkpoint state |
| --- | ---: | ---: | --- |
| Small generator | 213 | 389,916,184 | 2 inference tensors, 154,880 bytes |
| Medium generator | 381 | 1,267,739,160 | 2 inference tensors, 154,880 bytes |
| Large generator | 549 | 2,900,856,344 | 2 inference tensors, 154,880 bytes |
| Shared tokenizer | 136 | 597,793,280 | none at top level |

Each generator contains one I64 model tensor; its other model tensors are FP32.
Each inference bundle contains FP32 `level0_unigram` `[8192]` and `token_unigram`
`[30528]`. The tokenizer has 124 FP32 and 12 I64 tensors. No shared-storage group
was observed *within* an inspected dictionary; this is not a cross-checkpoint alias
claim. The small config declares stochastic sampling, frequency and adjacency
penalties and an initial unigram; these require explicit qualification rather than
assuming TinyStories behavior. The initial inventory did not test values,
tokenizer behavior or quality; subsequent CPU preparation is described below.

The observed config identities are small: 2,749 bytes / SHA-256
`4db04299a87f7eb1627de0abb2f0b257e50e5ea0138edda771ad287cdabd7edf`, medium:
2,754 bytes / `93a42e0923b38f06f186072fdc37c1a2b4db5298c7577755390dd36c651d8142`,
and large: 2,753 bytes /
`4410e3710360cae162f8be4186a44a0c7c473be674a1ac3d1d940bfe45d5e70c`.
These were local observations at inventory time; subsequent verification
against the pinned checkpoint revision is recorded below.
Two focused tests passed for tensor metadata, alias reporting and malformed input
rejection. Runtime bounds, export, all three model implementations, tokenization,
quality and CUDA/Metal gates remain open.

A follow-up `lm1b_unigram_inspect.py` verified each generator checkpoint hash
before and after reading its two inference tensors. Each respective tensor has
identical bytes across all three model sizes. All six are finite and nonnegative;
the level-zero unigram has 6,428 zero entries of 8,192 and sums to
1.000000013737008 in FP64, while the token unigram has 2,937 zero entries of
30,528 and sums to 1.0000000025714701. Their raw tensor SHA-256 values are
`2b344cd27b1cb3c81d13131a1aaa88a12c7739617c8882839b8a688031064889` and
`c27c53a7567bf0348c9113948f58be65321bb1eb12dc68e569ff7396a99c5952`.
This establishes metadata properties, not sampling parity. Result:
`reports/phase4-lm1b-unigram-inspect-v1.json`, SHA-256
`15d33d83b681bce3d95a8ec9554f9cb0b04dd6f82da5bb61fc6d10f9d2491698`.
The combined four inventory/unigram unit tests pass under the reference venv.

Each config was subsequently fetched over HTTPS from the existing checkpoint
revision at [small](https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/d342ce8aaea58830eef3b8ad473664673e41f692/lm1b/small/config.yaml),
[medium](https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/d342ce8aaea58830eef3b8ad473664673e41f692/lm1b/medium/config.yaml),
and [large](https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/d342ce8aaea58830eef3b8ad473664673e41f692/lm1b/large/config.yaml).
Each downloaded file matched its local counterpart byte-for-byte and had the
respective SHA-256 above. Each read used `curl -fsSL --max-time 20
https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/d342ce8aaea58830eef3b8ad473664673e41f692/lm1b/SIZE/config.yaml
-o reports/phase4-lm1b-SIZE-upstream-config-v1.yaml` with `SIZE` replaced by
the corresponding model size. This
verifies the config identities at the existing revision without changing the
upstream revision or claiming an accepted runtime artifact.

The pinned upstream `scripts/eval_merge.py::genppl_flm` describes a distinct
LM1B scoring path: it decodes raw generated IDs with special tokens retained,
retokenizes with the reference GPT-2 tokenizer with truncation to 128 tokens,
and computes corpus-level perplexity in batches of 16 using its explicit
first-EOS validity expression. It separately reports mean per-sample entropy
from raw IDs and aggregates over configured seeds. This is a source observation;
no LM1B scorer has yet been implemented, validated or run. In particular, the
TinyStories evaluator cannot simply be assumed equivalent.

## Single-sample upstream CPU probes

Each checkpoint size was loaded strictly with the shared tokenizer checkpoint,
and a single batch-one sample was generated on CPU using seed 42, the observed
level-zero unigram, `[CLS]` prefix ID 101, configured stochastic generation and
the pinned evaluator's frequency/adjacency selection function. The command
pattern was `setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
.reference/venv/bin/python tools/reference/lm1b_cpu_probe.py --upstream
.reference/upstream-ncm --source artifacts/upstream/lm1b-phase4 --size SIZE
--output NEW_REPORT`. It verified the checkpoint hashes and locally observed
config hash before and after execution. Each output contains 641 codes and 128
raw IDs; the output SHA-256 values are:

| Size | SHA-256 |
| --- | --- |
| Small | `0b1299d3f9b874a25ceb769f4d94733ccce0a62d8e2bc46b33a72eb28b6f850a` |
| Medium | `0b9cb97801932f8bd81b30f4648691d4ee9a353a69941b6378e2fce20150fc13` |
| Large | `8a7a3b83bed32f3df2156a88a28721a4c74fcc5d603ae581adc76bd98496d3f6` |

The small probe was repeated in a fresh process and produced identical bytes.
The probe source SHA-256 is
`c4180ddc12df044671ce0721ffbf9ccd41731aa96b7eff42bc650803ee7ae43a`.
Raw reports and logs are under `reports/phase4-lm1b-*-cpu-probe-v1` with `.json`
and `.log` suffixes; the small repeat is v2. These are CPU preparation fixtures,
not the evaluator's batch-eight CUDA sampling, a quality comparison, or any
runtime or GPU qualification.

## Lossless CPU export preparation

`tools/reference/lm1b_export_preparation.py` verifies the pinned checkpoint
and config identities plus the existing qualified BERT tokenizer inputs before
and after CPU-only `torch.load(weights_only=True, mmap=True)`. It preserves
all generator, codec and inference tensors in separate safetensors files,
round-tripping names, shapes, dtypes and bytes, and checks both distributions.
It retains all codec state without classifying training-only tensors. Existing
outputs are rejected. These are experimental preparation artifacts, not
runtime artifacts or evidence of model, CUDA, Metal or quality support.

With `setpriv --inh-caps=-all env OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false .reference/venv/bin/python
tools/reference/lm1b_export_preparation.py --source artifacts/upstream/lm1b-phase4
--tokenizer-asset artifacts/canonical/tokenizers-qualified/bert --size SIZE
--output artifacts/canonical/lm1b-SIZE-preparation-v1`, the three sizes produced:

| Size | Generator / codec / inference tensors | Manifest SHA-256 |
| --- | --- | --- |
| small | 213 / 136 / 2 | `0be260578b8a61aebfb945e1da6b461177e9d913467652a51b592021162d9c70` |
| medium | 381 / 136 / 2 | `8c8aa3db9a15d5992b2ccd9c00a95abf96f4eaa14701390c830920b618ccbae8` |
| large | 549 / 136 / 2 | `ddb6fc17c59e89eaabb3f1ff79dde0f086f100b6b12aab4576825b7692c006a0` |

All three codec files are byte-identical, SHA-256
`badd506a9325168a5842582be36653235c1d8bc4c481241548f5ec3b90909853`;
the inference files also match, SHA-256
`d6e38383177c8fcf039ceca8f7031acd11a1cf2c686028bcde4c9956dd5bc640`.
The small export repeated in a separate output directory and every file
matched byte-for-byte. The exporter SHA-256 is
`6f6102ac950595d78f3e4f180c4762f083a2e03c306d9f2f0ddf8855430b6902`.
Three focused tests pass for round trips, malformed metadata/distributions and
overwrite rejection. Logs and artifacts remain ignored local evidence. Runtime
loader limits, model semantics and all GPU gates remain open.

The repository standard `tools/check.sh` completed successfully; log SHA-256
`4a13ff7ce406f9a75e6a4b50edc902770db31c3028ac2442ff05b8f55828654e`.
It does not establish model quality or GPU validation.
