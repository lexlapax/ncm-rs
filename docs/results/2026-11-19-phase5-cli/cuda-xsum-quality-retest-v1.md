# XSum CUDA top-1 quality retest and CLI corpus comparison

2026-11-20. Repeated the frozen XSum top-1, CFG=0 quality protocol on all 11,334 rows. The
previously observed data makes this a **nonblind retest**, not a new held-out experiment. Python
and strict-cuDNN Rust captures used the existing two-shard, batch-32 tools and pinned converted
dataset SHA-256 `cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a`.
All 178 batches per shard completed for both runtimes; the Rust capture bound the binary, source,
artifact and linked-library hashes. The Rust example SHA-256 was
`1ee2df6764debc3d16c188a39fcbe7d8b80f7b49fcf510111211d5d076e91d97`, and the v2 artifact manifest
SHA-256 was `8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4`.

The existing checked scorer ran separately for Python and Rust, for the full corpus and indices
256–11,333, with pinned ROUGE and GPT-2 perplexity evaluators. The existing comparator passed the
frozen quality rule for both ranges: all 544,032 raw IDs and decoded texts match, all per-row ROUGE
scores and aggregate metrics match. One intermediate-code difference remains at row 2,690, code
position 100 (Python 568, Rust 2617), the previously investigated
[Python head tie](../2026-11-15-phase4-xsum/cuda-reference-ties-v1.md); the final IDs match. It is
not an exact-code parity pass.

| Range | Rows | ROUGE-1 | ROUGE-2 | ROUGE-L | GPT-2 PPL | Scored tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 11,334 | 34.9236 | 12.7078 | 27.9258 | 82.6234 | 277,595 |
| Previously held out | 11,078 | 34.9312 | 12.7069 | 27.9256 | 82.6764 | 271,249 |

The full and held-out comparison receipts have SHA-256
`cbcdeffa98ab8ffa839e3c7e34558f6b899165d701c993b9bb10eb22c1d7ab94` and
`d448d197f7afdd19493fd9f29b085a2941a8d115658ce8bea422e78a2858cf5e`. The ignored Python shard
manifest hashes are `2d6ad6f9bcbcb2e69353f59a9dae9d5c692249f2631633e7fb3018b0b055f890` and
`25fbd6ca9084e03016206f44cd62734014b88b7834a71302cce0f0cce6ae35ea`; Rust manifests are
`78a90b42b7699ea3a4f917ea7ad8ee7fbbd63e9c91b8026eafb8095f7cb17cc4` and
`5ca85db7c6336c000b09ae56c9718a21bc8cb829ac00eeabc5099b6a40eec74b`.

Separately, the current unified CLI SHA-256
`6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278` ran each of the same 356
shard batches on `cuda-cudnn`. Its codes and IDs matched the validated Rust capture on **every
row**, including the investigated tie. Its comparison checked all per-batch receipts and hashed
the binary, dataset, artifact manifest and all 358 shard files before and after. The ignored
summary `reports/phase5-xsum-cli-corpus-cuda-v1/summary.json` SHA-256 is
`6f202f51f1d6ea090af7de883f222e325a8badeffb0a23fa99e797b93cfdfa93`.

Captures, scoring receipts and per-batch CLI inputs/outputs remain ignored under `reports/phase5-xsum-*`;
they include evaluation-derived data and are not published as a raw bundle. Generation and scoring
used `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false`; the Rust
worker and CLI used the pinned reference cuDNN path. The desktop GPU was shared and not
continuously monitored. This is the scoped CUDA top-1 quality retest and CLI ID-input comparison,
not stochastic/CFG qualification, a performance comparison, Metal evidence or the full release
gate. Full source-text encoding and selected text-input generation are separately documented in
[the text screen](cuda-text-input-v1.md).
