# XSum CUDA quality after guidance support

The official CFG-zero comparison was rerun after the generator gained optional
guidance. The source-bound capture covers all 11,334 rows and matches pinned
Python on all 544,032 raw IDs and decoded texts. One intermediate code differs
at row 2,690, the same [previously investigated exact Python head tie](native-full-progress-v1.md);
the other 2,266,799 codes match. No tolerance or quality rule changed.

The full and held-out frozen quality comparisons both pass. Full ROUGE-1/2/L
is 34.9236/12.7078/27.9258, GPT-2 perplexity is 82.6234 and mean output length
is 20.6573 words for both runtimes. Held-out rows 256–11,333 give
34.9312/12.7069/27.9256, perplexity 82.6764 and 20.6512 words for both.
Per-row ROUGE and decoded text match as well. The pinned upstream evaluation
configuration itself specifies CFG zero; the separate positive-CFG
[development comparison](cuda-cfg-real-v1.md) is not a full quality gate.

Raw capture shards are `reports/phase4-xsum-quality-cfg-source-v1-s{0,1}/`.
Their manifests hash to
`4c238ace652a0183fabe216e9e9a464bc376419bd858006d0bd43d3979549196` and
`f6ee3e5d28162998d58c1182c4b96d37036caa5754b7cebbcfdc40b6b525d6d9`.
The full and held-out comparison JSON files under `reports/` hash to
`df263adac6ca0b06cd2ca8a71ff20a6aa98e8ead88f02f5493d4480f07b15596` and
`e675eb8b112e7ad198de1376772d9456bd1a1be8faa76b08375a59e05acbacfb`.
The initial scorer invocation had no visible CUDA device and failed before
producing scores; the subsequent GPU-enabled invocations passed.

The capture, scorer and comparator follow the commands and frozen rule in
[the prior quality report](native-quality-v2.md), using the new Rust shard
directories and fresh output paths. The manifest records the source, binary,
artifact, linked-library and GPU identity checked at capture and postflight.
General head ties, wider sampling and the XSum MLX gate remain open.

After the experimental categorical path changed the shared selection loop,
the same full and held-out CFG-zero capture and scoring were repeated on that
source. All 11,334 output rows and both quality metric sets again match Python;
the sole intermediate-code difference is still row 2,690. The two new shard
manifest hashes are
`6d264b21365fd4b8857e81962e03c93be77fb2560e4858f0c75527af64ed0929` and
`d07a016e340169161335da49f1c9ad0b79b19f353ebd19b22bb079f0aed0369b`;
the full and held-out comparison hashes are
`c654edcdaf2ccc373a8254cdb457a4d5f3e97fae6c75d7bf9d61591ebc0e25bb` and
`01f01e42e0aaf2468c0fa9928d338cd443bd17ca8410657f02c3634a6e635b87`.
The corresponding raw files are under
`reports/phase4-xsum-quality-stochastic-source-v1-s{0,1}/` and comparison
outputs in `reports/phase4-xsum-quality-stochastic-source-v1-comparison-*.json`.
