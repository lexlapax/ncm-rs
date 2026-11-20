# TinyStories five-seed CUDA quality retest

2026-11-20. Repeated the frozen [Phase 4 quality protocol](../2026-11-15-phase4-tinystories/development-quality-v1.md)
on the already observed official seeds `[42, 0, 1, 2, 3]`: 256 samples per seed per runtime,
generation batch 128, pinned upstream CUDA Python and current strict-cuDNN Rust. This is a
**nonblind retest**, not a new held-out cohort. Python and Rust intentionally use different RNG
streams; no cross-runtime sample identity is asserted.

The existing Rust `tinystories_samples` example was built with `cuda-cudnn` and generated the
candidate samples. For each seed, its first 128 codes/IDs matched the pre-change and current CLI
captures from the [five-seed routing screen](cuda-tinystories-five-seed-routing-v1.md); its second
128 matched newly generated pre-change and current unified CLI outputs with the example's chunk
seed `seed XOR 0x9e3779b97f4a7c15`. Thus all 1,280 scored Rust rows' codes and IDs match both CLI
surfaces on those fixed chunks. The current CLI's use of the ordinary single-request seed is not
claimed to reproduce Python's two-batch stream.

The qualified streaming GPT-2-large evaluator ran with batch 32 and output-projection chunk 4,
validating all sample shapes and upstream BPE decode/re-encode parity before scoring. It is the
same bounded evaluator and qualification described in the [Phase 4 report](../2026-11-15-phase4-tinystories/heldout-quality-v1.md):
the unmodified batch-32 scorer previously OOMed and its batch-8 comparison is not a direct
batch-32 equivalence proof. This run used Torch `2.15.0+cu132`, Transformers `4.57.3`, the pinned
GPT-2-large model and upstream revision. Results:

| Runtime | Five-seed mean genPPL | Five-seed mean entropy |
| --- | ---: | ---: |
| Python | 16.232494 | 3.968210 |
| Rust | 16.120773 | 3.967543 |

Rust minus Python genPPL was **−0.111721**, below the frozen +1.15 upper margin; absolute mean
entropy difference was **0.000667**, below 0.019. The frozen stochastic-quality component passes
on CUDA, subject to the evaluator caveat above. This does not establish performance, Metal or the
full Phase 5 gate. Per-seed values reproduce the published strict-CUDA table to its stated precision.

Generation used `tools/reference/tinystories_samples.py` for Python and the strict-CUDA Rust
example, with `--count 256 --batch 128` / `256 128`; scoring used
`tools/reference/tinystories_quality.py --batch 32 --stream-output --projection-chunk 4` on all
ten sample files. Commands ran with `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1`,
`TOKENIZERS_PARALLELISM=false`, reference cuDNN `LD_LIBRARY_PATH`, and
`PYTHONPATH=.reference/phase4-eval-deps` (absolute paths in the actual environment). The first
attempt without that isolated dependency path failed before Python inference with missing `regex`;
its partial output was retained separately in `reports/phase5-tinystories-quality-cuda-v1/`.
The successful sample captures, chunk comparisons and scoring receipt remain ignored under
`reports/phase5-tinystories-quality-cuda-v2/`; `quality.json` SHA-256 is
`303b602b017f6a496193ec471d4463b0fed75759d3ddb86c2d050b381a15be2c`.

The Rust example SHA-256 was `cbc6d85bd67c78499e492dde4fcfe1e42e90dc3bfc76a61582476a5e8bab3f39`,
Python generator `01a72753d5545af44e05e46b3e1a36c7de4107a4c5467fef5d830d5b7cabf031`, scorer
`e3f653086d7ff7d84559b2e44d74ab5537d0143823076f9829c44d8e202721c9`, artifact manifest
`0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1`, tokenizer payload
`fdf961a269fa4506a72d12056277450aaed13d31392b39fd640ff6db546516b4` and GPT-2-large weights
`5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce`; these and checkpoint
config/weights were hash-stable before/after. The upstream checkout remained clean at
`c70274273f489abc73832b2b98584a8483b6b203`. The shared desktop GPU was not exclusively reserved
and no continuous utilization capture was made; no latency or memory comparison is inferred.
