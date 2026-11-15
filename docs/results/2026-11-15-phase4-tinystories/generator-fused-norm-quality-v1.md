# Fused generator normalization: TinyStories quality retest

The isolated Metal candidate generated 256 samples at batch 128 for each official seed. The same pinned original Python CUDA samples, GPT-2-large evaluator and previously frozen criteria were used: candidate mean generative perplexity must be at most Python's mean plus 1.15, and the absolute mean entropy difference must be at most 0.019 nats. Earlier official-seed results had already been observed, so this is a **nonblind retest**, not a new held-out comparison. The candidate's sampled sequences differ from the earlier Rust Metal outputs; [initial numerical screening](generator-fused-norm-mlx-screen-v1.md) documents those differences.

| Seed | Python genPPL | Candidate genPPL | Python entropy | Candidate entropy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 16.572501 | 16.159859 | 3.966081 | 3.970828 |
| 1 | 15.659647 | 15.958079 | 3.964875 | 3.971773 |
| 2 | 16.402500 | 15.975153 | 3.976839 | 3.960740 |
| 3 | 16.351816 | 16.361276 | 3.973933 | 3.969839 |
| 42 | 16.176005 | 16.021596 | 3.959324 | 3.969768 |
| Mean | 16.232494 | 16.095193 | 3.968210 | 3.968590 |

The candidate mean PPL is 0.137301 lower, and its absolute mean entropy difference is 0.000379; both frozen stochastic-quality criteria pass. Each candidate sample's IDs, decoded text and re-encoded tokens passed the pinned BPE preflight during scoring. The scorer used the same qualified memory-bounded batch-32 output projection, not the unmodified scorer that previously exhausted GPU memory; [qualification and limits](heldout-quality-v1.md). This is a quality result only: it does not establish exact sampled-output identity, exhaustive numerical parity, Metal speed/memory or Phase 4 completion.

Capture inputs, source and binaries remained hash-stable, and the five copied sample hashes matched the Mac originals. The capture had 12 approximately two-second CPU activity samples and no GPU activity measurement. During scoring, a single observed Linux GPU snapshot showed 6,342 MiB used and 1,519 MiB free; before/after snapshots and other desktop processes were present, and no continuous GPU telemetry or exclusive reservation was available. No clock setting changed.

Candidate capture manifest `reports/phase4-fused-norm-mlx-quality-v1/manifest.json` SHA-256 is `8a333f5444e3a3b5b25274c45ae861899ac28a9d89095bb25823ffb3d8674932`; score JSON `reports/phase4-fused-norm-mlx-score-v1/quality.json` is `f24fcc4b7c5c5713d6ad5efd9243502ac8cc2cc7b7b33bbaeab7e8c30117a14d`. The input hash lists matched before and after scoring. The retained Python baseline score JSON has SHA-256 `a5ba3ba02d5fa49ee758692b43957c4e5d73ecfc0e83c9986d23e7b694a7e2fb`.
