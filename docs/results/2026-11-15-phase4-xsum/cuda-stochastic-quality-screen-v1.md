# XSum CUDA stochastic quality screen

The first 256 XSum development passages were sampled with temperature 1,
top-k 0, top-p 1 and CFG 0, in eight batches of 32. Each batch starts a fresh
process and uses seed 42 through 49; Rust's SplitMix64/inverse-CDF stream and
Python's PyTorch multinomial stream are different. Consequently these are
different generated summaries, not same-seed parity checks. The same pinned
ROUGE and GPT-2 evaluators used for the official comparison score both sets.

| Rows | Runtime | ROUGE-1 | ROUGE-2 | ROUGE-L | GPT-2 PPL | Mean words |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | Rust | 32.2345 | 10.2440 | 24.8274 | 195.3415 | 20.7188 |
| 32 | Python | 32.4411 | 9.7632 | 25.0342 | 173.2272 | 20.5938 |
| 256 | Rust | 32.8434 | 11.0635 | 25.7073 | 183.6247 | 20.8047 |
| 256 | Python | 31.1198 | 9.9003 | 24.6353 | 188.1584 | 20.8164 |

The 32-row perplexity difference reversed in the larger sample. This is a
descriptive development screen, not a held-out test, statistical equivalence
claim or acceptance gate; seeds and samples are too limited to establish
stochastic quality. The controlled-draw exact-decision failures remain
[documented separately](cuda-stochastic-investigation-v1.md).

Inputs, samples, source/binary/input fingerprints and scores are in
`reports/phase4-xsum-stochastic-quality-256-v1/`; fingerprints matched after
generation. The 32-row score SHA-256 is
`b35f18b796f2a84749f26b9b3f87b59f25e70b944e3a514d68b492ff122a7d06` and the
256-row score is
`4cf04df988d9a0ed2fcc888277a70dc886f1a42128633e580b1bd6ccd003e1df`.
The earlier 32-row score, using the same outputs and a previous scorer version,
has identical metrics. GPU activity and native linked-library hashes were not
captured for this exploratory run.

For each batch `b` from 0 to 7, extract the corresponding 32 padded prefixes
from the hash-verified XSum dataset, use seed `42+b`, and run the pinned
Python `xsum_cfg_trace.py --cfg-scale 0 --prefix-json INPUT --decisions-only
--stochastic --seed SEED` and native `ncm sample-xsum ARTIFACT cuda-cudnn
INPUT 0 SEED` in separate processes. Run `xsum_stochastic_quality_screen.py`
with all eight Python trace directories in order under `--python-reference`,
the eight native JSON files in order under `--rust`, and pinned `--data` and
`--upstream` paths. The scorer validates data, input rows, trace/checkpoint
hashes and evaluator files before scoring; use `OMP_NUM_THREADS=1` and the
local cuDNN path for native generation.
