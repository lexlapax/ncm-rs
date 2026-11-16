# XSum CUDA stochastic sampling investigation

An experimental XSum path supports temperature 1, top-k 0 and top-p 1, with
optional CFG. It uses request-local SplitMix64 words and the existing CUDA
inverse-CDF kernel. This targets the same unfiltered categorical distribution
as pinned Python, but does not reproduce PyTorch's random-number stream. Other
temperature, top-k and top-p settings fail by absence of an implementation.

On eight real prefixes, an independent Python float64 inverse-CDF reference
used exactly the same supplied words and pinned Python model logits. Full
generation at seed 42 had 36/1,600 code differences and 18/384 output-ID
differences at CFG 0; at CFG 1 it had 1/1,600 code difference and 2/384 output
ID differences. **These controlled full-model comparisons do not pass exact
decision parity.** Subsequent levels can diverge after an initial code choice.

To separate model and sampler effects, the existing CUDA sampler consumed the
captured Python logits and identical words at every level: all 1,600 sampled
codes matched the independent float64 reference. With the Python-selected codes
forced into the Rust generator, two initial CFG-0 divergence positions remained
different when sampled on Rust versus Python logits. For row 4, level 5,
position 25, the maximum absolute logit difference was 0.0001221 and the
choices were 2782/2786; for row 3, level 8, position 36, the maximum was
0.0001526 and choices were 2729/2728. The Python cumulative-probability
distances to the nearest decision boundary were about 1.55e-7 and 7.11e-8,
respectively. This identifies boundary sensitivity for those positions, not a
general bound or proof of equivalent output quality. No tolerance was changed.

A follow-up forced the Python-selected history and compared all 1,600
conditional head distributions on those eight rows. All 6,553,600 logits were
inside the unchanged `atol=5e-4, rtol=2e-4` diagnostic band; maximum absolute
logit difference was 0.0003357. For each row, FP64 softmax probabilities were
compared using total variation distance (half the L1 distance). The maximum was
about `5.4694e-6` and the mean across positions about `6.4439e-7`. Two controlled
draws chose different codes, at the positions described above. This compares
distributions under identical prior codes; it does not prove behavior after
generated histories diverge or qualify stochastic quality. The diagnostic JSON
is `reports/phase4-xsum-stochastic-fulltrace-cfg0-v1/distribution-probe-v1.json`
(SHA-256 `fc3a04a7e8f55f6797e0a965f15ce1c22a8788a639946a76f081a2a358f3df54`).

The trace, CLI output and diagnostic results are under
`reports/phase4-xsum-stochastic-reference-cfg{0,1}-v1/` and
`reports/phase4-xsum-stochastic-fulltrace-cfg0-v1/`. The same-input sampler
report hashes to
`d1ea3cb062bb46d4d175458fb08493998b5f7edfd47736c573edd652aa4fc749` and the
forced-code report to
`e533ab309891a7b0c10c06bc96431019b0f4c8789efcf9cdf159289255992d0b`.
The Python reference verifies the pinned source and checkpoint hashes; these
diagnostic runs did not fingerprint native libraries or measure workstation
activity. Full positive-CFG quality and stochastic quality remain unqualified.
The official XSum configuration remains temperature 1, top-k 1, top-p 0 and
CFG 0; its earlier full quality and repeated performance results are separate.

A short stochastic performance screen used two warmups and five measured
generations in one process per runtime and batch, with seed 42. Python resets
its seed before each trial; Rust uses request-local SplitMix64. Their sampled
outputs differ because their random streams differ. At batches 1/4, Rust median
times were 155.02/557.60 ms versus Python 188.40/681.69 ms; whole-process peak
RSS was 1064.70/1062.57 MiB versus 1846.00/1846.75 MiB. This is an exploratory
screen without activity telemetry, not repeated acceptance or a quality result.
Raw files and source/binary/input fingerprints are in
`reports/phase4-xsum-stochastic-performance-v1/` and matched after collection.
The native screen repeated the same seed on each trial and rejected changing
outputs; a separate seed-43 run changed 1,524/1,600 codes.

The modified top-1 path matched all Python codes and IDs on the existing
256-row development fixture. A full source-bound quality rerun after adding
stochastic sampling remains pending. The full `tools/check.sh --cuda` passed,
including 18 device operator cases, in
`reports/phase4-xsum-stochastic-check-v3.log`. An earlier run failed while
building two new diagnostics without their cuDNN feature declarations; the
next run hit a process-observer test failure, which passed in isolation before
the successful full rerun. Both failed logs are retained.

The reference command uses `tools/reference/xsum_cfg_trace.py` with the pinned
`--upstream` and `--source`, a fresh `--output`, the eight-prefix JSON from the
[guidance investigation](cuda-cfg-real-v1.md), `--controlled-seed 42`, and
`--cfg-scale 0` or `1`. Add `--decisions-only` to omit intermediate logits.
`ncm sample-xsum ARTIFACT cuda-cudnn PREFIX_JSON SCALE 42` produces the native
seeded sample. `xsum_categorical_probe` reads the full CFG-0 trace, and
`xsum_stochastic_model_probe ARTIFACT TRACE` supplies the captured codes to
the forced-code model. The short timing screen uses native
`xsum_perf_screen ARTIFACT PREFIX_JSON BATCH device 5 0 42` and Python
`xsum_perf_screen.py --batch BATCH --no-stage-sync --collect-garbage --trials 5
--stochastic-seed 42` with the pinned inputs, upstream and source paths. Use
`OMP_NUM_THREADS=1`, `/usr/bin/time -v` and the local cuDNN library path for
native tools.
