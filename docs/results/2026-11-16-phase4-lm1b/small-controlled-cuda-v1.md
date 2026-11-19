# Experimental LM1B-small controlled stochastic CUDA sampling

The experimental sampler draws initial codes from the pinned level-zero
unigram and subsequent codes from the full 8,192-way distribution with the
configured temperature 1, top-k 0, top-p 1 and CFG 0. It uses explicit CPU
FP64 inverse-CDF selection and a request-local SplitMix64 stream; it does not
reproduce PyTorch's RNG or its floating-point sampler. The sampled-output API
returns codes and CPU penalty-selected IDs. Traced and untraced outputs match
on the two inspected rows. This CPU implementation is not a performance claim
or production qualification.

Two fresh pinned upstream CUDA captures use seed 11 and the same independently
implemented controlled stream and inverse-CDF hook. They reproduce byte for
byte: report SHA-256
`aa2bd1f8db4da63cfba04bbb857997a31855c79a63d55ce51f86a46b89b1b643`,
trace SHA-256
`173f1be79510d9fabad465652e748a43d5d71d447c20acaa581e4c1476689c64`.
The capture source SHA-256 is
`c38c6b2474141f722d089de5870c520c804735beafcebef9486ca4531c9877f3`.
The Rust CUDA trace reproduced byte-for-byte across four captures, SHA-256
`8d5184bc9ffeb0600c724d2454682cf3acf79e6488de4079fd8d4c162bb364e5`.

**The exact cross-runtime comparison fails:** 21 of 51 tensors fail the
unchanged policy after 116 generated-code differences and 42 penalty-selected
ID differences. The first difference is level 7, row 1, position 43. Its
uniform is 0.7007409624056891; the cumulative boundary after code 3212 is
0.700740463967027 on Python logits and 0.700741351477724 on Rust logits.
The upstream selects 3215 and Rust selects 3212, and later choices cascade.
An independent replay on the captured Rust logits reproduces all 1,282 Rust
choices, including the two initial codes. This supports a near-boundary
explanation for this first difference, not a general cross-runtime parity
guarantee. The failure report SHA-256 is
`74df032c70b805bcd780167b27f4cca4f3bf89bf7504a2fea12922f69077c7c7`;
the replay audit SHA-256 is
`557a5bffcc1e61492f38240303e3961027ae7a92e3de7f43e89ad787c724270d`.
No tolerance was changed.

Commands (repeat the first with a new v2 directory):

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/lm1b_small_controlled_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/lm1b-phase4 --prefixes reports/phase4-lm1b-small-cuda-development-v1.json --output reports/phase4-lm1b-small-controlled-cuda-reference-v1
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_controlled_trace -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-controlled-cuda-reference-v1 reports/phase4-lm1b-small-controlled-rust-v4
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_lm1b_small_controlled_trace.py reports/phase4-lm1b-small-controlled-cuda-reference-v1/trace.safetensors reports/phase4-lm1b-small-controlled-rust-v1/traces.safetensors reports/phase4-lm1b-small-controlled-comparison-v1.json
.reference/venv/bin/python tools/reference/lm1b_small_controlled_audit.py reports/phase4-lm1b-small-controlled-cuda-reference-v1/trace.safetensors reports/phase4-lm1b-small-controlled-rust-v1/traces.safetensors reports/phase4-lm1b-small-controlled-audit-v1.json
RUSTC_WRAPPER= cargo test --locked -p ncm-model --all-features
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda > reports/phase4-lm1b-controlled-cuda-check-v1.log 2>&1
```

Focused RNG/invalid-input tests, all-feature model tests/lint and full CUDA
regression (18 device cases) pass. The initial lint found a style issue, which
was corrected. This two-row diagnostic
does not establish PyTorch-seed equivalence, stochastic quality, held-out
metrics, performance, Metal, or Phase 4 completion. Logs and traces are ignored
local artifacts.


## Reference-order operator screen

An explicit standalone example uses the existing CUDA reference-order
layer-normalization, RMS/rotary and GELU backend for the same controlled
histories. Feature-specific Clippy and release build pass; the trace contains
51 tensors and its traced/untraced outputs match. The unchanged comparator
still fails 21 tensors. Its first generated-code difference remains row 1,
position 216; the candidate has 83 total code and 31 penalized-ID differences
versus the pinned reference, compared with 116 and 42 in the earlier CUDA
trace. Later differences cascade, so those totals do not establish an
improvement in the underlying cause. This bounded exploratory result does
not qualify sampling, quality, performance or a runtime gate; no default or
tolerance changed.

The new trace and comparison SHA-256 values are
`f00764e9dacfdbf5990ebe9b7e889c09f6e7e55618d85d24239c52a008bce641` /
`39a4aefe1140b27a6a79dc3923dbe2b9f9ad8c049a9568b5e6f9090f5c130839`.
Run the explicit example with the same artifact and reference arguments as the
original above; its binary name is `lm1b_small_controlled_reference_ops`.
The comparator exits 1 for this retained failure. Logs and outputs remain in
`reports/phase4-lm1b-reference-ops-*` and
`reports/phase4-lm1b-small-controlled-reference-ops-*`.


## Level-seven same-input continuation

A bounded pinned-Python CUDA replay loaded the unchanged upstream model and
reconstructed the level-seven input from each retained capture's latent and
post-convolution output. With reference inputs it reproduced all 1,048,576
captured logits exactly. The reference CDF boundary at row 1, position 43
is `0.700740463967027`, below the draw `0.7007409624056891`. With either
the original Rust or reference-operator Rust inputs, the same Python CUDA
continuation produces boundary `0.7007403004565222`, also below the draw and
therefore selecting the reference code 3215. Recomputing the convolution in
Python from the original Rust latent yields the same logits as using its
captured convolution output. Thus the retained Rust input differences alone
do not cause this first crossing under the Python continuation; differences in
the subsequent target projection, transformer or head can independently do
so. This is one controlled history and does not isolate a unique operation or
establish a fix. The ignored script/log SHA-256 values are
`fa658b78611eec06ad66ee3c318f0cc22029c1e361b60a1ae7335a67521c5806` /
`bf3d32c75147e4cf822515225270cf282de12643fa8a81c520e7c83f5d65b6bc`.

## Level-seven internals and QKV algorithm diagnostic

An explicitly selected level-seven trace captures the target and transformer
internals without changing the normal model path. Its traced and untraced
samples agree, and all 51 tensors shared with the earlier reference-operator
capture are bit-identical. The 73-tensor capture SHA-256 is
`2282dd3424849b899fc534067798d390027d45d2df2bd0d0f3a6ef6a990273da`.
A pinned Python CUDA replay on the captured Rust target reproduces the first
block input and first normalization exactly, but its QKV projection differs in
258,466 of 299,520 values (maximum `2.0265579e-6`). The retained replay script,
log and tensors have SHA-256 values
`976024b6030cf2671bb2815c455a19c3669b991198ae7cfc5086f3d7e6e2f1a5`,
`591bc38f6b88e1b2c50113b627bebb4233e66e4b3ea2492b56343f5b24d759d0`, and
`95ee8dd551f756cc38e2ab6165a66a5280e4c6797558b8e7dced8091521fcee5`.

For this exact captured input, a cuBLASLt heuristic screen found that a 1 MiB
workspace preference selected an algorithm reproducing the Python QKV exactly;
the existing 4 MiB preference selected the existing Rust result. The same
screen was observed using the system and Python cuBLASLt libraries; changing the
library alone left the original Rust trace identical. The screen uses one shape
and input, so it does not establish portable numerical equivalence. The system
screen script, log and JSON hashes are
`99de09141bcba96b6796eeb435ca2197f3c028c29cd1354d51dadb704f36fff1`,
`95bb0aceacd4f59cc7340c1febab7ae5391e67bb11d30c08bcb6884e352e070d`, and
`222eceade2c1340e51142d870a14484626d456ec71ad4d7af9697b353333007c`.

An explicit experimental backend applies that preference only to the 2304 by
768 projection, retaining the existing reference-operator diagnostics. On the
same two-row trace its first QKV exactly matches Python with the same input,
but its first attention already differs in 83,537 of 99,840 values (maximum
`6.6757202e-6`). The complete controlled comparison still fails 21/51 tensors,
with 116 code and 42 penalized-ID differences. This is not a fix or a gate pass;
no default, tolerance or sampling policy changed. The 73-tensor trace, filtered
51-tensor trace and comparison hashes are respectively
`2bfc05cef275512aa8c5d2f86e30c43a51ca9f297d80d88fa6e9bb7ad76b6cec`,
`45369293e0aedca0a7704872c9125b436cf220b5e1064241558981dddedcd0d6`, and
`006af26f421205870ca56c1992ddda56116eebb823efaaaa4c8d53041aa9af22`.
The trace example is `lm1b_small_controlled_level7_trace`; setting
`NCM_LM1B_QKV_DIAGNOSTIC=1` selects the experimental backend. Reports and
scripts are retained under `reports/phase4-lm1b-level7-replay-v1` and
`reports/phase4-lm1b-qkv-candidate-v1`.

A further pinned-Python probe captured Q, K, V and mask from the same Python
continuation. Splitting the scale between Q and K before multiplication
reproduced all of Python's math-SDPA attention values exactly; scaling after
the dot product differed in 75,925/99,840 values, maximum `5.9604645e-7`.
Neither reproduces the captured Rust output, and the probe does not establish
whether independently computed Rust Q/K are identical. The ignored script and
log hashes are `8fae89e7dfcccb23051a2dd15e04be9415fc944aae92c22af7f5c74015c13e45`
and `0063cc6955175731fe69d01255aaa38a9ec504260053a3c2a98d9ddc58ddce3b`.

Validation: the diagnostic example passed feature-specific clippy with
`lm1b-experimental,cuda-cudnn`; `setpriv --inh-caps=-all env RUSTC_WRAPPER=
./tools/check.sh --cuda` passed, including 18 real-device operator cases. The
full log is `reports/phase4-lm1b-qkv-diagnostic-cuda-check-v1.log`.

A temporary candidate that also scaled Q and K before the dot product still
failed 21/51 tensors, with 105 code and 31 penalized-ID differences. Its first
attention differed from Python in 81,448/99,840 values, maximum `6.91414e-6`.
The exploratory edit was reverted; its retained patch, 73-tensor trace,
filtered trace and comparison have SHA-256 values
`62d043baa6083031b43567afc52554b4e6202c56a6e5018ba76f82569cbf26cf`,
`36c9affcef311a8a90383552846a561a9b2a78c5ff3fc474ef70b042d9009ad6`,
`00844c5563d02269e449408b7f65f0773bba834729ec1561636e6405466f4325`, and
`6785686340c75d46706025ef9830f18abd3e440b4fdeccae62d342d26fcd998b`.

The expanded diagnostic captures normalized and rotated Q/K, V, and the rotary
tables at level seven. On the same QKV input, normalized Q and K and V exactly
match the pinned Python continuation, but 274 cosine and 496 sine table values
differ (maximum `3.81470e-6` and `3.78490e-6`); rotated Q and K differ in
13,783 and 13,775 values, respectively. Supplying Rust's rotated Q/K/V to Python
math-SDPA still differs from Rust attention in 75,652 values, maximum
`6.55651e-7`, so attention arithmetic is a separate discrepancy in this bounded
comparison. The 80-tensor Rust trace and Python comparison script/log hashes are
`1c56a2dd8e47b387f3524493f82126d8afb7478142ce5242d77a3d2dd9ed7ca6`,
`e701ecec44e64a58dc3bbbbbefa870be4a63e883b5bfd5ea123feaf8c016a13c`, and
`ba753d254f9b0a9fbf45446ca5a7eb351bd14e0df3cfeabfee77c2bb5a86bb53`.
The trace checks that its sampled output agrees with untraced execution. These
are bounded diagnostics, not proof that either discrepancy alone causes the
sample crossing or a Phase 4 pass.

A same-input intervention applies Python rotary arithmetic to the exact
normalized Q/K using the captured Rust tables; it reproduces both rotated tensors
exactly. This identifies the table values as sufficient to explain this
controlled rotary difference. The script and log hashes are
`ff36fcd77ce0f42b3c0c20682d52a1d0c56120ddbbd7c7201876e1657efc048f` and
`21ea4ea24a671d73f64a53dfd5399aaa3fdcb2976d17a17e1cef7c873babb37d`.
The 80-tensor capture also reproduces all 73 tensors from the earlier candidate.
Feature-specific clippy passed, as did `tools/check.sh --cuda` including 18
device cases (`reports/phase4-lm1b-rotary-trace-cuda-check-v1.log`).

## Explicit reference-table intervention

An opt-in diagnostic supplies upstream-computed, nonpersistent rotary tables to
the existing reference-operator and 1 MiB QKV candidate. On the retained two-row
controlled trace, **all 51 tensors pass the unchanged comparison**, and all
1,282 generated codes and all penalized IDs match exactly. The capture is
experimental and does not modify the canonical artifact, its schema, or normal
inference. This is a bounded result for this combination of interventions; it
does not prove the table change alone causes the pass, general sampling parity,
held-out quality, performance or the CUDA/Metal profile gates.

The offline export uses the pinned upstream function, rejects a changed source,
and reproduces the same table file twice (SHA-256
`9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc`):

```sh
.reference/venv/bin/python tools/reference/lm1b_reference_rope_diagnostic.py --upstream .reference/upstream-ncm --output reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors
setpriv --inh-caps=-all env NCM_LM1B_QKV_DIAGNOSTIC=1 NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC="$PWD/reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors" RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_controlled_level7_trace -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-controlled-cuda-reference-v1 reports/phase4-lm1b-reference-rope-candidate-v1
```

The 80-tensor trace, filtered 51-tensor comparable payload and comparison report
have SHA-256 values
`2301837423f1b834cde4b52096107020ae4eca81154fb8374d1604515870c48a`,
`8223711230e1efff8bdaf2fbcec6e256ff8112a74771d743c046bec5e460bb90`, and
`25c78750ac947048b7e79378aff4ef4d7d3fde5db7bd7eb028a9ff4d2b4e94e6`.
The diagnostic validates table dimensions and finiteness; a targeted unit test
and diagnostic feature clippy pass. `tools/check.sh --cuda` passed including
18 real-device cases (`reports/phase4-lm1b-reference-rope-cuda-check-v1.log`).
The final-source combined capture is byte-identical to the retained candidate.
With the same reference tables but without the 1 MiB QKV intervention, 21/51
tensors fail, with 103 code and 31 penalized-ID differences; this does not
establish general necessity beyond this case. Its trace, filtered trace and
comparison hashes are
`89aadd61606a46d945055bcce70ccf8235aafa015d274f210338cbd288345ded`,
`78e60d3fff128a2944d50e1c73d35bb6d0c33d5df5973b0348306410bd60f525`, and
`6043c4bef55d0cad0b65f93f6106ed72c83e699b5de5ada17a2c01383be30989`.
Broader quality and performance evidence remain pending.

## Non-held-out candidate development screen

The same explicit QKV/reference-table candidate generated 256 samples in
batch-eight requests for each development seed 11–15, using the existing
request-local SplitMix64 streams. The pinned FLM evaluator matched its upstream
implementation on all five captures. These are descriptive development results,
not the held-out quality gate or proof of general stochastic parity:

| Seed | Candidate perplexity | Candidate raw-ID entropy |
| --- | ---: | ---: |
| 11 | 93.261720 | 4.311424 |
| 12 | 95.223602 | 4.305927 |
| 13 | 92.006153 | 4.312571 |
| 14 | 91.207176 | 4.311991 |
| 15 | 92.054570 | 4.315229 |

The five-seed candidate mean is 92.750644 perplexity and 4.311428 entropy;
the retained Python development means are 92.975852 and 4.311537. Candidate
minus Python is -0.225208 and -0.000109. Independent 10,000-resample seed
bootstrap percentile intervals are [-2.872697, 2.100387] and
[-0.003445, 0.002876], respectively; five seeds limit interpretation. Capture
and scorer identities are checked by the retained summary script
`reports/phase4-lm1b-level7-replay-v1/development-summary.py`, SHA-256
`fe7c9be5327cac7b5477e02c1068e77308424cc242ee83a241cfa225d2c24b9c`, and the
summary SHA-256 is
`f6497e6ef6341e2181f02fc0e1b17870f907b719ff423a2d569cd0c61397c192`.
The candidate's seed-11 capture differs from the earlier default-backend
capture in 12,086 codes across 98 rows and 3,362 IDs across 93 rows; existing
scores for that backend cannot be reused as candidate evidence. Each capture
records the diagnostic backend and table hash. The feature-specific capture
lint and seed-guard test pass; the seed-11 capture and its pinned scorer report repeat byte-for-byte.
`tools/check.sh --cuda` passed including 18 real-device cases
(`reports/phase4-lm1b-reference-rope-development-cuda-check-v1.log`).

## Candidate device sampling and capacity screen

The candidate's explicit CUDA inverse-CDF and FP64 penalty path exactly matches
all CPU-sampled candidate codes and raw IDs across five development seeds,
1,280 samples. The hash-checked comparison report is
`reports/phase4-lm1b-reference-rope-device-compare-v1.json`, SHA-256
`1848e4c6520ba9a0e455684973d8e22e4238a00d9733c6a310980807b97846d2`.
Therefore the candidate's five development scores apply to those exact device
outputs. The comparison does not establish held-out quality or general parity.

A source- and rotary-table-bound, non-held-out paired screen compared the
CUDA-sampling/penalty candidate and pinned Python in three fresh processes per
runtime, with two warmups and eight measured batches per process. The timing
includes initial sampling, generation, decoding, penalties, and code/ID host
readback; level-zero sampling remains on the CPU and model loading is excluded. Signatures repeat within each runtime,
not across their different random streams. The 24 pooled timings per cell are:

| Batch | Python median / p95 | Candidate median / p95 | Python / candidate median process peak RSS |
| ---: | ---: | ---: | ---: |
| 1 | 0.061774 / 0.066694 s | 0.045426 / 0.049354 s | 2,665,136 / 1,635,532 KiB |
| 4 | 0.133072 / 0.133971 s | 0.108290 / 0.109904 s | 2,657,720 / 1,626,260 KiB |
| 16 | 0.427715 / 0.428621 s | 0.362949 / 0.364626 s | 2,635,680 / 1,676,952 KiB |

The source-bound result hash is
`e4225ad3039afd41fb08eba039c82f7b554f4d2fe173ee765c0a2da4eacc4a9e`;
manifest, per-process output and sampled GPU activity are retained under
`reports/phase4-lm1b-reference-rope-capacity-screen-v1`. Other desktop GPU
processes were present, and sampled driver memory is not an allocator peak.
This is an exploratory screen, not the release performance/memory gate; it
does not replace held-out quality, broader numerical parity or Metal evidence.
The exact candidate table and source are hashed in the manifest. Feature-specific
lint, device-capture seed guard, capacity-summary tests, and the hash-checked
five-seed comparator pass. Full `tools/check.sh --cuda` passed including
18 real-device operator cases
(`reports/phase4-lm1b-reference-rope-device-capacity-cuda-check-v1.log`).

## Intrinsic reference-order rotary candidate

The experimental LM1B-small model computes FP32 inverse frequencies first
and then multiplies by position, as the pinned upstream does. This replaces
per-position division in this model only and needs no supplied table; shared
geometry for other profiles remains unchanged. On this Linux host the generated
128-by-32 tables differ from the pinned CPU PyTorch tables in 232 cosine and
132 sine values, at most `5.960464477539063e-8` absolute. They are not bit
identical and cross-platform math-library behavior remains unqualified. The
new unit control checks selected pinned values to `1e-7`; that is a local
regression check, not a changed acceptance tolerance.

With the existing 1 MiB QKV diagnostic, the retained two-row controlled CUDA
capture passes all 51 tensors under the unchanged comparison policy. The full
80-tensor capture SHA-256 is
`59a1473b4eefb7efa8cd9f1c198f10a0e01a3601a7abe3131d0865080daf666c`;
the filtered payload and comparison are respectively
`3e93506075374225a1480a1a199c8355502a9a0ce6ce049112dbe808fc8b0016` and
`9266b4ac180e79460930268f77963250c238a0aea06178462ffaaf11d3f80bb7`.
The model source SHA-256 is
`ffc3088326542ace96b6306af0273738a1eab81bf0f4e186b39a40998bf858f8`.

The explicitly selected QKV candidate generated and scored the existing five
non-held-out development seeds with the pinned upstream scorer; all scorer
cross-checks matched upstream. Each seed has 256 samples, batch eight. Changes
below are row counts against the earlier external-table device candidate:

| Seed | Perplexity | Mean entropy | Code rows changed | Raw-ID rows changed |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 93.267423 | 4.311027279 | 85 | 81 |
| 12 | 94.741863 | 4.307672182 | 105 | 99 |
| 13 | 91.081261 | 4.310955182 | 96 | 94 |
| 14 | 90.740706 | 4.312227994 | 103 | 100 |
| 15 | 91.875097 | 4.311800897 | 101 | 97 |

Descriptively, candidate minus Python development means are `-0.634582`
perplexity and `-0.000800239` entropy, with independent seed-bootstrap
intervals `[-3.337282, 1.746404]` and `[-0.003185621, 0.001394795]`.
These observed development seeds and the earlier controlled trace are not a
held-out quality pass. Since generated outputs differ, the external-table
candidate's failed held-out result and its performance measurements cannot be
transferred to this candidate. The frozen LM1B quality gate remains open.

The isolated candidate passed the focused Rust unit test, feature-specific
Clippy and full `tools/check.sh --cuda`, including 524 report tests and 18
real-device operator cases. The first full check failed four report tests
because its fresh isolated checkout had no `reports/` directory; creating the
empty directory and rerunning produced the passing result. Both logs, the
capture/score files, controlled trace, comparison and diagnostic source are
retained under `reports/phase4-lm1b-intrinsic-rope-v1/`. Linux/CUDA quality,
performance and memory gates, then Metal checks, remain required. No profile or
Phase 4 gate is claimed by this increment.

The controlled trace used the checked-in
`lm1b_small_controlled_level7_trace` example with
`NCM_LM1B_QKV_DIAGNOSTIC=1` and no reference-table override. Development
generation used the retained diagnostic source
`ncm-lm1b-development-rope-capture.rs` (SHA-256
`e5ab4026e84cab143edcbd785741cd94c4afb13d806ae3c4744fdae136f1cc89`),
`NCM_LM1B_INTRINSIC_ROPE_DIAGNOSTIC=1`, seeds 11–15 and `device-penalty`.
Each output was scored by `tools/reference/lm1b_quality_probe.py` with its
capture SHA-256 and the existing pinned evaluator, BERT and upstream inputs.
The passing full-check log SHA-256 is
`5fd8ab560d9eb912dece45f845f2dfe0cfca2387948a56b8ead6613187a678da`;
the initial failure log SHA-256 is
`ac0cba88433c7402719cdfba0616db9221a194bb2117db690ae059180aa2d436`.

A fresh non-held-out capacity screen alternated three Rust and three pinned
Python processes per batch, with two warmups and eight measured batches per
process. Timings include generation and code/ID readback; model load is
excluded. Median process peak RSS is from the three process high-water marks.

| Batch | Python median / p95 | Rust median / p95 | Python / Rust peak RSS median |
| ---: | ---: | ---: | ---: |
| 1 | 0.062264 / 0.065574 s | 0.045266 / 0.049245 s | 2,608,524 / 1,575,312 KiB |
| 4 | 0.133354 / 0.134002 s | 0.108781 / 0.110817 s | 2,601,736 / 1,564,928 KiB |
| 16 | 0.427370 / 0.428287 s | 0.363390 / 0.364022 s | 2,578,916 / 1,614,948 KiB |

Result SHA-256 is
`7f6df0c5595e5e6cc9da8824473368ab733e8d251f559a97c6ad36cf8af3f085`;
source-bound process records, sampled GPU activity, and exact diagnostic
sources are retained in the same evidence directory. Desktop GPU processes
were present. Sampled device memory is not an allocator peak, and eight
measurements per process do not meet the full performance gate. This screen
does not establish held-out quality or Metal support.

## Second controlled history and isolated arithmetic candidates

A separate two-row controlled draw with seed 16 exercises the current
intrinsic-rotary, 1 MiB QKV candidate. This is a numerical diagnostic, not a
quality sample. Two pinned Python CUDA captures reproduced byte-for-byte;
their trace SHA-256 is
`9771b1d59dacf941e572096b471f3de5b13d12d08e7312245a485967db745dc9`.
The current candidate fails 7 of 51 tensors under the unchanged comparison,
with two code differences and no penalized-ID differences. The first is row 1,
level 10 position 60: reference code 5429 and Rust code 5428 straddle draw
`0.8566738954978064`, with cumulative boundaries `0.8566738952863275`
and `0.8566739015165296`. Same-logit replay reproduces all 1,282 Rust choices.
Supplying the exact pinned rotary table still gives two code differences and
no ID differences. Comparison and replay audit SHA-256 values are
`d97266519878d98f879c7d9f5e151bfa5fc90d8ca9fcbe063690bb85fd96d49b` and
`357523a98023183e7670d925caf7c9c7f1347eb84c725c2006eaec34c5cc849b`.

A pinned Python CUDA continuation from reference level-10 inputs reproduces
all reference logits exactly. Replacing either its latent or its convolution
output with the corresponding Rust tensor independently selects 5428 at that
draw. Reconstruction first differs at level 1. On the same captured level-one
embedding, an independently written CUDA fused multiply-add interpolation
matches all 98,304 PyTorch values, whereas the existing separate operations
differ in 19,340. It also matches on one synthetic input for each of the ten
LM1B interpolation lengths. This is bounded operator evidence.

An isolated backend candidate combines that interpolation with a 1 MiB
cuBLASLt heuristic for the 384-to-768 input projection. On the captured
level-seven target input the heuristic matches all Python target values; the
previous 4 MiB choice differs in 32,949. A further explicit area-reduction
variant multiplies the sum by an FP32 reciprocal. On the captured 128-to-108
input that matches Python exactly; division differs in 4,059 values. Together
the three changes make reconstruction levels 0–9 and level-10 latent,
convolution, target, first normalization and QKV bit-identical in this one
history, but the same two code differences remain.

At that level, the intrinsic tables differ from Python in 194 cosine and 116
sine values, each by at most one FP32 ULP; normalized Q/K and V match, but
rotated Q/K and attention do not. A separately isolated split-scale attention
variant reduces the first-attention mismatch count but still gives two code
differences, with zero ID differences using intrinsic tables and one ID
difference using the exact table. The interpolation-only candidate had 98 code
and 30 ID differences, despite exact reconstruction through level 6. These
results show interacting rounding effects, not a qualified fix. No default,
artifact, numerical policy, or quality gate changed.

The final isolated source snapshot SHA-256 is
`6af7f2a86c687d05df0f31d819d6a906121e1dd833678eacde838b9081870d95`.
It, the controlled traces, intervention outputs, probe sources and receipts
are retained under `reports/phase4-lm1b-controlled-seed16-v1/`. The final
candidate passed feature-specific Clippy only; full source/device checks,
quality, performance and Metal qualification have not run for it. It is not
integrated into the working tree.

## Combined CUDA arithmetic diagnostic and observed-seed retest

An isolated candidate combined the earlier interpolation, area-reduction and
projection probes with separately scaled Q and K, a bounded warp softmax, and
separate shared-head projection and bias addition. The latter follows the
pinned upstream operation order. With the exact pinned Python rotary table,
both the original seed-11 and second seed-16 controlled histories pass all 51
tensors under the unchanged comparison. In a same-input seed-16 attention
probe, the earlier candidate's scaled Q, K and masked scores matched PyTorch,
but 69,423 softmax probabilities differed; the isolated warp probe matched the
captured inputs and one synthetic input at each supported sequence length.
These finite probes do not establish general operator equivalence.

A separate experiment embeds the fixed LM1B table generated from the retained
Python table (input SHA-256
`9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc`;
FP32 table SHA-256
`249b3e4a363941c9ddcedf832597529610924dd907b3c9e80453af97f8bff954`).
Without the runtime table override, its seed-11 and seed-16 captures are
byte-identical to the corresponding exact-table captures, SHA-256
`473d457f921454f17d4dacda74471dc55ccec839715954b7d6df8509fed1de56` and
`a3187ea946d56c7a010ac332629b9855255dbc60acddf123763b9b1e7b6f5db4`.
The table is an unintegrated experimental asset; portability and provenance
suitability for distribution have not been established. The intrinsic table
still yields two code differences in the second controlled history.

Five non-held-out development captures of 256 samples each scored exactly
against the pinned upstream evaluator. The independent seed-bootstrap
candidate-minus-Python intervals were `[-3.265117, 1.845462]` perplexity and
`[-0.003810, 0.002015]` entropy. These inspected data are descriptive only.
The same candidate was then retested against the five **already observed**
reference seeds with the original scoring and margins. The result fails the
entropy interval: `[-0.006477315, 0.002497267]` versus `[-0.0046, 0.0046]`;
the perplexity interval is `[-4.821202, -0.077211]` versus `[-5.46, 5.46]`.
All five scorer cross-checks matched upstream. This retest is not fresh
held-out evidence, and neither result qualifies the candidate.

The experimental source snapshot, controls, scores and logs are retained in
`reports/phase4-lm1b-combined-candidate-v1/`. The source archive SHA-256 is
`ca15b717903886b535d99fd6cceada7636f62926baff4b91379ecdb0b3d6aff6`;
the retest summary SHA-256 is
`9efd1dd2ccfea6add003204567e3b0a832eefd63cf012bf1527bfb79d42866e0`.
Feature-specific Clippy passed for the isolated candidate. Full regression,
operator edge-case and device-architecture checks, performance, and Metal
qualification have not been run for it. No candidate code, model artifact,
acceptance threshold or quality protocol was changed in the repository.

## Shared-head projection and bias order

The pinned upstream `_SharedHeadView` computes `F.linear(x, weight)` and then
adds its level-specific bias. The experimental LM1B-small teacher now follows
that order explicitly, rather than calling a backend operation that may fuse
the bias into its matrix multiplication. This is a correctness change to the
experimental path, not evidence that LM1B-small is qualified.

On the retained level-10 seed-16 hidden state, pinned PyTorch's separate
projection and bias reproduce all 1,769,472 recorded logits exactly. Its
biased-linear call differs from that separate expression in 634,235 values,
maximum absolute difference `7.62939453125e-6`. On the same input, the Rust
CUDA backend also matches all recorded logits with separate operations and
differs in 634,235 values with its fused operation. This is one same-input
control, not a universal equivalence claim. The level-seven controlled trace
example now independently recomputes its captured logits from the actual
hidden state with separate projection and bias. The changed model passes;
the prior model fails this check. The changed model also passes all 51
reference tensors in the existing two-row controlled comparison, with no
change to its policy (comparison SHA-256
`236dd8ea992b29aa16293af87b86d33179885818fe4700e5d062170a3c6961bb`).

The old-source baseline reproduced its earlier seed-11 development capture
byte-for-byte. The changed model generated and scored five already-inspected
development seeds, each with 256 samples; every pinned scorer cross-check
matched upstream. Relative to the old intrinsic candidate, 43, 52, 54, 52,
and 46 code rows changed across the five seeds. The descriptive
candidate-minus-Python bootstrap intervals were `[-2.825937, 1.938374]`
perplexity and `[-0.003458, 0.002539]` entropy; these are not held-out results.

On the five already observed quality seeds, all five scorer cross-checks
matched upstream, but the unchanged entropy interval still **fails**:
`[-0.007610838, 0.002300165]` versus `[-0.0046, 0.0046]`. The mean and
perplexity checks pass. The summary repeated byte-for-byte (SHA-256
`3e947d44a26a3bf80d7a1312dc37d39be082263f9a059fe918a23a9d2b19d4db`);
these are retests, not fresh held-out samples. The original failed result and
frozen protocol remain in force.

An alternating baseline-versus-candidate screen used three fresh processes
per variant and batch, two warmups and eight timed requests per process. The
candidate-to-baseline median ratios were `1.0066`, `1.0107` and `1.0130` for
batches 1, 4 and 16 respectively; corresponding p95 ratios were `0.9333`,
`1.0045` and `1.0073`. The screen measured process high-water RSS, but did
not monitor other GPU activity or allocator peaks. It is not the full
performance or memory gate. The old and changed binaries, model source and
results are identified by hashes in the retained diagnostic evidence under
`reports/phase4-lm1b-head-order-v1/`.

Focused `ncm-model` tests and LM1B-feature Clippy passed. Full Linux
`tools/check.sh --cuda` passed, including 524 report tests and 18 real-device
operator cases. The first full run failed while compiling because temporary
storage hit a quota; a disposable earlier debug build cache was removed and
the full suite reran successfully. Logs are retained with the evidence. The
first attempt to log the expected negative control used a read-only shell
variable and exited after the control ran; the corrected wrapper verified its
expected exit and message, and both control logs agree. No Metal check was run
for this change; the earlier XSum Metal gate precedes further profile
advancement.

## Same-logit sampling probability control

The configured LM1B sampler uses temperature 1, top-k 0 and top-p 1. The pinned
upstream still sorts logits, evaluates FP32 softmax, forms a cumulative sum and
masks entries whose preceding cumulative mass exceeds 1 before its final
softmax. The experimental Rust sampler instead uses the full distribution with
FP64 inverse-CDF arithmetic. Mathematically top-p 1 need not filter, but the
upstream FP32 computation can do so because of rounding.

A same-logit diagnostic invokes the pinned upstream `_sample_level_raw` on
captured logits while intercepting the probability tensor supplied to
`torch.multinomial`. It compares those FP32 weights, normalized in FP64, with
an FP64 softmax of the same logits. A separately reconstructed top-p mask and
final FP32 softmax match the intercepted upstream weights bit-for-bit. The
source checkout, source file, controlled RNG source and both traces are
hash-checked; no generation or oracle was changed.

| Captured history | Rows | Rows with masked entries | Masked entries | Maximum total variation | Mean total variation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned Python controlled history | 1,280 | 278 | 19,597 | `8.9811e-7` | `5.5802e-8` |
| Rust shared-head controlled history | 1,280 | 290 | 19,398 | `9.0838e-7` | `5.2475e-8` |

For the Rust history, the maximum total variation between unfiltered FP32 and
FP64 softmax was `3.5696e-8`; the maximum change caused by filtering relative
to unfiltered FP32 was `9.0838e-7`. Applying the retained midpoint uniforms to
the two compared distributions selected the same code in all 1,280 rows of
each trace. The FP64 calculation also reproduced all recorded controlled
choices. The two result files agree byte-for-byte.

This is a bounded distribution comparison on four previously inspected rows,
not a measurement of PyTorch's actual multinomial RNG, a proof of the Rust
CUDA sampler's precise rounding, or a quality gate. It does not establish that
the difference is absent on other histories or explain the observed quality
failure. Exact treatment of upstream FP32 filtering remains a numerical
compatibility gap; no sampler, protocol or tolerance was changed.

The hash-checked source and repeated results are retained under
`reports/phase4-lm1b-probability-evidence-v1/`. The final source SHA-256 is
`67485134c99548d0bf3607c485431885d2954797088a62b23dee455afb34ee2d`, and
the result SHA-256 is
`1f4a654317577d1e4444389b6b294015fca4ac763ce3fa1b623be15dda605e5e`.
The script passed Python syntax checking. Run it on the local CUDA device with
`setpriv --inh-caps=-all env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.reference/venv/bin/python reports/phase4-lm1b-probability-evidence-v1/ncm-lm1b-sampling-probability-v5.py`.

### Top-p cutoff ties and the existing CUDA sort

The two captured histories above each have a single mask transition on every
masked row, with no transition through equal sorted logits. That observation
is limited to those histories. To test ties separately, a deterministic
synthetic fixture rounds 1,024 normally distributed 8,192-way FP32 rows to
multiples of 1/8. The pinned upstream top-p path masks entries in five rows;
in all five, its cutoff splits equal logits. Its default and explicitly stable
sorts produce the same masked sets for these synthetic rows. The five selected
rows reproduce the full-batch PyTorch sort indices when sorted on their own.

On those same five synthetic rows, Candle's existing CUDA descending sort
produces numerically equal sorted values but 39,943 different sorted indices.
Applying the *upstream sorted-position mask* to Candle's order changes 76
original token-mask entries across all five rows. Holding the later FP32
softmax fixed, the largest distribution total variation caused by this mask
change is `1.3432e-8`. This deliberately isolates tie ordering: it is not a
Candle top-p implementation or a full-model result. It shows that simply
using the existing sort with the upstream mask positions is not exact on this
fixture; it does not prove a particular sort rule works generally.

An initial comparison incorrectly required identical float bit patterns and
failed on signed zeros. The corrected control confirms that 250 sorted
positions differ only by the sign bit of zero and that every sorted value is
numerically equal. The fixture, masks and result repeat byte-for-byte. The
retained diagnostic sources, binary, synthetic fixture and logs are under
`reports/phase4-lm1b-probability-evidence-v1/tie-order-v1/`. The fixture
SHA-256 is `6dc06555bba8ef1a780f16584b440c65cec5bcb0c7fee197af5b14b3b2e8506c`;
the signed-zero control log SHA-256 is
`dd3231a69f733f634198a1c65840cf2cf9afbc02ab914b66ef5034ecf4ae28a9`;
the fixed-softmax probability result SHA-256 is
`9aefcbdf50208114073c6ed5d2b7bf1a8cc9c10f46d9c2f1ff413d13c65558f3`.
Python syntax checks and the real-CUDA comparisons passed. No production
sampler or quality criterion was changed.

### Isolated FP32 top-p probability prototypes

An isolated CUDA experiment reconstructed the pinned PyTorch `top_p=1`
probability calculation, including sort, FP32 softmax and cumulative sum,
without changing the production sampler. On this RTX 3060 Ti, independently
implemented component probes reproduced all probability elements in each of
both 1,280-row captured histories, 1,024 synthetic quantized rows and five
cutoff-tie rows. This is evidence for those inputs and this software/hardware
combination, not a general cross-device equivalence guarantee. The scan and
softmax were developed from public PyTorch source; any production adaptation
requires a license and compatibility review. The experiment still draws with
NCM's SplitMix64 uniforms and FP64 inverse CDF, not PyTorch's multinomial RNG.

A native opt-in prototype matched pinned reference choices on 2,565 crafted
boundary rows, while the current sampler differed on 2,508; invalid-input and
single-row controls were also checked. The experimental implementation rejects
single-row inputs and has not received the production safety or full-suite
review. A fused variant reduced its scratch storage from five to
three full row-by-vocabulary buffers. It produced byte-identical outputs on
all five previously observed development seeds, and matching signatures across
the timing screen, but is not the same as a proof of probability equivalence
for every input.

Three alternating fresh processes per variant, with two warmups and eight
measured generations per process, gave the following medians. This bounded
screen includes generation and readback, but is not the frozen performance
qualification and does not compare against a simultaneous Python run.

| Batch | Current sampler | First prototype | Fused prototype |
| ---: | ---: | ---: | ---: |
| 1 | 45.56 ms | 49.14 ms | 49.29 ms |
| 4 | 109.75 ms | 120.85 ms | 124.18 ms |
| 16 | 369.38 ms | 413.42 ms | 433.10 ms |

The fused prototype was also regenerated and scored on the five **already
observed** quality seeds, using the unchanged pinned evaluator and original
Python references. All five scores matched the upstream scoring functions.
Rust minus Python mean perplexity was `-1.962556`, interval
`[-4.363115, 0.446628]`; mean entropy was `-0.002433581`, interval
`[-0.007166052, 0.002060141]`. The entropy interval still fails the frozen
`±0.0046` margin. A deterministic repeat produced the identical summary.
These are retests, not fresh held-out evidence, and neither prototype is a
quality or runtime gate pass. The original failure remains in force.

Source archives, captures, failed attempts, timing telemetry and scores are
retained in ignored `reports/phase4-lm1b-top-p-native-diagnostic-v1/`. The
summary SHA-256 is
`b9e594bc7e78858a2d1c01ee1c9992cff4e244bf75f587e1800038229a43edeb`;
the timing result SHA-256 is
`366a3b17458a7d24c869fa0e8062d2de13ae0363c72a4f66c17795159fd82ba6`.
The first timing launch could not access the sandboxed GPU; the first scoring
launch omitted the existing tokenizer dependency path. Both failures and the
corrected runs are retained. The prototype remains outside production.

## Controlled generation at evaluator convolution precision

The two-row CUDA development trace uses the same prefixes, seed 11,
SplitMix64 draws, FP32 matmul and math SDPA as the strict control. The
optional Python capture selects the evaluator's convolution settings
(TF32 enabled, deterministic convolution disabled, benchmarking disabled).
The Rust comparison uses the explicit cuDNN default-math backend. Strict
and default-math Rust both use the pinned rotary tables and reference-order
QKV. The comparison uses the existing numerical bands and scores the
same 51 tensors.

| Python convolution | Rust CUDA convolution | Tensors outside policy | First differing code |
| --- | --- | ---: | --- |
| Strict | Strict | 0 | All match |
| Evaluator | Default math | 33 | Level 5, row 1, position 27 |
| Evaluator | Strict | 48 | Level 2, row 0, position 10 |
| Strict | Default math | 48 | Level 2, row 0, position 10 |

The Python strict and evaluator traces also differ on 48 tensors, with four
changed codes at level 2. Row and position indices are zero-based. When both
runtimes allow convolution TF32, the code histories agree through level 4.
The level-5 head remains within the existing band with all argmaxes equal.
The disputed uniform is `0.010848570498637855`. The cumulative probability
through index 280 is `0.010848581828234794` in Python and
`0.010848533593788423` in Rust, so the same uniform selects 280 and 281.

The pinned Python post-upsampling convolution exactly reproduces its own
captured output. On the Rust latent at level 5, the Python convolution
output is within `5.73e-6` of Rust and differs from the original Python
output by up to `2.93e-4`. Feeding this recomputed convolution and the Rust
latent through the pinned Python transformer still selects 281. The
Python latent and Python convolution exactly reproduce the original head
and select 280.

The isolated codec capture locates the first difference before the first transformer head.
The level-zero embedding and learned transposed convolution are exact in
both modes. Under strict math the residual convolution is also exact.
With evaluator precision, the residual convolution differs at 86,638 of
98,304 values on identical input, with a maximum difference of
`1.8310546875e-4`. This produces a maximum difference of
`9.1552734375e-5` in the initial reconstruction. On the same Rust input,
PyTorch area pooling reproduces Rust at levels 1, 2, 4 and 5 exactly. The
level-3 maximum difference is `2.9802322387695312e-8`.

A separate legacy-API control reproduced all 98,304 Rust residual
convolution values on the same input. Public cuDNN API logs show PyTorch's
evaluator mode calling `cudnnBackendExecute`, while the legacy control
calls `cudnnConvolutionForward`. In the full legacy-API trace, all sampled
codes match Rust, with 17 tensors still outside the existing comparison
policy. The evaluator uses the graph API. The legacy control is recorded
in the capture receipt and is rejected as an evaluator reference by the
comparison command.

Run from the repository root with the pinned reference environment and
native cuDNN library paths used for the earlier CUDA checks. Use a new
report directory for each run.

```sh
run=$(mktemp -d reports/phase4-lm1b-controlled-precision-v1.XXXXXX)
reference=reports/phase4-lm1b-small-controlled-cuda-reference-v1
artifact=artifacts/canonical/lm1b-small-inference-v1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
.reference/venv/bin/python tools/reference/lm1b_small_controlled_trace.py \
  --upstream .reference/upstream-ncm --source artifacts/upstream/lm1b-phase4 \
  --prefixes reports/phase4-lm1b-small-cuda-development-v1.json \
  --output "$run/python-default" --upstream-convolution-precision
cargo build --locked --release -p ncm-backend-cuda \
  --features cuda-cudnn,lm1b-experimental --example lm1b_small_controlled_level7_trace
export NCM_LM1B_QKV_DIAGNOSTIC=1
export NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC="$PWD/reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors"
trace=target/release/examples/lm1b_small_controlled_level7_trace
"$trace" "$artifact" "$reference" "$run/rust-strict"
"$trace" "$artifact" "$reference" "$run/rust-default" cudnn-default-math
.reference/venv/bin/python tools/reference/lm1b_controlled_precision_compare.py \
  "$reference/trace.safetensors" "$run/python-default" \
  "$run/rust-strict/traces.safetensors" "$run/rust-default/traces.safetensors" \
  "$run/comparison.json"
```

The final Rust strict and default-math trace SHA-256 values are
`f2773867eacc0b1afbfe0a45af1d7b67d65bee6555640cf3a7f58a40d098760a`
and `c680003fc8e1a2b82fc42cfa1dc8e740118df89597017fd95a2281da833d1f26`.
The Python evaluator trace SHA-256 is
`91240d007b39132a195400b6952b620f40e620af89da6ba80aef343a2737aea5`.
The evaluator-precision receipt SHA-256 is
`2417a8e885fe590b4b5f2719fb505c523d205d4db55300b3afcf9718bfe9789a`.
The comparison summary repeats byte for byte at
`a191c5e8fbb6bfd83773f9895bfe5b5a77ed2efd36101e22b60aa677bf2d879f`.
Raw captures, API logs and the isolated replay scripts are retained under
`reports/phase4-lm1b-controlled-precision-v1.Sk0iDB/`. The
`initial-codec-replay.json` SHA-256 is
`0f859fee0d8ed56c593cf1ea0ea1c0719dfce9e090cfeb2b83604ec8971ca9a4`.
The `level5-continuation.json` SHA-256 is
`ee182e4966152e8e0d3a90dd04fc33c436ed3c9017fcb4fe5147dbe329661313`.
The Linux host suite passed all 532 report tests. The actual CUDA example
passed release build, Clippy and repeated device captures. The pinned
strict Python tensors and both Rust captures repeated byte for byte.
These are development diagnostics. The frozen held-out entropy result
remains outside its acceptance interval.


### Graph convolution and interpolation replay

An isolated Rust probe built a single forward convolution through the
[native cuDNN graph API](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/api/cudnn-graph-library.html)
using the existing cudarc binding. On the RTX 3060 Ti with cuDNN 9.26, the
first plans returned by the instant and B heuristics both matched PyTorch
bit for bit in all 28 fixed-input cases. The matrix covers all 11 generator
post-upsampling widths at batch 2, five refinement weights, and repeated
input controls at batches 1, 4 and 16. The largest matching plan workspace
was 4,915,216 bytes. Repeating the matrix produced byte-identical results.

An isolated model build used the graph for the 384-channel, three-tap
forward convolutions, while retaining cuDNN default math for the other
convolutions. The level-zero reconstruction now matches all 98,304 values.
The first sampled code difference moves from level 5 to level 9 (row 0,
position 71), and 23 of 51 tensors remain outside the same policy. Its
comparison summary SHA-256 is
`544528d1fee19c7b664d05e0ce059bc699633acafa224d5f2f8fe426daec1d9f`.

The first remaining reconstruction difference is at level 1, with a
maximum difference of `9.685754776e-8`. Across the eight common subsequent
levels, the separately rounded interpolation reproduced the Rust
reconstruction exactly when replayed with the captured previous Rust state.
Rounding the left product and fusing the right product with the addition
reproduced every pinned PyTorch interpolated value across those eight levels.
The graph convolution can amplify these small interpolation differences
after TF32 rounding.

The scripts, raw cases and replay summaries are retained in the preceding
report directory under `graph-probe/`. The repeatable `matrix.log` SHA-256
is `2bcaf20a34b1cdb09ea65a334dad28676710391f83d36f4a29636453bd76b598`
and the `interpolation-replay.json` SHA-256 is
`74241392e97d0f9968eb84c710aba140a351324b4332c1f4b6b3ebeddf5e75d4`.

The graph plus fused interpolation reproduced all values in reconstruction
levels 0–5, but the first code difference moved to level 6 (row 0,
position 18), and 26 tensors failed. With identical level-5 reconstruction
inputs, PyTorch area pooling at the next level differed from Rust at 8,253
of 39,936 values by at most `1.192092896e-7`. Multiplying each bin sum by
its FP32 reciprocal reproduced every Python value. The same operation
also reproduced the Python 128-to-24 area values at level 3.

Combining graph forward convolution, fused linear interpolation and
reciprocal area pooling passed all 51 controlled tensors on both rows
under the unchanged policy. All 1,282 sampled codes and all 256 greedy
and 256 penalized IDs match. All values in the 12 reconstructions match
exactly. The maximum final-logit difference is `1.907348633e-5`.
The combined release build and feature-specific Clippy passed. A second
device capture was byte-identical, and the strict and legacy mode
controls both reproduced their earlier captures byte for byte.

| Retained graph candidate evidence | SHA-256 |
| --- | --- |
| `rust-graph-fma-area/traces.safetensors` | `2abf80de9f6b9d4788801f83408e0aa0a6d1fd3f5414eecb9cd33194c88ecffd` |
| `graph-model-fma-area-comparison.json` | `37fbf5fc66281c7b8ac9afbe1e2abcd1fc0c96934562375873417ead8b1be4db` |
| `graph-probe/model-graph-fma-area.patch` | `ed26b3d0b2299ea1206325741e296823c5f5d73b625aad97e2824c6b3006bd6d` |
| `graph-probe/model-graph-fma-area-trace` | `ea2310ce7ae933d2904e5aa341f1230014d9b1c78a7e5854c3935094269c6c98` |

The patch applies over commit `b132f56`. Reuse the build and controlled
trace commands above with `cudnn-graph` in place of the trailing
`cudnn-default-math` argument and fresh output paths. The diagnostic
creates a handle and plan for each forward convolution and synchronizes
before releasing them. The runtime tree was restored after the experiment.
A performance candidate needs a bounded reusable plan and handle, additional
development-history checks and the standard device regressions. The original
held-out quality and performance gates still apply.

### Cached graph and second development history

The experimental backend supplies the same fused resampling operations and
selects the first instant-heuristic graph plan. It caches plans by batch
and width on a single CUDA stream. cuDNN calls on the reusable handle
are serialized. Each launch holds CUDA access guards through submission.
Plans remain alive until stream completion before eviction or destruction.
A backend clone starts with its own empty graph state. [cuDNN graph API](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/api/cudnn-graph-library.html)

Use the controlled commands above with the trailing Rust option
`cudnn-graph`. The seed-11 Rust trace and its initial codec capture repeat
the synchronous prototype byte for byte. For the second history, append
`--controlled-seed 16` to the Python command and `16` after the Rust
`cudnn-graph` option. Both captures use the same pinned prefixes and
a different deterministic sequence of draws. With PyTorch forced to math attention, each of the two development
histories passed all 51 checks under the existing bounds, matching all
1,282 sampled codes, 256 greedy IDs, 256 penalized IDs and every reconstruction
value per history. The largest final-logit differences are
`1.907348633e-5` and `2.002716064e-5`, respectively.

The raw traces and comparison summaries are retained at
`reports/phase4-lm1b-cached-graph-v1.UJFuiB/`. The summaries
`comparison-seed11.json` and `comparison-seed16.json` bind the Rust and
Python capture hashes. Their SHA-256 values are
`f3fff7bca29329abfa8dd248b42875496021fb0a8e47d5b81c0396fb5f621729`
and `25cb28eb40bd3db832d35d5296cf5f67e500bbea0519b57fb614e26a3a7838f0`.
The paired timing and allocator screen is recorded separately.
[Cached graph timing](small-cuda-capacity-screen-v1.md#cached-cudnn-graph-candidate)


On both histories the fully device-resident generator matches the codes and
IDs from the traced CPU-sampling path. The device traces repeat byte for
byte after hardening the native handle initialization failure path. The
Linux host suite passed all 532 report tests, the standard CUDA probe
passed 18 cases, and all four real-device cuDNN lifecycle tests passed.
The graph lifecycle case submits 20 shapes and revisits cached widths,
checks the queued output values, and drops the backend on another host
thread. Feature-specific Clippy passes.

### Default PyTorch attention control

The frozen Python capture leaves attention dispatch at its default. A
separate controlled seed-11 capture preserved the evaluator convolution
settings and let Torch select its attention kernel. On this device its
CPU profiler observed 136 calls to the CUDA efficient attention operator
and no math attention calls. Against this reference the cached Rust
candidate has 25 of 51 tensors outside the existing bounds. Their sampled
codes first differ at level 6, row 0, position 27. This is a comparison with
the same controlled draws and is separate from the held-out score.

In the Python-only control, switching attention dispatch reproduces the
same first sampled-code difference. The preceding reconstruction and
the inputs and outputs of the level-6 post-upsampling convolution are
bit-identical. The disputed logits differ by up to `2.670288086e-5`.
The common uniform draw is `0.23733242566231638`. The cumulative
probability through code 2668 is `0.23733223839194803` with math
attention and `0.23733243634480927` with default efficient attention.
The draw selects 2669 and 2668, respectively.

The retained `control-default-sdpa.py` runs with the same reference
arguments and the retained `default-sdpa-first-draw.py` reproduces the
draw calculation. The default-attention graph comparison and first-draw
summary have SHA-256 values
`8e80d2d7cbd44d36b42f67c4b7d6249cf6c7dfcf96d2d508a9fe11668dbe8f76`
and `3faa7faa4e3dea2491f8d99ebc0811c12605d108d6c3ba72948b1f02d7fd871f`.
The first remaining model-level discrepancy is now localized to the
attention dispatch. It does not establish the cause of the entropy
interval failure across independent sample streams.

A second replay ran both attention modes on the same Q, K, V and mask in
all 12 blocks at the disputed level. Returning the default output at each
block reproduced the captured evaluator head logits bit for bit. The two
attention outputs differ at 931,840 of 976,896 values, with maximum
absolute error `9.059906006e-6`. The retained `level6-sdpa-inputs.py`
saves the operand values, original strides and both outputs. The
`level6-sdpa-fixed-inputs.safetensors` SHA-256 is
`1ee084cc10c595115b39e91776bf3ef3692f86b57c16e9eaa66050207a2f079a`.

### Efficient attention and bounded projection plans

The public CUDA profiler identifies PyTorch's FP32 64-by-64 efficient
attention kernel on this Ampere device. A standalone probe first
reproduced the 976,896 captured level-6 attention values by splitting
each FP32 operand into two TF32 components and staging the three
matrix products before adding to the running FP32 value. The unstaged
prototype failed, while the known single-key values and uniform
three-, 25- and 53-key probes helped isolate the staged product and row
normalization. The fused version carries each query tile's numerator,
maximum and denominator across 64-key tiles and keeps scores and
weights in block-local storage. [PyTorch attention source](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h) · [CUTLASS FP32 decomposition](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h) · [PTX matrix fragments](https://docs.nvidia.com/cuda/archive/11.4.4/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688)

The staged and fused operators each match all 22,341,888 expected
values across 129 cases. The cases cover the 12 captured level-6 blocks,
13 synthetic sequence lengths through 129, batches 1, 2 and 8, and
prefix, absent and per-batch additive masks. The additive masks exercise
rows with no valid key in the first tile. The two comparison reports
are byte-identical.

With the fused attention alone, the first sampled differences remained
at level 6 in seed 11 and level 9 in seed 16. The first differing tensor
in the seed-11 continuation is the 384-to-768 target projection, at
25,704 of 79,872 values. The seed-16 continuation is exact through the
first attention and second normalization. Its first discrepancy is the
768-to-3,072 feed-forward input projection, at 481,285 of 571,392 values.
The cuBLASLt screen verifies that a 1 MiB preference matches pinned
PyTorch on identical inputs for the target, QKV and both feed-forward
projections at these two shapes. The earlier graph and other backend modes
keep their existing preference.

The fused operator and selected cuBLASLt plans together pass all 51
checks for both seed 11 and seed 16 under PyTorch's **default efficient
attention** dispatch. Both histories match every code and final ID
with both CPU and fully device-resident sampling, and all codec
reconstruction values are bit-identical. Every captured intermediate
at the two previously disputed transformer levels matches exactly.
The additional captures keep the existing thresholds.

To reproduce the Rust controlled histories, use the earlier capture
command with `cudnn-graph-efficient`, appending `16` for the second
history. The retained `control-default-sdpa.py` uses Torch's default
attention selector with the same Python capture arguments. The
scripts, raw traces, projection screen and comparisons are retained
under `reports/phase4-lm1b-efficient-attention-v1.xYWlWf/`.

| Receipt | SHA-256 |
| --- | --- |
| `fused.json` and `online.json` | `8efdcc6c40236663ef3c55c357225393bd1e243ff2735619907690c8ba017925` |
| `components-staged.json` | `b10ac1dba22191b458e742134084d53947421fc0d9bfb330bf029ed9b00ca4dd` |
| `projections.json` | `b348302942a56dfc0388eb9dd1b216be067e3477ec80da5581971846890c6290` |
| `projection-model-seed11.json` | `d9a56803ff96660bf4259e325ab4ae6bbbf3b6584818b8b59be67a2ac4262077` |
| `projection-model-seed16.json` | `669104b2a1ad0bf7eee930e1ef1f77f0e70d513604dcb8c4c3aece5358cd2c65` |

The exact shared transformer inventory contains 28 tensors at
each disputed level. In the broader model replay the maximum
final-logit differences are `2.098083496e-5` and `1.716613770e-5`
for seeds 11 and 16. Both are within the existing bounds. The
earlier strict comparison repeats byte for byte and continues
to reject substitution of an efficient-attention oracle.

Feature-specific Clippy and five native cuDNN lifecycle cases pass.
An initial Linux host run overlapped scoring and compilation and
failed two existing process-observation fixtures. The isolated
`tools/check.sh --cuda` rerun passes all 532 report tests and
18 CUDA operator cases. Its log SHA-256 is
`eda806572dc07e1f353ced707db34df3a0f4d57580966e6955645dd1c39b7d40`.
The initial log is retained at
`d2408cba6f99be57eeee4595c9fc3216d8ba636839136f9550395751afe4838f`.

### Full observed sampler probability audit

On the efficient-attention checkpoint path, the pinned Python
`_sample_level_raw` supplies the actual FP32 top-p probabilities.
The audit intercepts that tensor and normalizes it in FP64 before
comparison with unfiltered FP64 softmax on the same logits. It
covers all 160 previously observed batch-eight requests in each
of two independently retained populations.

| Replay population | Sampled distributions | Rows with zero FP32 mass in at least one position | Maximum total variation | Mean total variation |
| --- | ---: | ---: | ---: | ---: |
| Rust request-local draws | 819,200 | 169,647 | `1.7710e-6` | `6.4084e-8` |
| Original Python native draws | 819,200 | 170,172 | `1.6485e-6` | `6.4144e-8` |

With Rust's midpoint uniforms and token-ID-ordered FP64 inverse CDF,
the reference FP32 weights produce 46 different local choices on
identical logits. The FP32 trajectory differs from the archived FP64
Rust samples at 6,404 concept positions and 1,730 raw token positions
after subsequent levels are generated. The maximum conditional
same-uniform disagreement probability is `0.003490`, obtained from
the overlap of the two sets of cumulative intervals. This describes
that choice of coupling. Total variation instead describes the
distance between the categorical distributions.

A separate replay uses PyTorch's native multinomial and its original
RNG state. It reproduces every archived original Python concept code
and raw ID. On each captured logit row, substituting the FP64 softmax
rounded to FP32 while restoring the same native RNG state changes
zero selected concepts. The alternate draw advances the RNG to the
same state, and the original state is restored before continuation.
The two sample audits describe those observed trajectories. They
preserve the independent-stream quality protocol and its original
failure.

The checkpoint hashes, pinned sampler source and artifact-bound
captures are checked in the retained `audit.py`. Both outputs are
under `reports/phase4-lm1b-sampler-audit-v1.O4RAPj/`.

| Receipt | SHA-256 |
| --- | --- |
| `audit.py` | `bc80c1d62e03679390e23d2a327056b34bdea4799aaa32761a780a95447c36ec` |
| `inverse-cdf.json` | `a5626efb1d9c96f53e9d86d730e2212052d872d9b800fdb6a133627d35505698` |
| `multinomial.json` | `ec1c39fa724aabea89aa75d005cf5beefc5a3af9e07a24e3022d608e4b7012f4` |

### Native FP32 probability experiment

The earlier [FP32 top-p prototype](#isolated-fp32-top-p-probability-prototypes)
was applied temporarily to `2376644`, retaining the newer graph, resampling,
efficient attention and projection paths. It uses the reference-shaped
FP32 sort, softmax, prefix sum, cutoff and final softmax with Rust's
token-ID-ordered FP64 inverse CDF. The temporary example selector was
`cudnn-graph-efficient-fp32`. The candidate accepts score batches up to 32,
sequence lengths up to 128 and at least two flattened rows. It stayed
explicitly experimental.

Feature-specific Clippy and two real-device tests pass. The tests
include 2,565 crafted FP32 boundary rows, strided scores, invalid
values and shape rejection. The local patch SHA-256 is
`3b528fb1c97439395569259217b88a35d0faa0a3908144b9c4f7b49a338ae24b`.
The patch, build/test logs and both executables are under
`reports/phase4-lm1b-sampler-audit-v1.O4RAPj/`.
[Paired timing](small-cuda-capacity-screen-v1.md#efficient-attention-with-native-fp32-top-p)

The pinned Python model, using its own FP32 weights with Rust's exact
uniforms and inverse CDF, reproduces the native FP32 candidate across
all 160 previously observed batch-eight requests. All 820,480 codes
and 163,840 raw token IDs match. Comparing the native FP32 and FP64 Rust
captures independently counts the same 6,404 code and 1,730 raw-ID changes
as the Python intervention. The bounded result identifies the probability
path responsible for those changes.

The retained `native-fp32-matched-draws.py` checks the candidate's capture
and binary receipts and the pinned Python checkout and checkpoints.
Its source SHA-256 is `f00028b6b96bc3e9efe2d230a7be05359ef5c0d9a594bfa0c40415c50186b18d`.
Its result SHA-256 is `b09dbf3d4e49d1bfb6e540b677c4c311bcb5f0ee642306acb590f9f94cbc5b34`.
The [independent-stream retest](small-heldout-cuda-v1.md#native-fp32-top-p-observed-seed-retest)
still fails the frozen entropy check. The FP32 candidate is archived as
a diagnostic and the faster FP64 sampler remains in the tracked source.

### Experimental Linux CLI

The opt-in [CLI command](../../PHASE4-TEXT.md#experimental-lm1b-small-cuda-command)
selects the existing efficient-attention CUDA backend and the FP64
sampler and final penalty path. It uses the checked inference artifact
and requires the hash-pinned rotary export explicitly. The rotary file
decoder is shared with the capacity probe. The CLI releases the host
artifact after loading the model and emits JSON arrays of intermediate
codes and raw token IDs for each start-token row.

The final release binary reproduces 15,384 concept codes and 3,072 raw
IDs exactly across three archived batch-eight requests using seeds 42,
0 and 3 and request indices 0, 1 and 31, respectively. The first
development timing signatures match at batches 1, 4, 8 and 16. Two
batch-32 calls using the development request seed repeat byte for byte.
Across both tested builds all nine positive outputs have identical
hashes.

The CLI rejects an empty batch, a wrong or multi-token prefix, more
than 32 rows, a short or wrong-hash rotary file, an invalid request seed
and an unsupported device. These checks return errors before opening
the supplied missing model artifact. The default build explicitly
reports that the experimental CUDA path is unavailable. The feature
build, three CLI unit tests, feature-specific CLI Clippy and CUDA
capacity-example Clippy pass. The full Linux `tools/check.sh --cuda`
suite passes 533 report and data tests and all 18 real-device CUDA
operator cases. Its final log SHA-256 is
`cc22456fa65b576ff2c0fa92caeec933bc6262004ee62602b91a6141b5d610dd`.
After sharing the rotary validator, the existing capacity example
also reproduces all eight retained batch-eight output signatures.
Its replay log SHA-256 is
`765075fd5ce5c7bdf45f341d34853d4d5bf33adaa41a040d2bc421f099738e94`.

The first build exposed a missing CLI forwarding flag for the model
feature. The first focused Clippy check flagged the fixed-size slice
calls. Both were corrected and verified on the final binary. Raw
captures, two sets of logs, and negative controls remain under
`reports/phase4-lm1b-small-cli-v1/`. The release binary SHA-256 is
`782da09d8dbc4cc0ca3fd749619de12c1109502b08ea0094bee1ca20de69b300`.
The final smoke script and receipt SHA-256 values are
`8adedab9aa5026698782ac9d93ec61e5670ad768557cacc24ba260f6de0d5373`
and `1f02b13099fe6442ee0c8190286c0af9585aaa8fdad7d96966193e516ccb8acf`.

The [independent CUDA quality gate](small-heldout-cuda-v1.md)
remains open. These requests establish a reproducible CLI entry point
for this experimental runtime on the tested device.

### Metal controlled stochastic comparison

The experimental Metal path ran the pinned two-row controlled sample on
Mac Studio. The returned codes and IDs match its untraced replay. The
existing comparison fails 34 of 51 tensor checks. Generated codes differ
at 505 of 1,282 positions and penalty-adjusted IDs at 130 of 256 positions.
Level-five logits pass the existing band. Level-six post-convolution input
is the first level-specific generator tensor outside the band.

The Metal trace SHA-256 is
`a464a0e036a8af5d6d191e43747c67007ba14e6e5880d1f214a93a4ab525e870`.
The failed comparison SHA-256 is
`1b466fbc90746a203c037b25c0685b34e5075dae0774205d874d75b195a3d783`.
The Mac trace and log are retained under
`reports/phase4-lm1b-small-controlled-metal-v1/` and
`reports/phase4-lm1b-small-controlled-metal-v1.log`. The unchanged
comparison uses the earlier strict CUDA reference and remains a failed
stochastic diagnostic. Fixed-history and greedy Metal traces pass separately.

The first changed Metal code is row 0, level 5, position 20. The
shared uniform is `0.4033909708959982`. The cumulative probability
through code 3054 is `0.40339013162117493` on the pinned CUDA logits
and `0.40339146448637764` on Metal. The CUDA reference selects 3055
and Metal selects 3054. A replay using the captured Metal logits and
the same stream reproduces all 1,282 Metal choices. This identifies
the observed first divergence and leaves the full stochastic gate open.
The audit and result are retained under
`reports/phase4-lm1b-metal-first-choice-v1/`.

Before the first sampled-code difference, Metal and strict CUDA already
differ in the initial reconstruction. On the captured two-row stochastic
history, `reconstruction.level_00` differs in 74,984 values with maximum
absolute difference `0.00012969970703125`. Level-five logits differ in
619,374 values with maximum `0.00017547607421875`, within the accumulated
comparison band. The raw per-boundary audit is retained under
`reports/phase4-lm1b-metal-prechoice-v1/`; result SHA-256 is
`c1165cfd8cfacc88e08b0aaf0f0ea4c9c80049ad19b36019f0b91fe243ccd26a`.
The separately captured [codec fixtures](small-codec-cuda-rust-v1.md#metal-codec-comparison)
show level-zero numerical differences on their own inputs.

## Adapted Python Metal generator diagnostics

A separate Python MLX adaptation used the authenticated LM1B artifact,
existing MLX transformer operations and the bounded LM1B codec adaptation
in the [codec record](small-codec-cuda-rust-v1.md#same-device-adapted-codec-comparison).
On the pinned two-row fixed history, all 45 captured generator and
reconstruction intermediate tensors pass the existing policy against Rust
Metal, including all checked per-level logit argmaxes. This subset does not
include decoder output or sampling. Its trace and comparison SHA-256 values
are `a7965e1cd2d954227361d838d935760da4d5499fe7024a78fee4fbd1f122bb25`
and `52ae6938bdd73e5c85a151de8841620c3c6cdfb45a9e63d94a45181a1f4926cd`.

With the same controlled SplitMix64 draws, the adapted generator fails 30
of 47 captured intermediate and code comparisons against Rust Metal, with
686 of 1,282 sampled codes different after continuations diverge. The first
different choice in draw order is level 4, row 1, position 17: Rust selects
6513 and Python selects 6512 at shared uniform `0.7931153684621677`.
The Python cumulative probability through 6512 is `0.7931153701605017`.
Level-four logits remain inside the existing numerical band, with maximum
absolute difference `0.00003814697265625` and no argmax differences. A
host replay of each capture's own logits and the same draw stream reproduced
all 1,282 of that capture's choices. This shows the observed divergence is
sensitive to the captured logits, but does not independently qualify the
adaptation or establish full stochastic or decoder parity.

The controlled capture and comparison SHA-256 values are
`970dc5bbeae791f25faaecac33562713e38d814764fe86bd91eb57ec048b392b`
and `1b6160fd33be0a31d935cabe5cb47db6f09d770fc3a5882aa5e5c5835c0a02e5`.
The source scripts have SHA-256
`e3be190ede219323eec7f261bf4e4c656ec729e2ac4d636038e355ea48f69758`
and `02aaf20e556f24aa61fb90c521e7f33300e08470e221c51cd57badac67d77cf4`.
Scripts and results are retained respectively under
`reports/phase4-lm1b-small-teacher-python-metal-v2/` and
`reports/phase4-lm1b-small-controlled-python-metal-v1/`; the shared codec
adaptation is retained with the codec evidence. These remain development
checks, not quality or performance qualification.
