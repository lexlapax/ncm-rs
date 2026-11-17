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
