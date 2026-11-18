# Agent context map

Last updated: 2026-11-17 UTC. Read first; load linked context lazily.

## Current state and user decisions

- **LM1B combined CUDA diagnostic:** with pinned rotary tables, two controlled histories pass all 51 comparisons; an isolated fixed-table version reproduces both traces byte-for-byte. Five development seeds fit descriptive margins, but an already observed five-seed retest still fails the unchanged entropy interval `[-0.006477, 0.002497]` against `[-0.0046, 0.0046]`. This is not fresh held-out evidence. The fixed table and arithmetic changes remain unintegrated and unqualified; full checks, performance and Metal are unrun. [Evidence](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#combined-cuda-arithmetic-diagnostic-and-observed-seed-retest). XSum Metal and the frozen LM1B quality gate remain open.

- **LM1B second controlled history and isolated CUDA candidates:** a new two-row seed-16 comparison fails 7/51 tensors after two code differences; the first is an inverse-CDF boundary crossing. Exact reference rotary tables do not remove it. Independent same-input probes identify FP32 fused interpolation, target-projection heuristic and area reciprocal order as sources of drift; combined experimental changes make reconstruction through level 9 and level-10 QKV exact in this history, but two code differences persist. A split-scale attention variant also retains two differences, so no candidate was integrated or qualified. Targeted Clippy passes; full checks, quality, performance and Metal are unrun for these isolated changes. [Evidence](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#second-controlled-history-and-isolated-arithmetic-candidates). The XSum gate and frozen LM1B quality gate remain open.

- **XSum reciprocal-square-root control:** on the retained 518 CUDA variances, mathematically nearest FP32 reciprocal square roots differ from CUDA in 48 cases by one ULP; 80-digit decimal midpoint checks verify the rounding for these inputs. This rules out assuming correct rounding alone reproduces the captured CUDA values, not a full-model fix or gate result. The control repeated and its Python syntax check passed. [Evidence](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md#correct-rounding-control). The exact Metal gate remains open.

- **LM1B intrinsic observed-seed retest fails:** after the intrinsic rotary change, all five already observed seeds were regenerated and scored with the pinned evaluator. The unchanged comparator fails entropy interval `[-0.007822, 0.002026]` against `[-0.0046, 0.0046]`; the mean and perplexity checks pass. The repeated summary is identical. These are retests, not fresh held-out data; the original failure remains. Seed-guard test, diagnostic Clippy, scorer syntax, invalid-identity refusal and three comparator tests pass. [Evidence](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md#intrinsic-rotary-observed-seed-retest). The protocol decision remains pending; XSum Metal is still open.

- **LM1B intrinsic rotary candidate:** experimental LM1B-small geometry computes inverse frequencies before multiplying positions, matching upstream operation order without an external table. On Linux this narrows the reference table's maximum difference to `5.96e-8`; the two-row controlled CUDA comparison passes all 51 tensors. Five observed development seeds meet descriptive comparison bounds, but many generated rows differ from the earlier external-table candidate; no prior held-out or performance result transfers. A fresh 3-process, 8-trial capacity screen favors Rust in median/p95 and process RSS at all three batches, but is not the full gate. The full isolated Linux/CUDA check passes, including 524 report tests and 18 device cases; a first run failed because the isolated checkout lacked `reports/`. [Evidence](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#intrinsic-reference-order-rotary-candidate). The frozen LM1B quality gate, full performance and Metal qualification remain open; XSum Metal still precedes further profile advancement.

- **LM1B observed-reference partitions:** 169 of 252 overlapping five-versus-five splits of the ten observed Python seeds fail the frozen entropy interval; nine splits cross-check against the comparator; the control reproduced twice and its Python syntax check passed. These are dependent, post-hoc descriptive controls, not a population failure rate or Rust qualification. The protocol remains unchanged while the operator decision on a separately labeled alternative is pending. [Evidence](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md#observed-reference-partition-control).

- **Bounded precision and quality controls:** the captured XSum row's CUDA head decision changes when its codebook dot is evaluated in FP64, even with the CUDA hidden state; this does not produce a portable fix. The frozen LM1B comparator passes identical retained Python scores on both sides, while the actual candidate still fails. All three diagnostics ran, CUDA on the real device; Python syntax, three comparator tests and diff checks pass. [XSum](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md#head-precision-control), [LM1B](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md#identical-reference-control). Next: resolve the unchanged XSum Metal and LM1B CUDA gates.

- **Session handoff:** [RESUME.md](RESUME.md) captures the current Phase 4 objective, open gates, machine/service state, authorizations, and next actions for another CLI session. Recheck live state before acting.

- **XSum six-part Metal screens:** after explicit user approval, three combinations ran on both retained batches with `dotsetup.rmlx` stopped; all six outputs are byte-identical to the earlier candidate, retaining the row-4937 exact mismatch. Mac release Clippy/build pass, input hashes remain unchanged, and the service was restored and verified running. An initial asynchronous shutdown/restoration attempt aborted before testing and was recovered; [details](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md). XSum Metal and LM1B-small CUDA quality gates remain open.

- **Linux validation scheduling:** three independently isolated existing report tests overlap after all others run serially. The scheduler checks full test identities and fails on omissions or failures; all 520 prior tests plus four scheduler tests and 18 CUDA device cases pass. A single suite comparison saved about 21 seconds (167.35 to about 146.48 s); [evidence](docs/results/2026-11-17-phase4-check-scheduling.md). No model or acceptance gate changed.

- **XSum six-part head combinations:** explicit diagnostics combine the earlier six-part head with reference-order projection, refined normalization, and optionally CUDA-order GELU. No default, selection rule or tolerance changes. Full Linux/CUDA validation passes (18 device cases). A CUDA-only refined-hidden head screen selects the differing code for all layouts; the subsequent Metal results are recorded above. [Evidence](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md).

- **XSum GELU ULP-partition control:** on the captured CUDA continuation, replacing only high-precision GELU values two to four FP32 ULPs away flips the code; one-ULP substitutions alone retain it and five-or-more produce a tie. Positive and negative subsets of the two-to-four group each flip. Replays repeat byte-for-byte; this is case-specific evidence, not a tolerance or gate change. [Details](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md).

- **XSum CPU erf controls:** a separate FP32 CPU erf expression and system C-library erf also fail to reproduce the CUDA decision under the retained continuation; these are bounded controls, not runtime fallbacks. [Evidence](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md).

- **XSum high-precision GELU control:** FP64-erf/FP32-expression intervention matches 1,493,683/1,591,296 CUDA values but independently flips the decisive CUDA continuation; exact CUDA GELU retains the reference choice. Prefix-only substitution flips while target-only retains it; both halves of the first 128 prefix tokens independently flip. This rules out assuming more accurate erf alone fixes this case, without proving a general cause. [Evidence](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md).

- **XSum independent mathematical GELU diagnostic:** an erf-table/Taylor approximation matches 1,494,171/1,591,296 CUDA values on the refined row and 1,516,010 on a synthetic input. A CUDA continuation reaches an exact tie (not a robust pass). An explicit full-model candidate passed CUDA and targeted Metal checks but its two Metal outputs were byte-identical to the prior candidate and retain the failure. The unqualified runtime candidate was removed; the restored Metal regression passes and the checked-in mathematical probe reproduces its bounded result. [Evidence](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md).

- **XSum vendor-library restriction:** an investigation disassembled the installed CUDA device library before noticing the SDK license restriction; temporary output was deleted and no vendor implementation was copied or used. Do not repeat; use independently developed or appropriately licensed material.

- **Mac full-check compatibility:** the first full MLX check failed before Metal tests because Rust 1.99 Clippy rejects three existing constant-size chunk iterators. Equivalent fixed-size chunk fixes pass full Linux/CUDA and Mac/Metal validation (18 device cases each); all four Metal screens reproduce byte-for-byte. A separate native `metal::erf` probe failed to compile; no model change. [Evidence](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md).

- **XSum refined-input same-operator GELU:** the best tested Metal expression matches 1,181,365/1,591,296 same-input CUDA values, leaving an operator discrepancy. [Bounded evidence](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md); no full-model fix or exact gate pass.

- **XSum refined-normalization GELU isolation:** the forced Metal row still selects the wrong code. Its first-MLP-input CUDA substitution retains the reference choice, but GELU-output substitution flips it; [bounded evidence](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md). Two explicit refined/GELU combinations pass full CUDA checks (18 device cases) and the Metal normalization regression, but both Metal screens match the prior candidate byte-for-byte and retain the row-4937 failure.

- **XSum refined-normalization trace preparation:** a separate opt-in Metal forced-row example will capture the existing refined candidate at row 4937 to locate its remaining differences. Full Linux/CUDA validation passes (18 device cases); the Mac trace and subsequent replay are recorded above. The exact gate remains failed.

- **LM1B Python self-control:** the same frozen metric comparison between the five Python development seeds and five Python held-out seeds also fails the entropy interval (lower endpoint -0.006221). The [descriptive control](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md) repeats exactly; it does not waive the candidate failure or change the gate.

- **LM1B-small CUDA held-out quality fails:** the frozen five-seed candidate comparison passes perplexity and entropy point checks but fails the entropy bootstrap lower bound (-0.007118 versus -0.0046). [Results](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md), scores and reproducible failed summary are retained. The seed set is now observed; later same-seed runs are retests. Do not change the frozen margin or claim the CUDA/Phase 4 gate.

- **LM1B held-out tool preparation:** separate Python and Rust capture paths enforce the frozen seeds and 256-sample batch-eight protocol; a separate scorer checks generation identities and requires exact agreement with pinned upstream. The subsequent official run is reported above. Seed guard, feature lint, metric-comparator tests, missing-input rejection and full CUDA validation pass (18 device cases).

- **LM1B candidate device/performance screen:** CUDA inverse-CDF and penalty match candidate CPU codes/IDs on all 1,280 development samples. Three-process non-held-out paired screens favor the candidate in median/p95 and process RSS at batches 1/4/16; desktop activity and sampled-memory limits apply. [Details](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md). Feature lint, focused tests and full CUDA validation pass (18 device cases); official gates remain pending.

- **LM1B candidate development quality:** the opt-in reference-table/QKV candidate was scored on five non-held-out 256-sample seeds; candidate-minus-Python mean perplexity is -0.225208 and entropy -0.000109, with all pinned scorers matching upstream. These are descriptive, not held-out results. Feature lint, seed guard, seed-11 capture/scorer byte-for-byte repeat and full CUDA validation pass (18 device cases); official quality remains pending. [Details](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md).

- **LM1B reference-table intervention:** the explicit QKV/reference-operator candidate supplied pinned Python rotary tables passes the retained two-row controlled comparison (51/51 tensors, exact generated codes/IDs). Tables are diagnostic input, not an artifact/schema/default change. The table-only variant still fails 21/51 tensors. Dimension and finiteness regression, feature clippy and full CUDA validation pass (18 device cases); broader quality/performance evidence is pending. [Details](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md).

- **LM1B rotary and attention isolation:** expanded opt-in traces show normalized Q/K and V match Python exactly while rotary tables and rotated Q/K differ. Python rotary arithmetic with Rust tables reproduces Rust Q/K exactly; Python math-SDPA with Rust Q/K/V still differs from Rust attention. A separate-operand scaling candidate still fails 21/51 tensors and was reverted. Feature clippy and full CUDA checks pass (18 device cases). [Details](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md).

- **LM1B level-seven diagnostics:** the opt-in trace preserves the previous capture. A same-input cuBLASLt screen identified a QKV algorithm preference that exactly matches the Python projection for this case; an explicit candidate reproduces it but still fails the unchanged controlled comparison (21/51 tensors, 116 codes and 42 IDs). Attention differs immediately after matching QKV; a Python same-input probe exactly reproduces math-SDPA by scaling Q and K separately. Feature clippy and full CUDA checks pass (18 device cases). [Details and hashes](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md); default execution and the Phase 4 gates remain unchanged.

- **LM1B controlled-draw reference-operator diagnostic:** a standalone CUDA example selects the existing reference-order backend for the retained two-row stochastic trace. Clippy/build pass, but the unchanged comparison still fails 21/51 tensors with the same first controlled-draw divergence; 83 codes and 31 penalized IDs differ. A same-input pinned-Python CUDA replay exactly reproduces reference level-seven logits and selects the reference code even when supplied either Rust candidate’s captured latent/convolution values, locating an independently sufficient difference downstream. It remains experimental; no default or tolerance changed.

- **WMT14 allocator evidence preparation:** explicit Rust CUDA-pool and pinned-Python allocator probes record load, live output and post-drop snapshots for the qualified selection candidate. [Source-bound seven-cell observations](docs/results/2026-11-16-phase4-wmt14/cuda-fast-argmax-plan-v1.md) completed with matching outputs and recorded load, peak and post-drop counters. CUDA pool and PyTorch allocator scopes and startup boundaries differ; WMT14 Metal and Phase 4 remain pending.

- **XSum refined-normalization combination:** an explicitly selected Metal diagnostic combines reference-order projection and refinement with a Welford normalization using refined inverse; full `tools/check.sh --cuda` passed including 18 device cases and the Metal regression passed. Both Metal free-generation batches are byte-identical to the prior candidate, retaining row 4937’s three code and five raw-ID differences. The exact gate remains failed. Default inference is unchanged.

- **WMT14 CUDA final selection diagnostic:** synchronized profiling isolates the existing generic first-tie selection at about 94 ms for batch 32 on retained real logits. An explicit bounded one-pass CUDA candidate matched the generic result on real logits and ties/NaN/infinity tests; isolated median fell from 94.05 to 2.93 ms on batch 32. An opt-in backend passes focused and full CUDA checks (18 device cases), both 3,003-row base/CFG exact retests and frozen scoring. [Repeated paired timing](docs/results/2026-11-16-phase4-wmt14/cuda-fast-argmax-plan-v1.md) favors Rust in median/p95, RSS and sampled GPU memory at all scoped batches, with desktop activity. The first full check failed an unrelated lifecycle fixture; isolation and the full rerun passed. Default inference and remaining gates are unchanged.

- **WMT14 CUDA internal profiling:** a standalone example reuses existing explicitly enabled level-one trace checkpoints to break down the uncached generator path. CUDA build/lint passed, base/CFG traces match retained Rust outputs, and matching pinned Python block profiles reproduce those outputs. These synchronized intervals are instrumentation, not a performance gate.

- **XSum first-block continuation:** the reference-order candidate reproduces CUDA projection on its captured latent, but its forced Metal block-0 output independently flips the exact CUDA continuation. Substituting either captured prefix or target first-normalization outputs also flips it. The [same-input normalization probe](tools/reference/xsum_candidate_norm_metal_probe.py) finds Welford differs less than fast Metal but still flips the CUDA continuation. Safe Welford statistics match all 518 CUDA means, but reciprocal standard deviation differs for 53; forcing CUDA mean and reciprocal standard deviation yields all 397,824 exact Metal outputs. Safe Welford variance matches all 518 CUDA values; tested Metal inverse formulas still differ from CUDA in 48–77 values. The refined inverse yields an exact tie in one CUDA continuation; `1/sqrt` still chooses the differing code. This is not a robust fix or gate pass.

- **XSum reference-order projection diagnostic:** Metal two-part matrix products with bias on the first part exactly match CUDA on one captured and four seeded inputs. An explicitly selected candidate combines this with the refinement diagnostic; full `tools/check.sh --cuda` passed including 18 device cases and Metal bias-order regression passed. The two Metal free-generation outputs are byte-identical to the preceding candidate, retaining the known row-4937 failure; a forced trace is pending. Default inference is unchanged.

- **XSum three-part projection diagnostic:** an explicitly selected Metal candidate combines the prior refinement diagnostic with three-part projection on the six-vector target. Full `tools/check.sh --cuda` passed including 18 device cases. Both Metal free-generation batches are byte-identical to the prior refinement candidate, retaining three code and five raw-ID differences on row 4937. It remains experimental; the exact gate fails.

- **XSum target projection screen:** [same-input Metal layouts and CUDA continuations](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) confirm the captured Metal projection; none exactly matches CUDA. Three parts reaches the CUDA-selected code only through an exact tie in the controlled continuation. No inference change or gate pass.

- **XSum refinement layout candidate:** the flattened Metal convolution exactly matches CUDA on one controlled input. Full `tools/check.sh --cuda` passed including 18 device cases. [Metal free-generation screens](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) match CUDA on batch 4864 but retain the three code and five raw-ID differences on batch 4928. A forced-row trace and controlled CUDA replay show the candidate latent with CUDA projection preserves the reference choice, while the captured candidate transformer input flips it. Default inference is unchanged and the exact gate remains failed.

- **XSum refinement same-input probe:** the [controlled Metal comparison](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) reproduces the captured Metal reconstruction exactly but differs from CUDA in 16,653/18,432 convolution values. This is one captured row, not a fix or gate pass. Default inference is unchanged.

- **XSum CUDA intervention replay:** the [bounded row-4937 replay](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) exactly reproduces CUDA hidden states and the decisive head margin from captured block boundaries. Substituting only Metal target input, or Metal latent followed by CUDA projection, flips the choice under CUDA replay. A further CUDA replay preserves the reference choice from Metal cascade normalization or convolution, but flips it from Metal reconstruction, narrowing this one intervention to refinement or its interactions. This does not qualify a fix or pass the Metal gate.

- **XSum Metal-hidden head-chain screen:** the [controlled chain probe](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) tried five head layouts on the captured Metal split-head hidden state; all select 3690. Its CUDA reference scores use a different hidden state, so exact-match counts are not same-input accuracy. No inference change or gate pass.

- **XSum head-chain probe:** the [standalone Metal screen](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) found six-part projection reproduced the decisive controlled CUDA margin on the inspected CUDA hidden state and had the most exact matches on four synthetic screens, but most logits still differed. The explicit six-part model passes full CUDA checks and Metal bounded screens, but its outputs match the prior split-head candidate and retain the row-4937 failure. Default inference is unchanged; the exact gate remains open.

- **XSum head-projection layout screen:** the [same-input Metal comparison](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) tested three-, four- and six-part layouts. None reproduced CUDA; six parts matched 502/2,304 values and reduced maximum error to `3.8147e-6`. No runtime change or gate pass.

- **XSum double-split head diagnostic:** an explicitly selected Metal backend splits both the hidden-to-head and codebook projections into two products, with the previous upsampling candidate. Default inference is unchanged. Full `tools/check.sh --cuda` passed including 18 device cases; [Metal free-generation screens](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) reproduce the earlier split-head outputs, retaining the row-4937 differences. The exact gate remains failed.

- **XSum same-input head projection:** [PyTorch CUDA and Metal comparisons](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) use captured hidden states and original batch 32. CUDA reproduces all 2,304 retained values; ordinary Metal matches 271 and split Metal matches 439. The split reduces maximum error but is not reference-equivalent; no inference change or gate pass.

- **XSum head norm diagnostic:** the [controlled head probe](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) finds independently computed Metal norms differ from PyTorch CUDA norms, but CUDA dots with Metal norms retain the CUDA pair choice. CUDA head arithmetic on the retained Metal split-head projection selects the Metal code instead; upstream projection drift can independently change the decision. This is a controlled row, not an exact runtime-norm replay or gate pass.

- **XSum split-head combinations:** explicit Metal diagnostics combine the split head with prior group normalization, compiled GELU, and all-operation variants. Default inference is unchanged. Full `tools/check.sh --cuda` passes, including 18 device cases. [Metal free-generation screens](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) are byte-identical to the earlier split candidate on both known failure batches and retain the row-4937 differences; the exact gate remains failed.

- **XSum head-layout screen:** the [controlled Metal comparison](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) found two 192-wide products sum to exact controlled CUDA dots on the inspected row and four fresh synthetic inputs. The split-head plus upsampling Metal model narrows the forced row-4937 margin to `-9.5367e-7` but free generation still differs from CUDA in three codes and five IDs on that batch; the 4864 batch matches. Full CUDA check passes 18 device cases; full quality/performance and the exact gate remain open. Default inference is unchanged.

- **XSum controlled head-stage probe:** the [same-input comparison](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) found Metal post-dot arithmetic exactly matches all 24,576 controlled CUDA logits when supplied CUDA dot products; independently computed Metal dots differ and retain the pair crossing. This localizes the controlled discrepancy to the dot product, not full model causality or a gate pass. No inference change.

- **XSum same-input head screen:** the [controlled Metal and CUDA comparison](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) uses identical captured CUDA projection, codebook, CUDA norms/scales and 32-row geometry. It still crosses 3182/3690, showing the head arithmetic can independently produce the difference. Controlled PyTorch CUDA reproduces the reference pair margin but differs in 5,692/24,576 logits, so this is not an exact replay or gate pass.

- **XSum head precision diagnostic:** on row 4937's captured CUDA projection, an [FP64 unrounded distance recomputation](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) favors code 3690 while the pinned FP32 reference chooses 3182. A generic higher-precision near-tie fallback would not preserve this reference decision; no rule or tolerance changed.

- **XSum free-generation diagnostic:** an explicit Metal example captures complete codes and raw IDs for 32 rows using selected experimental backends; it does not change inference defaults. `tools/check.sh` passed and prior CUDA runtime validation remains applicable. The [bounded Metal screens](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) found all five variants correct the row-4883 raw-ID difference but byte-identical to default on the batch containing row 4937, retaining three code and five raw-ID differences there. The exact Metal gate remains failed.

- **XSum all-operation diagnostic:** an explicitly selected Metal backend combines experimental upsampling, group normalization, Welford layer normalization and compiled GELU. Default inference is unchanged. Full `tools/check.sh --cuda` passed including 18 device cases (`reports/phase4-xsum-all-candidate-cuda-check-v1.log`). The [row-4937 Metal trace](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) still has the wrong level-1 code and a worse margin than the previous candidate. No gate is claimed.

- **XSum compiled GELU screen:** the [standalone Metal comparison](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) found more CUDA matches for compiled multiplication than compiled division on the captured and synthetic inputs, though fewer than uncompiled multiplication. The explicit Metal model diagnostic's row-4937 capture is byte-identical to the prior uncompiled candidate and retains the code crossing. Full CUDA checks pass including 18 device cases; default inference is unchanged and the Metal gate remains open.

- **XSum second normalization same-input probe:** on 512 captured prefix vectors, [Metal fast and Welford normalization](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) matched 245,194 and 299,356 of 393,216 CUDA values respectively; pinned PyTorch CUDA matched all. This is a bounded operation comparison, not a causal fix or gate pass.

- **XSum same-input MLP projection probe:** the [standalone Metal diagnostic](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) and CUDA both reproduce all 1,591,296 retained projection values on the captured row in original 32-row geometry with other rows zero-filled; row-only CUDA geometry differed. This isolates the observed projection drift to differing inputs for this case, not general parity. The Metal gate remains open.

- **XSum synthetic GELU screen:** a [fresh synthetic input](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) shows 1,203,197/1,591,296 exact CUDA matches for the standalone CUDA-order Metal expression versus 1,080,669 for composed Metal. It remains a bounded operation test; the model candidate still fails the known crossing and the Metal gate remains open.

- **XSum GELU model diagnostic:** an explicit experimental Metal backend combines the previous cascade probes with an uncompiled MLX CUDA-order GELU expression. The default backend is unchanged. Full `tools/check.sh --cuda` passed, including 18 device cases (`reports/phase4-xsum-gelu-model-cuda-check-v1.log`); the [Metal row-4937 trace](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) still selects the wrong level-1 code, although the margin narrowed slightly. No quality or performance gate is claimed.

- **XSum GELU order screen:** the [standalone Metal same-input probe](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) compared a CUDA-order expression under safe, relaxed and fast math. After two documented JIT header failures, all modes matched 1,180,744/1,591,296 CUDA values versus 1,034,783 for composed GELU. This is a single inspected row, not a Rust model or performance result; no runtime behavior changes.

- **XSum same-input GELU probe:** the [standalone Metal comparison](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) uses captured first-block MLP input and CUDA GELU output. PyTorch CUDA reproduces all 1,591,296 values; Metal composed matches 1,034,783, maximum difference `7.1526e-7`. A row-sized compiled Python graph has different statistics and does not prove full-batch Rust behavior. The exact Metal gate remains open.

- **XSum first-block MLP trace:** a [fresh 39-tensor CUDA forced-history capture](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) reproduces every retained CUDA tensor exactly and compares the three later MLP-stage tensors with the existing Metal candidate. The largest first-block difference is in a prefix position; input differences prevent attributing causality to an operation. The exact Metal gate remains failed.

- **XSum synthetic normalization screen:** [four synthetic inputs](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) compare reference-layout and refined inverse operations against CUDA. Refinement improves two cases, worsens one and ties one; neither gives general exact parity. These are synthetic diagnostic inputs, not quality or performance evidence.

- **XSum inverse rounding audit:** the [isolated CUDA and Metal variance captures](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) have bit-identical means and variances in all eight groups, and the CUDA probe reproduces all retained CUDA outputs. Group 5's CUDA inverse is one step below Metal; group 4's CUDA inverse is one step above a CPU double-precision inverse rounded to FP32. Correct rounding alone cannot reproduce all eight CUDA values. No inference behavior changes.

- **XSum refined inverse probe:** the [standalone Metal refinement](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) fixes the inspected row's group-5 inverse discrepancy but introduces a one-step discrepancy in group 4; 4,094/4,608 output values match CUDA. It is not a runtime fix.

- **XSum group statistics diagnostic:** [CUDA and Metal reference-order means](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) match in all eight groups for the inspected row; inverse standard deviations match except group 5, where Metal is one FP32 step higher. Explicit `1 / sqrt` fixes group 5 but introduces three other group differences; no inference behavior changes.

- **XSum group-normalization math-mode screen:** the [standalone Metal probe](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) compared safe, relaxed and fast reference-kernel math on the captured row. Safe matched 4,117/4,608 CUDA values, with all 491 differences in group 5; relaxed and fast matched 2,523, with differences in seven groups. No model backend change or gate acceptance follows.

- **XSum group-normalization model diagnostic:** an explicit experimental Metal backend combines the first-upsample candidate with the CUDA-layout group-normalization kernel, restricted to the validated XSum shape. The default backend is unchanged. Full `tools/check.sh --cuda` passed (18 real-device operator cases); the Rust Metal build passes. The first forced-history run failed because the tracer expected 36 tensors despite three later instrumentation additions; the corrected 39-tensor guard and full CUDA check passed. The [corrected row-4937 Metal capture](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) has 491 differing normalization values, all in group 5; seven groups are bit-identical. It retains the level-1 crossing with the same two logits as original Metal. No Metal gate is claimed.

- **Mac testing pushes:** the user approved all pushes needed for Mac testing via pulls going forward. Review scope before each push; do not overwrite unrelated remote work.

- **XSum group-normalization probe:** the [bounded Metal diagnostic](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) compares operations on a hash-bound captured row. The reference-layout 512-thread reduction matched 4,117/4,608 CUDA values exactly versus 3,033 for the current composed operation; maximum absolute difference remains `4.7684e-7`. A forced-history model test is the next step; the Metal gate remains open.

- **WMT14 eight-row uncached follow-up:** the [diagnostic](docs/results/2026-11-16-phase4-wmt14/cuda-uncached-cfg-eight-v1.md) exactly matches all 3,003 pinned CUDA Python codes and IDs, and independently replayed frozen scoring passes. In the repeated batch-16 comparison Rust median was 1.58% slower than Python, so performance is not accepted. Prioritize the earliest failing XSum Metal gate; avoid further later-profile optimization until it is unblocked. Full candidate Metal work requires the separately requested push authorization.

- **Phase 4 active; Mac available:** the clean Mac checkout is at pushed `a826032`. The [full Metal top-1 comparison](docs/results/2026-11-15-phase4-xsum/mlx-full-top1-v1.md) fails the frozen raw-ID gate: nine code and six ID differences versus Python, all ID differences after EOS. CUDA same-batch and batch-2 teacher diagnostics reveal near-tie/context sensitivity; the existing Metal batch-2 fixed-history probe passes, but does not resolve the batch-32 failure. The [same-batch forced-history traces](docs/results/2026-11-15-phase4-xsum/mlx-full-top1-v1.md) identify an in-band final-score crossing for row 4883 and an in-band head-score crossing for row 4937; the raw-ID gate remains failed. The [Metal stochastic follow-up](docs/results/2026-11-15-phase4-xsum/mlx-stochastic-followup-v1.md) meets all reused frozen bounds with exact scorer/bootstrap replays; the original CUDA stochastic failure and Metal top-1 failure remain visible.

- **Owned XSum loader candidate:** the candidate releases validated codec bytes after upload. [Linux CUDA validation](docs/results/2026-11-15-phase4-xsum/cuda-owned-loader-v1.md) includes a full check, exact 11,334-row code/ID match to the previously scored Rust capture and repeated speed/RSS comparisons favoring Rust. Full `tools/check.sh --mlx` passed at `d018aa3` with 18 Metal cases; [the short candidate screen](docs/results/2026-11-15-phase4-xsum/mlx-owned-loader-screen-v1.md) still exceeds Python RSS by 50.25/10.53 MiB at batches 1/4. An intrusive heap snapshot found no live codec-sized allocation at the Rust sample but substantial resident empty large regions; stages differ, so causation is unproven. Full candidate quality/performance and memory remediation remain open. The fresh full CUDA check of the updated diagnostics passed, including 18 real-device operator cases (`reports/phase4-resumed-final-cuda-check-v1.log`). A sandbox check and capability-bearing check failed for environmental reasons; a later default check hit an observer fixture failure which passed in isolation. Retain all logs.

- **Borrowed upload experiment:** the [candidate](docs/results/2026-11-15-phase4-xsum/borrowed-upload-plan-v1.md) uses already-locked `bytemuck 1.25.2` for checked aligned little-endian views, with a copy fallback; shared codec and XSum generator host uploads change. Alignment/bit unit tests and full `tools/check.sh --cuda` pass (`reports/phase4-xsum-borrowed-upload-cuda-check-v2.log`); v1 stopped at a corrected Clippy style error. Actual Linux checked-artifact payloads were all aligned in one probe. Full Linux CUDA parity, quality and performance precede any Metal candidate run; no memory benefit is claimed yet. The earlier baseline's two Metal forced-history traces repeated byte-for-byte and their free-generation batches matched the original Metal results while CUDA ran; the exact gate remains failed.

- **Borrowed upload CUDA validation:** the [receipt-validated full 11,334-row comparison](docs/results/2026-11-15-phase4-xsum/borrowed-upload-plan-v1.md) to retained scored CUDA codes/IDs passes. Both real-CUDA and CPU Sudoku codec qualifications passed 642 traces with two near ties and zero ID disagreements. The completed repeated XSum CUDA matrix favors Rust versus pinned Python in median, pooled p95 and process RSS at batches 1/4/32; four other GPU processes were observed. Initial 32-row CUDA and Metal CLI checks accidentally used previous binaries because the combined build selected examples only; explicit rebuilds and corrected 32-row comparisons pass on both. Commit `f03650b` passed a full Metal check before the corrected CUDA CLI check and another afterward; the first short Metal screen is exploratory. The Metal raw-ID failure remains open.

- **Current Metal investigation:** corrected same-contiguous-batch CUDA diagnostics also match retained Python for rows 9813, 10009 and 10015; Metal still has six code differences across those rows with no ID differences in the two batches. [Same-history CUDA/Metal traces](docs/results/2026-11-15-phase4-xsum/mlx-full-top1-v1.md) find one in-band near-tie head crossing for each of those rows; forced final IDs agree across backends and with free Metal IDs. The [repeated Metal comparison](docs/results/2026-11-15-phase4-xsum/mlx-borrowed-upload-repeated-v1.md) favors Rust versus adapted Python MLX in median, pooled p95 and process RSS at all three batches; batch-32 variation and system-wide telemetry limits are disclosed. Ordered full Metal checks, the 32-row CLI and seed-42 replay pass, but candidate full-corpus quality remains pending and the existing exact raw-ID gate remains failed.

- **Candidate full Metal capture:** the [capture and receipt-validated comparison](docs/results/2026-11-15-phase4-xsum/mlx-borrowed-upload-full-quality-plan-v1.md) completed and matched all 11,334 retained Metal codes and raw IDs exactly. The retained baseline's EOS-filtered decoder inputs match Python, but its six post-EOS raw-ID differences still fail the frozen exact gate; no fresh Metal scorer was run. Candidate stochastic quality remains pending.

- **WMT14 artifact work in parallel:** the [explicit artifact profile](docs/results/2026-11-16-phase4-wmt14/artifact-plan-v1.md) preserves all 282 tensors; independent exports from different paths are byte-identical. Real full/inference-only CPU loading, profile-bound negative tests and explicit unsupported-model rejection pass. CPU-only codec and generator preflights validate the pinned configs and all 249 generation-path tensors; full CUDA regression and a 32-row XSum CLI regression pass after these changes. The experimental CUDA execution and its limits are recorded below; no WMT14 GPU quality is claimed.

- **WMT14 CUDA codec development:** two independent pinned upstream captures over two source-derived inputs produced byte-identical reports and 202-tensor payloads. An opt-in experimental Rust CUDA capture passes all 202 under the unchanged policy, with exact recorded codes and IDs; feature model tests, lint and full CUDA regression (18 device cases) pass. The [record](docs/results/2026-11-16-phase4-wmt14/codec-cuda-reference-plan-v1.md) gives commands, hashes and limits. The default WMT14 loader remains unsupported. Generator execution, full task quality, performance, Metal and Phase 4 completion remain pending.

- **WMT14 fixed-history teacher:** two pinned upstream CUDA captures over two source prefixes produced identical 26-tensor traces and reports. The experimental opt-in Rust CUDA teacher passes all 26 under unchanged bands and all 626 head choices, with exact recorded IDs and forced codes; feature tests/lint and the CUDA regression pass. The [record](docs/results/2026-11-16-phase4-wmt14/teacher-cuda-reference-v1.md) gives commands and limits. This is a bounded teacher fixture, not free generation, task quality, performance or Metal qualification.

- **WMT14 top-1 CUDA development:** two pinned upstream CUDA free-generation captures over two source prefixes are byte-identical; the opt-in Rust path matches all 26 tensors under unchanged bands and all 626 codes/head choices, the fixed-history teacher trace remains byte-identical, and full CUDA regression passes 18 real-device cases. The [record](docs/results/2026-11-16-phase4-wmt14/top1-cuda-reference-v1.md) includes hashes and limits. Exact head ties and nonfinite scores are rejected; CFG, full quality, performance and Metal remain pending.

- **WMT14 CFG CUDA development:** two upstream CUDA captures using explicit diagnostic CFG=1 and the evaluator's null prefix are byte-identical; the opt-in Rust comparison passes all 26 tensors and 626 guided choices. CFG-zero top-1 and teacher traces remain byte-identical, and full CUDA regression passes 18 device cases. The [record](docs/results/2026-11-16-phase4-wmt14/cfg-cuda-reference-v1.md) gives identities and limits. This two-prefix diagnostic is not task quality, performance or Metal evidence; the released configuration remains CFG=0.

- **WMT14 32-row CUDA screen:** the experimental output API's traced/untraced outputs match on the two-row fixture. The [four-at-a-time development screen](docs/results/2026-11-16-phase4-wmt14/cuda-32-row-screen-v1.md) matches all 10,016 upstream codes and 2,048 raw IDs; two Python captures repeat byte-for-byte, and full CUDA regression passes 18 real-device cases. This does not qualify full-corpus BLEU/chrF, performance or Metal.

- **WMT14 range capture preparation:** bounded Python/Rust CUDA tools match all outputs for 35 rows including a partial final batch and for a separate nonzero-start three-row smoke; the [record](docs/results/2026-11-16-phase4-wmt14/cuda-range-capture-v1.md) designates rows 0–127 as development before freezing quality criteria. The tools are not resumable. Held-out quality, base/CFG performance and Metal remain pending; the subsequently pinned scorer is described below.

- **WMT14 development quality:** all 128 development rows match in codes, raw IDs and decoded text for base and explicit CFG=1. Explicitly pinned sacreBLEU 2.5.1 and dependencies reproduce offline; [base](docs/results/2026-11-16-phase4-wmt14/cuda-development-quality-v1.md) and [CFG](docs/results/2026-11-16-phase4-wmt14/cuda-cfg-development-quality-v1.md) records give metrics and bootstrap intervals. The [held-out protocol](docs/results/2026-11-16-phase4-wmt14/heldout-protocol-v1.md) was defined before held-out evaluation. The CFG output API full CUDA regression passes 18 device cases. Full quality, performance and Metal remain pending.

- **WMT14 held-out tooling:** the source-bound CUDA capture supervisor and paired scorer pass focused admission and gate tests; 35-row base and CFG supervisor smokes match prior outputs and validate receipts. Full captures completed under the frozen rule below. Neither runner resumes interrupted captures.

- **WMT14 CUDA held-out quality:** both 3,003-row captures and checked scoring completed under the [frozen protocol](docs/results/2026-11-16-phase4-wmt14/heldout-protocol-v1.md). [Base passes](docs/results/2026-11-16-phase4-wmt14/cuda-heldout-quality-v1.md) with all codes and raw IDs exact and independently replayed metrics. CFG=1 has seven code and seven raw-ID differences at row 2181, all IDs after the common EOS; decoded text and metrics match, but the exact raw-ID gate **fails**. The [same-batch investigation](docs/results/2026-11-16-phase4-wmt14/cfg-row2181-investigation-v1.md) finds an in-band level-5 near-tie crossing; strict CUDA and reference rotary-table diagnostics do not fix it. Investigate before claiming CFG or WMT14 CUDA acceptance. Performance and Metal remain pending.

- **WMT14 reference-order candidate:** an explicit experimental CUDA backend narrows strict cuDNN and PyTorch-order normalization to the initial reconstruction. Eight exact normalization shape/batch checks and the original divergent four-row batch pass; full standard CUDA regression and cuDNN-feature release regression pass. Both full 3,003-row base/CFG candidate captures now match all Python codes and raw IDs; [the unchanged quality rules pass](docs/results/2026-11-16-phase4-wmt14/cuda-reference-candidate-retest-v1.md), with independently replayed CFG scoring. This is a retest on previously inspected data, not a fresh held-out study; the prior CFG failure remains documented. Performance and Metal remain pending.

- **XSum Metal near-tie diagnostic:** the [level-one plan](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) adds forced-history hidden/projection capture to investigate row 4937. CUDA and Metal captures reproduce their ten previous tensors exactly, and both full device checks pass 18 cases. The backends already differ in the level-one transformer output before the head projection; A [12-block follow-up](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md) reproduces all prior tensors, and CUDA/Metal checks pass 18 cases each. The prefix matches at transformer entry but target vectors already differ; prefix differences appear after block 0. A first-block stage trace reproduces all prior tensors on both devices; CUDA and Metal checks pass 18 cases (the CUDA rerun followed an unrelated observer-fixture failure). Identical prefix input first differs at layer normalization by at most 1.1921e-7; a causal fix remains unproven. The target-input follow-up reproduces all prior tensors and both device checks pass 18 cases. The first reconstruction already differs before resizing (maximum 1.9073e-6); the matching level-zero cascade traces reproduce all prior tensors and both device checks pass 18 cases. Codebook values are exact; the first upsampling convolution already differs, with maximum 2.3842e-6. The Metal raw-ID gate remains failed.

- **XSum upsample probe:** a [bounded Metal diagnostic](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) compares the first level-zero transposed convolution with equivalent FP32 matrix formulations on the captured batch. The flattened FP32 matmul matched CUDA exactly on the inspected row while the native convolution did not. The flattened formulation matched all 147,456 values in the inspected 32-row batch. An explicit experimental backend passes the full 18-case CUDA check; the full Metal check and layout regression pass. Forced row 4937 retains its level-1 crossing; its first convolution is exact, while the following group normalization differs in 1,575/4,608 values. PyTorch CUDA group normalization exactly reproduces the retained CUDA output on this row; two decomposed reductions do not. Row 4883 now matches CUDA raw IDs. Three additional forced histories show the head crossing removed for row 9813 but retained for 10009 and 10015; final IDs match on all three. Free-generation and full-quality validation remain pending, with no gate waiver.

- **Combined XSum Metal diagnostic:** a separate explicitly selected backend combines experimental normalization and first-upsample operations. The [Metal forced-history tests](docs/results/2026-11-15-phase4-xsum/metal-upsample-probe-v1.md) match CUDA head choices and final IDs for row 4883, but row 4937 retains its level-1 code crossing; three further histories retain head crossings, including row 9813 which upsampling alone had corrected. Final IDs match on all five forced histories. The default remains unchanged; the full Linux CUDA check passes 18 operator cases. Free-generation and quality/performance gates remain pending.

- **XSum normalization probe:** a [bounded Metal comparison](docs/results/2026-11-15-phase4-xsum/metal-layernorm-probe-v1.md) is prepared from 512 bit-identical CUDA/Metal prefix inputs and captured CUDA outputs. The standalone safe-math Welford probe matches more CUDA values than the existing fast norm but still differs in 30,191 values; it is not a production change or a gate fix. SIMD-checked safe math repeats that result; relaxed/fast modes match fewer values than the existing norm. An explicitly selected experimental Metal backend is being prepared for a forced-history test; the default backend remains unchanged. The full CUDA check passes 18 device cases; The first macOS check found a missing scoped unsafe allowance; after correction the full Metal check passed 18 cases. The forced-history candidate reproduces the isolated norm result but still selects the wrong code at row 4937 and worsens some differences; it remains experimental and the default is unchanged. A second forced-row test also leaves row 4883’s raw-ID mismatch unchanged.

- **WMT14 CUDA performance:** the [comparison plan](docs/results/2026-11-16-phase4-wmt14/cuda-performance-plan-v1.md) covers base and CFG, batches 1/4/16, synchronized timing, memory and output identity. The [full comparison](docs/results/2026-11-16-phase4-wmt14/cuda-performance-v1.md) completed with exact outputs across all 36 processes; desktop GPU activity, stage and capacity measurements remain limitations; token throughput has been added, so the complete CUDA gate and WMT14 Metal testing remain pending.

- **WMT14 deferred-validation candidate:** the [plan](docs/results/2026-11-16-phase4-wmt14/cuda-deferred-validation-plan-v1.md) defers generated-code sentinel rejection until concatenation using the existing bounds-safe embedding. A real-CUDA unique/tied/NaN sentinel regression and the full CUDA check pass; a 32-row smoke matches earlier outputs. Experimental-feature release tests pass; both 3,003-row replays are byte-identical to previously scored captures. Repeated timing found no meaningful gain and Rust remains slower at base batches 16/32, so immediate validation was restored byte-for-byte; the restored-path CUDA check passes 18 cases.

- **WMT14 cache investigation:** the [upstream KV-cache diagnostic plan](docs/results/2026-11-16-phase4-wmt14/cuda-kv-cache-probe-plan-v1.md) notes that the pinned cache path omits padding bias. The bounded unmodified CUDA cache run changed 7,505 codes and 1,005 raw IDs on 32 padded rows. A separate padding-mask-corrected diagnostic matches all codes/IDs on the 32-row sample and is substantially faster in one process. An explicit opt-in Rust candidate matches base/CFG 35-row smokes and a real-CUDA regression covering padded, unpadded, null and mixed-padding inputs. One batch-32 timing smoke is promising; the full CUDA check passes 18 cases. Feature-enabled tests and a complete forced-history band/index comparison pass. The [full cache retest](docs/results/2026-11-16-phase4-wmt14/cuda-cached-candidate-v1.md) passes base codes/IDs and independently replayed quality, but CFG fails exact raw IDs in rows 2181/2658. Repeated base timing favors cached Rust at all batches, while sampled GPU memory is higher at 16/32 and desktop activity remains. A base-only guard passes its targeted CUDA test; the first full check failed on a missing example feature declaration, which was corrected; the rerun passes 18 CUDA cases. The [explicit microbatch experiment](docs/results/2026-11-16-phase4-wmt14/cuda-cached-candidate-v1.md) matches outputs and is faster with lower sampled GPU/host memory at external batches 16/32, subject to desktop activity and observed variability; the explicit experimental host-output API passes full 3,003-row exact replay and independently replayed quality, and repeated batch-16/32 timing shows lower latency and sampled host/GPU memory than Python. Desktop activity and nonexclusive sampling remain limitations; a same-batch CFG trace reproduces row 2181’s in-band near tie; a same-batch row 2658 trace finds another level-9 near-tie crossing. An uncached Rust trace matches Python codes/IDs, while cached and uncached logits first drift at level 1 after bit-identical level 0. Explicit internal tracing locates the first drift in Q/K/V projection; shape-matched projections and attention make the first target block bit-identical but block 1 and final choices still differ. A prefix trace finds the first prefix drift at the first attention output projection; full CUDA checks and three artifact-backed cache tests pass after the follow-up instrumentation; no oracle change.

- **WMT14 explicit recomputation diagnostic:** an opt-in guided cache experiment recomputes a level with the uncached path when a cached top-two margin is at most `1e-4`. Both previously failing four-row batches, a separate 35-row smoke and a source-bound full 3,003-row replay match Python codes/IDs; the frozen full and held-out scorer passes. This is not a validated error bound or qualified fallback; repeated CUDA benchmarks favor Rust in latency and host RSS at batches 1/4/16, but sampled GPU memory is higher at batch 16 and desktop GPU activity remains; the full CUDA check passes 18 cases and the feature-specific unit/cache tests pass, and the ordinary CFG guard remains unchanged. See the [cache evidence](docs/results/2026-11-16-phase4-wmt14/cuda-cached-candidate-v1.md).

- **WMT14 CFG microbatch diagnostic:** an explicit recomputation-only host-output method schedules up to 16 external rows in groups of four. Four real-CUDA artifact tests and a 35-row exact Python smoke pass; full CUDA checks pass 18 cases. The initial full capture provisionally matched all rows but its comparator rejected different external batch sizes; the tested external-16/internal-4 comparator and source-bound rerun match all 3,003 rows and the frozen scorer passes. Repeated batch-16 timing favors Rust in latency, RSS and sampled GPU memory, with desktop activity and unseen-input heuristic limits. The margin heuristic and ordinary CFG guard remain unqualified. [Evidence](docs/results/2026-11-16-phase4-wmt14/cuda-cached-candidate-v1.md).

- **WMT14 uncached CFG microbatch:** the explicit [heuristic-free candidate](docs/results/2026-11-16-phase4-wmt14/cuda-uncached-cfg-microbatch-v1.md) passes real-CUDA chunk/invalid-input tests, all-feature lint and full CUDA regression (18 device cases). A clean-source replay matches all 3,003 Python rows and the frozen full/held-out scorer repeats exactly. Repeated batch-16 median/p95 is 1.291226/1.296859 s Rust versus 1.237539/1.240858 s Python, so the desired speed direction fails despite lower sampled GPU and host memory. The ordinary cached CFG guard remains; Metal and the phase gate remain open.

- **WMT14 uncached CFG eight-row diagnostic:** an explicit [larger grouping](docs/results/2026-11-16-phase4-wmt14/cuda-uncached-cfg-eight-v1.md) passes chunk tests and matches Python on 35 rows and both known difficult 16-row ranges; the four-row regression, all-feature lint and full CUDA check (18 device cases) pass. One batch-16 timing smoke is close to Python; full source-bound corpus, frozen scorer and repeated performance remain pending. This is a changed numerical execution shape, not a gate pass.

- **WMT14 CFG score audit:** an explicit observer captures raw cached guided scores before possible recomputation and compares them with uncached scores, output IDs and batch-level trigger flags. A four-row CUDA smoke, full 18-case CUDA check and four cache regressions pass; the source-bound full 3,003-row audit finds two raw argmax disagreements, both covered by recomputation, with all final IDs/codes matching Python. The largest score drift exceeds the trigger; no unseen-input bound is proven. [Evidence](docs/results/2026-11-16-phase4-wmt14/cuda-cached-candidate-v1.md).

- **WMT14 stage profile:** an [instrumented CUDA diagnostic](docs/results/2026-11-16-phase4-wmt14/cuda-stage-profile-v1.md) matches retained outputs and finds generator-level intervals dominate at base batch 32 and CFG batch 16. Finer transformer and matching Python profiles are pending; this is not a performance gate.

- **WMT14 capacity follow-up:** the [batch-32 plan](docs/results/2026-11-16-phase4-wmt14/cuda-performance-plan-v1.md) measures CFG=0 at the scoped backend’s maximum internal batch; CFG=1 is limited to external batch 16. The harness rejects unsupported CFG=1 batch 32. The [three-process comparison](docs/results/2026-11-16-phase4-wmt14/cuda-performance-v1.md) matched outputs and fit both runtimes; Rust was slower at the supported base limit, and desktop GPU activity remains a limitation.

- **Candidate Metal stochastic replay:** the [five-seed replay](docs/results/2026-11-15-phase4-xsum/mlx-borrowed-upload-stochastic-plan-v1.md) completed and receipt-validated all 10,240 rows against previously scored Metal captures with zero code or raw-ID differences; all copied files match remote hashes. This reuses scored inputs rather than a fresh scorer run or new held-out study. The known top-1 raw-ID failure and previously accepted CUDA stochastic exceptions remain visible.

- **Later-profile preparation:** [WMT14](docs/results/2026-11-16-phase4-wmt14/inventory-preparation-v1.md) checkpoint preflight passed, an experimental CPU export round-tripped all 282 tensors byte-for-byte, all 3,003 source encodings/padded prefixes and 6,006 synthetic EOS/PAD/special-token decodes match pinned Python, and a single upstream CPU output repeated byte-for-byte; its export is not a runtime artifact and inference classification remains provisional. [LM1B](docs/results/2026-11-16-phase4-lm1b/inventory-preparation-v1.md) CPU inventory, unigram checks and one upstream CPU sample cover all three model sizes; the small sample repeated byte-for-byte. All four config files were subsequently verified byte-for-byte against the existing checkpoint revision. Lossless CPU preparation exports now round-trip all tensors for all three sizes; a second small export is byte-identical and standard checks pass. A repeated fixed-code CPU probe found unchanged generator inputs, all 12 reconstruction levels and decoder logits after mutating 36 candidate EMA/hit tensors; this is bounded evidence, not a classification for all APIs; standard checks pass. These are not runtime artifacts, and LM1B execution and GPU gates remain open.

- **LM1B CUDA development:** five non-held-out seeds 11–15 each provide 256 pinned upstream batch-eight samples; per-seed FLM scores match upstream, one full capture/score repeat byte-for-byte, standard checks pass, and a hash-checked summary supports the pre-held-out [small-model protocol](docs/results/2026-11-16-phase4-lm1b/heldout-protocol-v1.md). No official or held-out quality or Rust parity is claimed. [Evidence](docs/results/2026-11-16-phase4-lm1b/inventory-preparation-v1.md).

- **LM1B-small artifact layout preparation:** two byte-identical CPU [exports](docs/results/2026-11-16-phase4-lm1b/small-inference-artifact-plan-v1.md) preserve all 351 tensors across generator, codec, retained training and inference bundles. The freshly built CLI explicitly rejects this preparation profile and standard checks pass; opt-in host-only preflights pass the real 213-generator, 100-codec and two-distribution files; all-feature tests/lint and full CUDA regression pass. Loader bounds, codec execution, model support and ordered GPU gates remain open.

- **LM1B-small bounded artifact:** the explicit [artifact profile](docs/results/2026-11-16-phase4-lm1b/small-runtime-artifact-v1.md) preserves all 351 tensors and has independent 448 MiB per-file/1024 MiB total limits. Actual full/inference-only CPU loads and opt-in host generator/codec preflights pass; normal model construction explicitly rejects before upload. Focused tests/lint and the full 18-case CUDA regression pass. This does not enable LM1B execution or advance prior profile gates.

- **LM1B-small codec reference:** two pinned upstream [CUDA traces](docs/results/2026-11-16-phase4-lm1b/small-codec-cuda-reference-v1.md) over two development-ID rows captured 362 boundaries and reproduced byte-for-byte. Existing diagnostic margins flag 27 encode and one forced-decode near ties; standard checks pass. The bounded Rust comparison is below.

- **LM1B-small experimental CUDA codec:** the [two-row Rust comparison](docs/results/2026-11-16-phase4-lm1b/small-codec-cuda-rust-v1.md) passes all 362 tensors with strict FP32 cuDNN, including exact codes and IDs; the initial Candle path failed 120 numerical comparisons. Shared architecture, bounded cuDNN shape and configuration regression tests were added; all-feature model tests/lint and full CUDA regression (18 device cases) pass. Normal loading remains unsupported; generator, quality, performance and Metal remain open.

- **LM1B-small fixed-history generator:** two pinned upstream [CUDA traces](docs/results/2026-11-16-phase4-lm1b/small-teacher-cuda-reference-v1.md) reproduce 51 boundaries byte-for-byte. The [experimental Rust comparison](docs/results/2026-11-16-phase4-lm1b/small-teacher-cuda-rust-v1.md) passes all 51 and 1,280 head choices under unchanged policy, with exact forced codes and greedy/penalized IDs; all-feature tests/lint and full CUDA regression (18 device cases) pass. This is two fixed histories, not sampling or quality.

- **LM1B-small greedy continuation:** two pinned upstream captures repeat byte-for-byte and the [experimental CUDA comparison](docs/results/2026-11-16-phase4-lm1b/small-greedy-cuda-v1.md) passes all 51 tensors, 1,280 head choices and exact generated codes/IDs on two development rows using supplied level-zero codes. The final-source greedy trace and previous teacher trace repeat exactly; input tests, all-feature model tests/lint and full CUDA regression (18 device cases) pass. This is not stochastic sampling, quality or a runtime gate.

- **LM1B-small controlled sampling:** an experimental seeded CPU inverse-CDF sampler draws the initial unigram and full later distributions; traced/untraced CUDA outputs match. The [two-row controlled comparison](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md) fails 21/51 tensors after a level-7 near-boundary choice cascades; same-logit replay matches all 1,282 Rust choices. PyTorch RNG equivalence and stochastic quality remain unproven. Input tests, all-feature model tests/lint and full CUDA regression (18 device cases) pass; normal loading remains unsupported.

- **LM1B-small Rust development quality:** five non-held-out seeds of 256 CUDA samples each were scored against pinned upstream FLM; [descriptive results](docs/results/2026-11-16-phase4-lm1b/small-rust-development-quality-v1.md) have Rust-minus-Python mean perplexity -0.301871 and entropy +0.00003997 nats. Seed-11 capture/scorer and the checked summary repeat byte-for-byte; seed-guard test, all-feature lint and isolated standard check pass. The first standard check failed one lifecycle test during concurrent work; its log is retained. This does not pass the official quality gate or resolve the controlled-draw failure; performance and Metal remain open.

- **LM1B-small device sampling:** the explicit [CUDA sampler candidate](docs/results/2026-11-16-phase4-lm1b/small-device-sampling-v1.md) matches CPU draws on 1,280 retained logits and exact codes/IDs across all five 256-sample development seeds; their existing FLM scores therefore apply. A small nonexclusive batch-eight screen favors device sampling, not a performance gate. Full CUDA regression (18 device cases), feature tests and lint pass; the Python controlled-draw failure and official quality/Metal gates remain open.

- **LM1B-small device penalty/performance:** the explicit [CUDA FP64 penalty candidate](docs/results/2026-11-16-phase4-lm1b/small-cuda-performance-screen-v1.md) matches CPU codes/IDs across five 256-sample development seeds and passes captured, tie and invalid-input probes. Three-process batch-eight screens show Python/Rust median 0.228202/0.195581 seconds with sampled GPU activity; the prior CPU-penalty Rust candidate was slower than Python. Feature tests/lint and full CUDA regression (18 device cases and XSum penalty test) pass. This is development evidence, not the release performance gate; controlled stochastic parity, held-out quality and Metal remain open.

- **LM1B-small capacity screen:** the [repeated CUDA comparison](docs/results/2026-11-16-phase4-lm1b/small-cuda-capacity-screen-v1.md) favors Rust in median and pooled p95 at batches 1/4/16; output signatures repeat within each runtime and sampled GPU memory is sparse. Evidence-validator tests, all-feature lint and the standard check pass. It is not cross-runtime output parity or an official performance/memory gate; desktop GPU activity and the controlled-draw failure remain.

- **LM1B BERT scorer decoding:** the [development comparison](docs/results/2026-11-16-phase4-lm1b/bert-decode-development-v1.md) matches 1,280 captured rows and eight synthetic cases against pinned Python after explicitly filtering the six model-valid IDs outside the tokenizer vocabulary. No development row contained those IDs; the generic Rust decoder rejects them. An eventual LM1B scorer must implement that bounded upstream behavior. This is not a model or quality gate.

- **LM1B metric comparison preparation:** a tested metric-only helper implements the frozen small-model seed bootstrap and threshold checks; standard checks pass, with strict seed/count/finite validation. It does not validate generation, scoring or identities and cannot alone establish a gate; no official samples were scored. [Evidence](docs/results/2026-11-16-phase4-lm1b/inventory-preparation-v1.md).

- **LM1B scorer preparation:** the [bounded FLM probe](docs/results/2026-11-16-phase4-lm1b/inventory-preparation-v1.md) matches pinned upstream GPT-2-large perplexity and raw-ID entropy exactly on three already-inspected CPU samples, with identical reports across three runs and standard checks passing. This is scorer development only; a repeated 18-row batch-16-plus-tail check also matches upstream and fits in two runs; standard checks pass; official distinct samples and held-out quality remain pending.

- **Machine state:** the user authorized stopping `llama-server.service`; it is inactive and should remain stopped until requested otherwise. Mac report artifacts are under ignored `reports/`. One transient new-SSH host-key failure resolved with the existing strict configuration; no keys were changed. Preserve unrelated `.pi/`.

- **Historical Codex update handoff:** the user requested a stop to update Codex and has since resumed. The [handoff](docs/results/2026-11-15-phase4-xsum/codex-update-handoff-v1.md) records the pushed XSum CUDA/Metal results, retained stochastic failures, small-batch Metal RSS regression, dependencies and safe resume steps. At that handoff no incomplete implementation remained; current work and `.pi/` are described above.

- **XSum CUDA advancement exceptions accepted:** the user explicitly chose to [accept the documented first stochastic quality-bound failure and two near-boundary controlled-draw differences](docs/results/2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md) and proceed to MLX. The independent larger follow-up passes its stricter bounds; the original failures remain visible. Released top-1 quality and repeated top-1/stochastic CUDA performance favor Rust within the cited scopes. Mac and Metal work subsequently proceeded; the current top-1 failure and pending gates are recorded above.

- **XSum first Metal teacher probe passed:** [two-prefix fixed-code comparison](docs/results/2026-11-15-phase4-xsum/mlx-teacher-v1.md) passed the existing accumulated band for all 6,799,360 floats and matched 400 level argmax decisions and 96 final penalized IDs against pinned Python CUDA. Linux full `tools/check.sh --cuda` passed first with 18 device operator cases (`reports/phase4-xsum-mlx-teacher-prep-linux-v1.log`); the real-Metal release example then built and ran at `c4e68ac`. This is a bounded fixed-history result, not generation, quality or performance. Implement explicit Metal selection and the CLI, validate on Linux before Mac, then qualify generated outputs and performance. The copied probe result is `reports/phase4-xsum-mlx-teacher-captured-v1/result.json`.

- **XSum Metal initial top-1 passed:** [two-synthetic-prefix results](docs/results/2026-11-15-phase4-xsum/mlx-top1-v1.md) show both artifact profiles match all 400 pinned-Python CUDA generator codes and 96 final IDs, with all 6,799,360 floats inside the existing band. The real-Metal tie/nonfinite selector test passed; the final penalty used the explicit CPU FP64 path. Linux full `tools/check.sh --cuda` passed first (`reports/phase4-xsum-mlx-top1-prep-linux-v1.log`, 18 CUDA cases). One SFTP brace-pattern copy failed; separate copies and hashes passed. This is not a real-passage, CLI, stochastic, task-quality, performance or memory gate; those remain pending.

- **XSum Metal CLI initial real-passage comparison passed:** [32-passage result](docs/results/2026-11-15-phase4-xsum/mlx-cli-real-v1.md) matches all 6,400 Python CUDA codes and 1,536 final IDs using the explicitly selected CPU-FP64 final penalty; at that commit positive CFG and stochastic CLI requests were rejected. Linux full `tools/check.sh --cuda` passed (18 CUDA cases), strict-cuDNN Clippy and tie regression passed, traced/untraced outputs matched for both artifact profiles, and rebuilt strict-CUDA CLI matched the same reference. Earlier failures exposed cached CUDA PTX missing the reference-tie symbol; regeneration corrected it and failed logs remain. This single Metal batch is not full quality or performance; Metal stochastic support, broader quality, device validation and repeated speed/memory remain pending.

- **XSum Metal seeded exact comparison failed:** [32-passage same-word diagnostic](docs/results/2026-11-15-phase4-xsum/mlx-stochastic-real-v1.md) differs from strict CUDA in 284/6,400 codes and 106/1,536 final IDs, confined to four code rows; later differences can cascade. The cause is not established. Strict CUDA CPU/device-final seeded paths matched all codes and IDs; Linux full `tools/check.sh --cuda` passed (18 CUDA cases), and strict-cuDNN backend/CLI Clippy passed before the Mac run. Compare both samplers on identical logits and words and compare model logits under forced shared histories; retain the failure. Positive CFG remains rejected. Metal stochastic quality and performance are unqualified.

- **XSum Metal stochastic diagnostic:** [eight-passage result](docs/results/2026-11-15-phase4-xsum/mlx-stochastic-diagnostic-v1.md) found one of 1,600 draws differs on identical Python logits and words, and two choices differ under forced histories while all 6,553,600 logits stay within the existing band; maximum conditional softmax total variation was 1.5605e-5. This separate trace does not explain every 32-passage divergence. Linux full `tools/check.sh --cuda` passed first (`reports/phase4-xsum-mlx-stochastic-diagnostic-linux-v1.log`, 18 CUDA cases); both Metal probes ran. Investigate sampler precision, preserve failures, and do not infer stochastic quality or performance.

- **XSum 32-passage shared-history diagnostic:** [result](docs/results/2026-11-15-phase4-xsum/mlx-shared-history-v1.md) reproduces three of four original first divergences when Metal samples identical CUDA logits and words; the fourth appears only with Metal forced-history logits. Across all 6,400 controlled positions, five same-logit and six forced-logit choices differ; all 26,214,400 logits remain in the existing band. This identifies contributions at these positions, not a universal cause or stochastic-quality pass. Linux full `tools/check.sh --cuda` and strict diagnostic Clippy passed before Metal; the 104,858,504-byte CUDA trace and result are retained. Investigate sampler accumulation precision while preserving all failures.

- **XSum fixed-point sampler experiment:** [result](docs/results/2026-11-15-phase4-xsum/mlx-fixedpoint-sampler-v1.md) reduced 6,400 same-logit differences versus CUDA from five to one; an independent FP64 inverse CDF matched all 6,400 experimental outputs, versus four prior-Metal and one CUDA mismatch. CUDA resampling reproduced all saved CUDA choices. This is an isolated fixture, not production or a universal equivalence/quality claim; FP32 exponentials and truncation below 2^-48 remain. Linux full `tools/check.sh --cuda` preceded the Metal experiment and strict-CUDA resampling Clippy passed. Investigate a narrowly scoped XSum implementation with boundary tests and Linux-then-Metal quality/performance validation; retain prior failures and preserve TinyStories sampling.

- **XSum-only fixed-point Metal candidate screened:** [32-passage result](docs/results/2026-11-15-phase4-xsum/mlx-fixedpoint-candidate-v1.md) passed three real-Metal boundary tests and reduced free-generation mismatches from 284 to 102 codes and 106 to 51 IDs, but exact comparison still fails and a new row diverges. Shared-history analysis finds one same-logit difference versus CUDA (where independent FP64 agrees with Metal) and three forced-logit differences; all 26,214,400 logits remain in band. Linux full CUDA and strict lint passed first; full `tools/check.sh --mlx` then passed with 18 real-Metal cases. TinyStories keeps its existing sampler. Retain both failed comparisons and evaluate stochastic quality, performance and memory before deciding whether to keep the candidate.

- **XSum Python MLX reference initially qualified:** [32-passage comparison](docs/results/2026-11-15-phase4-xsum/python-mlx-reference-v1.md) matched all 6,400 pinned-Python CUDA codes and 1,536 IDs; the independent NumPy FP64 penalty selector also matched upstream fixtures. First Mac adapter attempt failed before inference because resolving the virtualenv symlink bypassed its environment; corrected retry passed. First Linux full check failed an unrelated process-observer fixture; isolated retry and full `tools/check.sh --cuda` rerun passed, including 18 CUDA cases. The adapted reference is qualified only for the bounded top-1 screen; run the prepared short Rust/Python Metal comparisons, then broader quality and repeated performance/memory qualification.

- **XSum Metal short performance screen:** [one-process results](docs/results/2026-11-15-phase4-xsum/mlx-performance-screen-v1.md) have matching Rust/Python top-1 outputs and faster Rust medians at batches 1/4/32, but Rust peak process RSS is 57.50 MiB and 13.65 MiB higher at batches 1 and 4; batch 32 favors Rust for both. Rust MLX peak is lower, but its scope differs from RSS. These are short, non-gating measurements with activity limitations. Investigate the small-batch RSS, then rerun controlled repeated performance and full task quality; do not claim the Metal gate passed.

- **XSum stochastic held-out quality failed:** [five-seed frozen comparison](docs/results/2026-11-15-phase4-xsum/stochastic-heldout-v1.md) passed its perplexity bound but missed the precommitted uncertainty limits for ROUGE-1/2/L and mean output length. The observed mean differences are inside the margins; uncertainty is too large to establish the required rule, which is not proof of a quality regression. A [post-hoc decomposition](docs/results/2026-11-15-phase4-xsum/stochastic-uncertainty-v1.md) finds contributions from both inputs and sampling seeds; it is not a pass or an identified model defect. Preserve the failure; the user later accepted it for MLX advancement as recorded above. No numerical or quality margin has changed. The [development protocol](docs/results/2026-11-15-phase4-xsum/stochastic-development-v1.md) and raw five-seed captures are retained under `reports/phase4-xsum-stochastic-qualification-*`. [Repeated stochastic performance](docs/results/2026-11-15-phase4-xsum/stochastic-performance-v2.md) favors Rust in median/p95 latency, RSS and sampled driver memory at batches 1/4/32; an earlier partial run was stopped due to overlapping analysis. Focused integrity, bootstrap-regression and activity-summary checks pass.

- **XSum independent follow-up passed its own bounds:** [2,048-row five-seed comparison](docs/results/2026-11-15-phase4-xsum/stochastic-followup-result-v1.md) met all original quality margins using stricter 97.5% degradation bounds. This additional look was selected after the original failure and does not erase it; the controlled-draw near-boundary differences also remain. Raw captures and scored rows are in `reports/phase4-xsum-stochastic-followup-{python,rust}-{seed}*`; focused integrity and syntax checks pass. The user subsequently accepted these exceptions for MLX advancement as recorded above.

- **TinyStories exceptions accepted for advancement:** the user explicitly chose to proceed to XSum with the [numerical and historical latency exceptions recorded](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md). [Diagnostic plan](docs/results/2026-11-15-phase4-tinystories/mlx-cross-runtime-investigation-plan-v1.md) prepares a bounded CPU comparison of retained Python MLX generator logits directly against pinned Python CUDA; synthetic correctness and syntax checks pass. [Direct Python MLX versus pinned Python CUDA](docs/results/2026-11-15-phase4-tinystories/mlx-python-cross-runtime-v1.md) has 1,124 outside-band values versus Rust Metal's 1,131 across 293.6 million generator values, with no checked Python argmax differences; counts do not prove identical positions or identify the responsible operation. [Independent MLX reconstruction](docs/results/2026-11-15-phase4-tinystories/mlx-cascade-independent-v1.md) shows first-level differences and matches Rust's per-level outside-band counts across all five seeds; [same-input stage comparison](docs/results/2026-11-15-phase4-tinystories/mlx-cascade-stages-v1.md) finds all four upsampling stages bit-identical but the residual convolution differs; [FP64 comparison](docs/results/2026-11-15-phase4-tinystories/mlx-residual-fp64-v1.md) finds Metal closer on more values but worse worst-case error, and the CUDA oracle itself outside the old band on 7,697 values. A [sampled reduction-order screen](docs/results/2026-11-15-phase4-tinystories/residual-reduction-screen-v1.md) found no simple tested FMA schedule close to bit-identical; it does not rule out other algorithms. A [same-input MLX convolution-formulation screen](docs/results/2026-11-15-phase4-tinystories/mlx-residual-variants-screen-v1.md) found three matrix multiplies reduced blended outside-band values 15,618→10,654, improved FP64 error and was faster in one operator screen; an [isolated first-refinement Rust candidate](docs/results/2026-11-15-phase4-tinystories/mlx-first-residual-three-gemm-screen-v1.md) passed the operator check but regressed full-model diagnostics to 117,505 upstream outside-band values, one changed head argmax and three changed codec codes; it is rejected. [Batch-shape and Rust/Python follow-up](docs/results/2026-11-15-phase4-tinystories/mlx-first-residual-shape-investigation-v1.md) found equal outside-band counts for the first 16 rows at both MLX batch shapes, and byte-identical Rust and Python three-matrix outputs on that input. The CUDA reference was captured at batch 128; this does not establish behavior for all inputs or explain the full-model sensitivity. [A same-position overlap diagnostic](docs/results/2026-11-15-phase4-tinystories/mlx-outlier-overlap-v1.md) compares all five retained seeds: 1,113 CUDA-relative outside-band positions are shared, 18 are Rust-only and 11 are Python-MLX-only; prior per-tensor marginal counts agree. A historical comparison in the same report finds 1,132 outliers before Q/K fusion versus 1,131 after; individual historical positions were not captured. A [bounded detail probe and CPU validation](docs/results/2026-11-15-phase4-tinystories/mlx-rust-only-outliers-v1.md) measured the 18 Rust-only crossings: absolute error is 1.000036–1.060243 times the existing band; the largest excess is 4.265e-5. All remain band failures. This supports a shared cross-runtime contribution, but is not a numerical pass or a proven cause. No tolerance changed. Historical Rust and later Python slow episodes have similar plateau/recovery shapes, but recorded CPU windows and absent GPU counters do not establish a cause. [Python request-aligned GPU diagnostic](docs/results/2026-11-15-phase4-tinystories/mlx-python-tail-gpu-v1.md) reproduced eight slow requests: 34 plateau GPU snapshots reported 39% device utilization; approximately aligned `ps` showed low Python CPU, with differing global memory counters during recovery. Host gaps were sub-millisecond; global, uncalibrated counters do not identify the cause. An [opt-in process-local Python cache diagnostic](docs/results/2026-11-15-phase4-tinystories/mlx-python-cache-diagnostic-v1.md) reproduced ten slow requests and the 39% plateau with a 4 GiB free-cache limit. A large uncapped Python cache is not necessary for this observed plateau; root cause and the earlier Rust event remain unresolved. The exceptions remain visible failures and are accepted only for advancing; XSum starts with Linux/CUDA and requires its own gates.

- **XSum restricted generation:** [source-bound CUDA comparison](docs/results/2026-11-15-phase4-xsum/cuda-top1-v1.md) matches pinned Python on all 400 generated codes and 96 penalized IDs for both two synthetic padded inputs and a separate two-passage development sample, with all 6.8 million float values within the existing diagnostic band in each. The temperature-1/top-k-1/top-p-0/CFG-0 path uses explicit native CUDA unique-maximum selection and bounded CPU FP64 final selection; exact head ties and unsupported backends fail explicitly. The full `tools/check.sh --cuda` passed on the later CPU selection source, including the selector and 18 operator cases. Later device-selection, performance, quality development and CLI-preparation evidence is recorded below; CFG and tie requirements remain open, and Linux/CUDA must complete before MLX.

- **XSum performance investigation:** [short real-input screens](docs/results/2026-11-15-phase4-xsum/cuda-perf-screen-v1.md) found Rust lower in whole-process RSS but slower than Python at batches 1/4. Same-order FP64 CPU selection and bounded row parallelism reduced the cost while preserving tested outputs. The initial tiled CUDA prototype and slower single-block version are retained there; later integration and the remaining batch-4 failure are recorded below.

- **XSum device FP64 selection:** [CUDA implementation and repeated performance](docs/results/2026-11-15-phase4-xsum/cuda-device-selection-v1.md) preserve checked outputs and reduce readback; the full `tools/check.sh --cuda` passed. Rust median/p95 beats Python at batch 1 and uses less whole-process RSS at both batches; batch 4 remains slower and fails the speed requirement. The longer run has activity and device-memory limitations, and larger batches, official quality, CFG and head ties remain open. Deferred-synchronization and fused-head experiments did not improve their short screens and were reverted. Continue diagnosing Linux model execution before any XSum MLX run.

- **XSum level timing:** [instrumented batch-4 diagnostic](docs/results/2026-11-15-phase4-xsum/cuda-level-diagnostic-v1.md) found the event-timed gap spread across all nine levels, not one isolated level; synchronized timings were materially perturbed. This does not establish a cause or replace the failed performance comparison. Next inspect common per-block execution and allocator/device-memory scope.

- **XSum fused long attention:** [CUDA implementation and repeated results](docs/results/2026-11-15-phase4-xsum/cuda-fused-attention-v1.md) match tested existing softmax/attention outputs and saved Python model cases; the full `tools/check.sh --cuda` passed on rerun after a retained CPU observer fixture failure and isolated pass. At batches 1/4, Rust median and p95 are about 16% lower than Python's and peak RSS is about 43% lower; device pool and Python allocator scopes are separately disclosed. A short official-batch-32 screen favors Rust with matching outputs. These are not full XSum acceptance: repeated batch 32, ROUGE, broader parity, CFG and head ties remain open, as does XSum MLX.

- **XSum quality evaluator preparation:** [explicit ROUGE dependencies and GPT-2 evaluator pin](docs/results/2026-11-15-phase4-xsum/quality-evaluator-preparation-v1.md) are installed locally and pass offline CPU smoke checks. The later development baseline and proposed held-out rule are recorded below.

- **XSum quality development:** the [256-row Python baseline and frozen held-out rule](docs/results/2026-11-15-phase4-xsum/quality-development-protocol-v1.md) record two identical upstream seeds, ROUGE and GPT-2 perplexity with direct upstream scorer agreement, and an exact-ID/zero-quality-drop rule established before held-out comparison. A two-batch upstream CUDA restart reproduced uninterrupted output byte-for-byte. After a retained linker-path failure and lint fix, the Rust CUDA batch runner matched all 51,200 codes, 12,288 IDs and scored metrics on 256 development rows. The full `tools/check.sh --cuda` passed with 18 real-device cases before the example's serialization-only lint fix; targeted lint, rebuild and revised CUDA development run passed afterward. The scorer's explicit row-range output uses documented schema 2 and matches its schema-1 development metrics. Next run the held-out and official full CUDA quality comparisons; CFG, tie behavior and broader performance reporting remain open. No held-out quality comparison or XSum MLX gate is claimed.

- **XSum v2 CUDA quality comparison passed:** [tie-enabled full and held-out quality](docs/results/2026-11-15-phase4-xsum/cuda-reference-ties-v1.md) match Python on all raw IDs, texts, per-row ROUGE and aggregate ROUGE/GPT-2 perplexity. The one intermediate-code difference is the same investigated exact Python maximum tie. Current-source repeated performance, broader CFG and MLX remain pending.

- **XSum v2 repeated CUDA performance:** [tie-enabled three-process comparisons](docs/results/2026-11-15-phase4-xsum/cuda-reference-tie-repeated-v1.md) match outputs and favor Rust versus Python in latency, process RSS and sampled driver GPU memory at batches 1/4/32, but Rust latency rose 7–18% from the prior run; Python also slowed, and desktop activity was observed. An alternating selector microbenchmark and direct three-pair full-model control with byte-identical old/current binaries found no reproducible tie-change slowdown (median paired changes −0.02%, −0.15%, +0.03%); the original occupied-run cause remains unknown. Broader CFG, stochastic qualification and MLX remain pending.

- **XSum CUDA CFG:** [restricted top-1 guidance](docs/results/2026-11-15-phase4-xsum/cuda-cfg-v1.md) follows the upstream null-prefix and level-dependent score rule. Two synthetic comparisons at scales 1 and 2 and [eight real-passage comparisons at scale 1](docs/results/2026-11-15-phase4-xsum/cuda-cfg-real-v1.md) match all codes and IDs on CUDA; full local CUDA checks pass. Short positive-CFG speed and RSS screens favor Rust at batches 1/4, without activity telemetry. Full positive-CFG quality, broader guidance and tie coverage, and MLX remain pending.

- **XSum experimental stochastic sampling:** [controlled-draw investigation](docs/results/2026-11-15-phase4-xsum/cuda-stochastic-investigation-v1.md) records failed exact full-model comparisons, 1,600/1,600 matching draws on identical Python logits, and two forced-code near-boundary differences traced to differing model logits. All 6,553,600 forced-code logits remain within the existing diagnostic band; maximum per-position probability total variation was 5.47e-6 over 1,600 positions. One-process stochastic speed/RSS screens favor Rust at batches 1/4. [32- and 256-row development quality screens](docs/results/2026-11-15-phase4-xsum/cuda-stochastic-quality-screen-v1.md) use different random streams; stochastic quality is not qualified. Current-source full/held-out top-1 quality passes; a paired control did not reproduce the earlier tie-change performance slowdown. User explicitly chose full stochastic qualification before MLX: define and freeze a justified uncertainty-aware quality protocol from Python development data, then perform held-out CUDA quality and required performance comparisons. No exception has been accepted for XSum.

- **XSum CUDA ties:** the [reference-profile tie path](docs/results/2026-11-15-phase4-xsum/cuda-reference-ties-v1.md) selects the lowest finite maximum, matching 1,024 captured pinned-PyTorch CUDA decisions and the inspected 4096-way stable sort. The older profile still rejects ties and all profiles reject non-finite logits. Full CUDA checks and full/held-out quality pass. Current-source repeated performance favors Rust versus Python; its slowdown against the previous run did not reproduce in the direct paired control. Stochastic qualification and MLX remain pending. No general PyTorch sort guarantee or XSum exception is claimed.

- **XSum Mac preparation:** user confirmed Mac availability. The clean `~/projects/lexlapax/ncm-rs` checkout was fast-forwarded to pushed `902344f`; two long-running Python processes were idle when inspected, and `.reference/mlx-venv/bin/python` exists. Newly created `artifacts/canonical/xsum-phase4-v2/` and `artifacts/canonical/text-phase4-v1/xsum.json` were copied without replacing existing files; all nine artifact files, manifest SHA-256 `5df37749e64418c586472a4678600603992357b9bc5e3611da307918e13b237e`, and dataset SHA-256 `cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a` verified on Mac. The alternate `xsum-inference-rope-v2-final` artifact was prepared from copy-on-write clones of identical files and its three version-specific files, with all nine files and manifest SHA-256 `8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4` verified. The first verifier failed because system Python 3.9 lacks `hashlib.file_digest`; installed `/opt/homebrew/bin/python3` 3.15 succeeded. No XSum MLX run has started. Current-source Linux full/held-out quality passes, and a paired control did not reproduce the tie-change slowdown. Stochastic qualification remains open; resolve the scoped Linux gate before the first MLX run, recheck Mac state and availability for competing work, and announce that run.

- **XSum v2 full CUDA capture complete:** [quality checkpoint](docs/results/2026-11-15-phase4-xsum/native-full-progress-v1.md) records all 11,334 rows and 544,032 raw IDs matching Python, with one intermediate-code difference. A real-CUDA replay reproduced both outputs: Python had an exact maximum tie at row 2,690, while Rust separated the candidates by one FP32 step; final IDs match. ROUGE/perplexity scoring and the frozen-rule comparison subsequently passed as described above; repeated performance, remaining feature requirements and MLX remain pending.

- **XSum native reference-order CUDA candidate:** [implementation and paired screens](docs/results/2026-11-15-phase4-xsum/cuda-reference-native-v1.md) retain fast v1 behavior and implement v2 normalization, GELU, rotary and projection arithmetic without Python. The full local `tools/check.sh --cuda` passed with 18 real-device cases, and the separate pinned CUDA operator fixture passes; a source- and library-bound 128-row smoke and a separate 259-row screen (including the formerly failing batch) match all Python codes and IDs. One-process paired batch 1/4/32 screens show lower Rust median latency and peak RSS; these are not the repeated performance gate. The later full quality pass and remaining work are described above.

- **XSum opt-in rotary artifact:** [experimental v2 profile](docs/results/2026-11-15-phase4-xsum/rotary-artifact-profile-v2.md) validates pinned rotary tensors and builds XSum geometry from them without Python at inference. Existing v1 remains available. A 32-row CUDA screen matched Python, but the previously failed batch still changes three codes and five IDs; an unpaired short v1/v2 screen recorded a small latency/RSS increase. The full local `tools/check.sh --cuda` passed, including 18 real-device operator cases; final v1/v2 artifact preflight and targeted exporter tests pass. That table-only measurement predates the native arithmetic described above; full CUDA/MLX model validation remains pending.

- **XSum rotary reference preparation:** [offline table preparation](docs/results/2026-11-15-phase4-xsum/rotary-reference-preparation-v1.md) verifies the pinned upstream source, Torch version and exact table hashes, and produced two byte-identical native-readable reference payloads. Targeted tests pass. This is a separately versioned reference rather than a quality gate; the subsequent opt-in artifact profile is described above.

- **XSum near-tie investigation:** [same-input operator and full-batch diagnostics](docs/results/2026-11-15-phase4-xsum/cuda-near-tie-investigation-v1.md) locate differences in normalization, rotary tables/arithmetic and attention; reference-order diagnostic operators match captured Python inputs, but the layer/RMS/attention full-batch candidate changes three codes and five IDs. An initial combined rotary build failed to link and its attempted run reused the prior executable; a successful rebuild still changes three codes and five IDs. Its first attention projection differed because cuBLASLt selected a different algorithm; a shape-specific preference plus a reference-order GELU made all 12 first-level transformer blocks bit-identical, but the full batch still changes three codes and five IDs. The next-level target first differed in cascade group normalization; an upstream-order diagnostic plus the other captured-reference changes matches all codes and IDs in 256 sampled rows, including the failed batch, with original attention and captured rotary tables; the failed batch also matches with system cuBLAS. A later generalized-batch diagnostic also matched those rows plus the three-row final batch (259 total); captured rotary tables and specialized kernels remain. Reduced variants and a native reciprocal-order rotary attempt failed; a short unpaired screen is not a performance gate. The native implementation is now described above; validate full quality, speed and memory without changing the frozen rule. Linux quality and MLX remain pending.

- **XSum full CUDA quality stopped on a numerical head tie:** [pinned-Python seed-42 reference generation](docs/results/2026-11-15-phase4-xsum/python-full-reference-v1.md) completed and validated all 11,334 rows. The [source-bound Rust run](docs/results/2026-11-15-phase4-xsum/rust-full-progress-v1.md) matched all codes and IDs for the completed 8,131 rows, but shard 1 batch 77 failed with the explicit tied-or-nonfinite head rejection. A targeted real-CUDA trace at dataset row 4,937, level 1 position 4 found Rust's maximum exactly tied at codes 3,182/3,690; Python's corresponding scores differed by one FP32 step and selected 3,182. Source and binary fingerprints remain unchanged. Investigation is under `reports/phase4-xsum-tie-probe-v1/`; preserve the failure and partial outputs, do not guess a tie break or change the frozen quality rule. Full Rust generation, held-out scoring and official metrics remain pending.

- **XSum quality reproducibility:** the development protocol now includes the recorded generation, scoring, and bootstrap commands; its frozen rule and measured results are unchanged.

- **XSum Python full-reference metrics:** [the completed reference capture](docs/results/2026-11-15-phase4-xsum/python-full-reference-v1.md) now includes CPU-scored full-set ROUGE-1/2/L 34.9236/12.7078/27.9258 and mean 20.6573 words, plus held-out ROUGE and length. Full-set perplexity and all Rust held-out quality comparisons are still pending.

- **XSum capacity preparation:** [the exploratory timing bounds, source-verified 128-row fixture and sequential screen runner](docs/results/2026-11-15-phase4-xsum/cuda-capacity-preparation-v1.md) are prepared for a later real-device larger-batch screen. Formatting, Python syntax, targeted Rust lint and release build pass; no batch above 32 or capacity-runner CUDA execution has been claimed. Do not interrupt the current quality run for this screen.

- **XSum structure review:** [the XSum-specific follow-up](docs/results/2026-11-15-phase4-xsum/rust-structure-review-v1.md) supplements the earlier workspace-wide declaration audit with model, configuration, result and CUDA operation ownership and logical tensor sizes. A possible duplicate code readback and stale comments are recorded for after source-bound quality scoring; no unmeasured optimization was made.

- **XSum CLI preparation:** [the restricted CUDA-cuDNN top-1 command](docs/results/2026-11-15-phase4-xsum/cli-preparation-v1.md) builds, passes CPU input tests and targeted lint, and a release-binary CPU request exits with the explicit no-fallback error. A source-bound single-batch CUDA comparison helper is prepared. Real-CUDA CLI output and end-to-end performance/memory are pending after the current full quality run; do not treat the command as a qualified XSum gate.

- **XSum Python memory and repeated performance:** the first repeated batch-32 comparison failed when Python exhausted GPU memory after the Rust process completed. [Investigation](docs/results/2026-11-15-phase4-xsum/b32-python-memory-v1.md) found roughly 187 MiB of unreachable GPU-backed objects per batch, released by explicit garbage collection. The reference harness times cleanup separately and also reports cleanup-inclusive cost; no model math changed. Repeated [batch 32](docs/results/2026-11-15-phase4-xsum/cuda-b32-gc-comparison-v1.md) and [batches 1/4](docs/results/2026-11-15-phase4-xsum/cuda-small-batch-gc-comparison-v1.md) completed with identical codes/IDs. Rust/Python median generation times were 160.49/190.45, 580.50/691.66 and 4,500.39/5,336.60 ms at batches 1/4/32, and median whole-process RSS was about 1.04/1.82 GiB at each. GPU activity and differing allocator scopes are disclosed; earlier no-cleanup Python allocator peaks remain provisional. A short upstream Python quality-generator smoke is in progress; real-CUDA restart and scoring remain pending.

- **XSum teacher trace:** fixed-code execution with padded prefixes, nine tied heads and reconstruction agrees with pinned Python CUDA across 6.8 million values and all checked choices on two synthetic rows; [source-bound evidence](docs/results/2026-11-15-phase4-xsum/cuda-teacher-v1.md) and full local CUDA checks pass.

- **XSum generator host preflight:** strict pinned configuration, all 192 parameter shapes, nine head temperatures, level/mask buffers and token allowance are checked before device allocation; the actual v2 checkpoint and negative schema/buffer tests pass. [Evidence](docs/results/2026-11-15-phase4-xsum/generator-preflight-v1.md) retains an unrelated CPU process-observer fixture failure, successful isolated retry and successful full local rerun.

- **XSum codec first level:** explicit four-position XSum validation and corrected first-level cascade geometry match two small pinned Python CUDA inputs within the existing component band; the source-bound replay and full `tools/check.sh --cuda` pass, including 18 real-device cases. [Evidence and scope](docs/results/2026-11-15-phase4-xsum/cuda-cascade-v1.md). This is not full codec/generator parity or the XSum Linux gate; padded-prefix masking, nine tied heads, selection integration, quality and performance remain.

- **XSum explicit FP64 selection:** a native, explicit CPU final-token selector follows the pinned XSum penalties and matches an exact upstream CPU/CUDA oracle on three small varied, tie and repeated-history cases; the default local check passes. [Scope](docs/results/2026-11-15-phase4-xsum/fp64-selection-v1.md). The restricted generation path now uses it with bounded row chunks; broader correctness, real workload performance/memory and MLX are pending.

- **XSum artifact boundary:** [pinned XSum export and embedded GPT-2 checks](docs/results/2026-11-15-phase4-xsum/artifact-export-v1.md) now preserve 194 generator, 53 codec, 27 training and one allowance tensor in an explicitly bounded profile. Actual artifact inspection and tokenizer fixtures pass on Linux; the subsequent limited CUDA codec cascade is recorded above. Full XSum model inference and profile gates remain pending, Linux/CUDA before MLX.

- **CPU handoff fixture follow-up:** the fake worker and workload now use distinct executable inodes, avoiding one source of observer identity ambiguity without relaxing production mismatch checks. Focused checks and three repeats of the earlier failing case pass; `RUSTC_WRAPPER= tools/check.sh` passes 493 report tests. The earlier default-wrapper attempt failed in sandboxed `sccache`. The [integration record](docs/results/2026-11-15-phase4-tinystories/generator-qk-rope-integration-v1.md) retains both limitations; the exact original exec race remains unproven.

- **Q/K rotary production integration:** the prepared production source matches the measured guarded candidate except for a module comment. The [integration record](docs/results/2026-11-15-phase4-tinystories/generator-qk-rope-integration-v1.md) records full CUDA checkout validation, the first failed CPU observer fixture and successful unchanged rerun, and the inherited quality/performance evidence. The pushed production checkout passed full MLX validation (18 Metal cases), matched all five scored files, and closely matched the measured isolated binary in a one-process timing screen. The upstream cross-runtime float diagnostic and old tail remain documented. Broader Phase 4 profiles are not complete.

- **Experimental Q/K RMS and RoPE fusion:** [plan and Linux checks](docs/results/2026-11-15-phase4-tinystories/mlx-generator-qk-rope-plan-v1.md); [Metal numerical, quality and old/new performance screen](docs/results/2026-11-15-phase4-tinystories/mlx-generator-qk-rope-screen-v1.md). Focused CUDA checks and same-device 80-row Metal diagnostics pass; original upstream CUDA float-band exceptions remain. Five-seed nonblind stochastic quality passes, although 23/128 seed-42 texts differ. One-process batch-32/128 median improved 4.6%/7.5% versus the prior Rust candidate, with slightly higher process RSS. [Three-process batch-128 Python comparison](docs/results/2026-11-15-phase4-tinystories/mlx-generator-qk-rope-python-b128-v1.md) favored Rust in median/p95 and measured RSS/MLX peak; Python exhibited the large tail in this run. Full CUDA then Metal checks pass (18 operator cases each); [batch-1/4/32 comparisons](docs/results/2026-11-15-phase4-tinystories/mlx-generator-qk-rope-python-small-v1.md) also favor Rust for measured latency and memory. A [guarded width-portability follow-up](docs/results/2026-11-15-phase4-tinystories/mlx-generator-qk-rope-portable-plan-v1.md) passes focused CUDA checks; its first Metal build failed on a misplaced unsafe allowance, now corrected in a replayed follow-up that again passes focused CUDA. Its [corrected Metal screen](docs/results/2026-11-15-phase4-tinystories/mlx-generator-qk-rope-portable-screen-v2.md) passes forced scalar and compiler-guard tests; five quality files and ten fixed-code summaries match the scored predecessor, and an old/new screen shows similar batch-128 performance. Full CUDA then Metal checks pass on the corrected source (18 operator cases each). [Repeated Python comparisons on the corrected source](docs/results/2026-11-15-phase4-tinystories/mlx-generator-qk-rope-portable-python-v2.md) favor Rust at all four batches in measured median/p95 and RSS/MLX peak; Python again showed the large batch-128 tail. Actual non-32-wide hardware behavior remains open; prior tail failure is retained.

- **Phase 4 active — XSum in progress:** Linux/CUDA precedes MLX per profile; [contract](docs/PHASE4-TEXT.md). The sentinel-safe strict-cuDNN TinyStories candidate passed the full CUDA check, fixed-code numerical diagnostics, and five byte-identical previously scored quality files. Its [source-bound comparison](docs/results/2026-11-15-phase4-tinystories/cuda-safe-split-performance-v1.md) found Rust faster and lower in process RSS at all four batches, with disclosed activity and allocator-scope limitations. The default CUDA path is not covered by the strict numerical result. XSum/WMT14 evaluation data at the existing dataset revision were hash-verified and converted to ignored non-executable JSON; the XSum work and open gate are recorded above. OWT's released checkpoint remains unavailable.

- **Generator Metal GELU experiment deferred:** an [isolated candidate](docs/results/2026-11-15-phase4-tinystories/mlx-generator-gelu-plan-v1.md) attempted to fuse the existing exact-erf expression for large width-3072 activations. Formatting, patch replay and focused CUDA Sudoku golden and strict-cuDNN TinyStories output checks passed. The first Metal build failed on a grid argument type mismatch; after checked conversion, the focused Metal test failed because its custom source did not define MLX's `erf`. The handwritten candidate is not integrated; consider the previously studied shape-independent compiler for the unchanged expression instead. FP32 behavior, both full gates and any tail improvement remain unproven.

- **Compiled generator GELU experiment:** an [isolated replacement](docs/results/2026-11-15-phase4-tinystories/mlx-generator-compiled-gelu-plan-v1.md) uses MLX shape-independent compilation of the unchanged exact-erf expression for large width-3072 inputs. Formatting, patch replay, focused CUDA Sudoku and strict-cuDNN TinyStories checks, and full `tools/check.sh --cuda` then `--mlx` pass with 18 real-device operator cases each. Its [screen](docs/results/2026-11-15-phase4-tinystories/mlx-generator-compiled-gelu-screen-v1.md) reduced one-process batch-32/128 medians 582/2039 to 542/1845 ms with lower MLX peak but slightly higher RSS and batch-128 cache; all five official-seed files match the previously scored hybrid output byte for byte. Ten compiled-candidate fixed-code summaries across 80 rows match the earlier fused-normalization summaries byte for byte; same-device band and decisions pass, with existing upstream float-band exceptions retained. The [three-process Python comparison](docs/results/2026-11-15-phase4-tinystories/mlx-generator-compiled-gelu-python-performance-v1.md) favored Rust in latency and measured memory at batches 1/4/32 but failed at 128: Rust/Python median 2186/2168 ms and p95 12215/2539 ms, with eight Rust trials above four seconds. First-use cost, compiler cache panic behavior and tail cause remain open; nothing is integrated.

- **Compiler error guard screened:** an [isolated follow-up](docs/results/2026-11-15-phase4-tinystories/mlx-compiled-gelu-error-guard-plan-v1.md) catches compiler-constructor panics under unwinding and explicitly rejects the compiled path under `panic=abort`, without silent fallback. Formatting, patch replay, Linux default checks and focused CUDA golden/output checks pass. [Mac checks](docs/results/2026-11-15-phase4-tinystories/mlx-compiled-gelu-error-guard-screen-v1.md) passed simulated initializer failure, the real-Metal operator test, explicit abort-build rejection and one 128-row seed identity comparison. Actual MLX failure injection, full guarded-source gates and repeated performance remain pending; the unguarded batch-128 performance failure remains open. Nothing is integrated.

- **Alternating-runtime GPU diagnostic:** the [instrumented run](docs/results/2026-11-15-phase4-tinystories/mlx-compiled-gelu-tail-observation-v1.md) used three alternating Python/Rust processes with system-wide GPU snapshots and timestamped Rust requests. No Rust request exceeded four seconds; four Python requests did, with a 6.01 s maximum. This does not establish a common cause or erase the uninstrumented Rust failure. Local order, observer and timestamp-alignment tests and real-run clock checks pass. Polling may perturb timing; global counters and last-submitter PID cannot prove process GPU time or cause.

- **Generator RMS fusion experiment deferred:** [isolated plan](docs/results/2026-11-15-phase4-tinystories/mlx-generator-rms-fusion-plan-v1.md) applies the previously deferred MLX weighted-RMS patch to the hybrid candidate; the generator uses 264 such Q/K normalizations per request. Focused CUDA Sudoku golden and strict-cuDNN TinyStories output checks, formatting, the real-Metal weighted-RMS test and one seed-42 128-row identity check pass. A [one-process old/new screen](docs/results/2026-11-15-phase4-tinystories/mlx-generator-rms-fusion-screen-v1.md) found a 3.4% batch-32 median improvement but unchanged batch-128 median and peak, with increased batch-128 cache. Neither process reproduced the tail. The candidate is not integrated; no general performance benefit or tail fix is claimed.

- **Optional Metal activity observer prepared:** the old/new runner can optionally sample a whitelist of system-wide `IOAccelerator` registry statistics with timestamps. These are not per-process GPU time; the last-submitter PID is only a snapshot and sampling can perturb timing. Local parser/readiness/error tests, syntax and help checks pass. [Mac idle and six-second synthetic-load checks](docs/results/2026-11-15-phase4-tinystories/mlx-gpu-observer-v1.md) collected records without errors; device utilization rose and the last-submitter field matched the synthetic test process while it ran. Registry sampling windows, general PID attribution and real observer/benchmark integration remain unvalidated; do not use it to claim exclusive activity or causal attribution.

- **Request-aligned Metal diagnostic:** an [isolated timestamp patch and analyzer](docs/results/2026-11-15-phase4-tinystories/mlx-wall-gpu-diagnostic-plan-v1.md) connect individual lifecycle requests to system-wide GPU snapshots without modifying model code. Formatting, local model check, patch replay and synthetic alignment/rejection tests pass. The [real-Mac diagnostic](docs/results/2026-11-15-phase4-tinystories/mlx-wall-gpu-diagnostic-v1.md) completed with no observer errors, but no request exceeded four seconds. Twenty-nine of 30 marked requests had GPU snapshots reporting 94–95% device utilization; the counters are global and their averaging window is uncalibrated. No tail cause or fix is established; the previous failure remains open.

- **Metal level-boundary experiment rejected:** an [isolated patch](docs/results/2026-11-15-phase4-tinystories/mlx-level-boundary-plan-v1.md) tested synchronous MLX materialization at generator level boundaries on top of hybrid normalization; CUDA retained a no-op hook. The focused CUDA Sudoku golden and strict-cuDNN retained TinyStories outputs passed, as did formatting. On Mac the 128-row seeded output matched the hybrid candidate byte for byte, but a [one-process old/new screen](docs/results/2026-11-15-phase4-tinystories/mlx-level-boundary-screen-v1.md) was slower (2166/1980 ms candidate/baseline) with virtually unchanged MLX peak. Neither run reproduced the tail. The experiment is not integrated; no tail cause or fix is established. The exploratory old/new runner accepts selected batches and repeated candidate-source fingerprints; syntax, help and duplicate-batch rejection passed locally.

- **Batch-64 experimental split not integrated:** the existing hybrid binary had a one-process 1066.47 ms median at batch 64; [probe and limits](docs/results/2026-11-15-phase4-tinystories/mlx-batch64-probe-v1.md). An [isolated prototype](docs/results/2026-11-15-phase4-tinystories/mlx-chunk64-plan-v1.md) splits MLX generation above 64 rows, retaining level-major random-word assignments and output order; CUDA remains unsplit. Its random-word boundary test, formatting, model check, pinned CUDA Sudoku golden and strict-cuDNN TinyStories output comparison passed. [Metal screening](docs/results/2026-11-15-phase4-tinystories/mlx-chunk64-screen-v1.md) matched the unsplit output at batches 64/127/128 for one seed but changed one batch-65 row, where the one-row chunk selects the composed normalization branch. At batch 128 the split was slower in a single-process comparison (2111/2002 ms split/unsplit), with MLX peak 2471/2490 MiB. Neither process reproduced the tail. No tail remedy or gate pass is established.

- **Batch-sensitive normalization candidate prepared:** the [isolated plan](docs/results/2026-11-15-phase4-tinystories/generator-hybrid-norm-plan-v1.md) retains composed normalization below batch 16 and uses the prior fused path at larger batches, targeting the observed small-batch MLX peak regression. The isolated CUDA Sudoku golden and strict-cuDNN 128-row TinyStories output comparison passed (`reports/phase4-hybrid-norm-cuda-v1/test.log`), with watched hashes stable. Full isolated `tools/check.sh --cuda` followed by `--mlx` passed on real devices with 18 operator cases each. [Metal screening](docs/results/2026-11-15-phase4-tinystories/generator-hybrid-norm-screen-v1.md) matched original samples at batches 1/4/15 and fused samples at 16/128; all five official-seed 256-sample files are byte-identical to the previously scored fused candidate's nonblind quality retest. Three-process small-batch comparisons had lower Rust median/p95 and measured memory than Python; MLX peak margins were only 1.33/4.52 MiB. The [large-batch comparison](docs/results/2026-11-15-phase4-tinystories/generator-hybrid-norm-mlx-large-performance-v1.md) favored Rust at batch 32 but failed at 128: Rust/Python median 2553/2225 ms, p95 16354/2454 ms, with nine Rust trials above four seconds. CPU snapshots show WindowServer activity during both slow and fast Rust processes, without GPU attribution or proof of cause. Source-bound numerical coverage and a real tail remedy remain open; the candidate is not integrated.

- **Generator fused-normalization candidate prepared:** the generator currently executes 275 composed layer norms per TinyStories request; an [isolated candidate](docs/results/2026-11-15-phase4-tinystories/generator-fused-norm-plan-v1.md) dispatches the existing backend accelerated hook. CUDA defaults to the same operation, Metal uses existing fused MLX normalization and may differ in FP32 rounding. The isolated Linux candidate passed the full `tools/check.sh --cuda` (`reports/phase4-fused-norm-cuda-full-v1.log`, 18 real-device cases), pinned Sudoku golden and strict-cuDNN retained 128-row TinyStories code/ID comparison (`reports/phase4-fused-norm-cuda-v1/test.log`); watched focused-check hashes remained stable. [Metal screening](docs/results/2026-11-15-phase4-tinystories/generator-fused-norm-mlx-screen-v1.md) passed the fused operator and Sudoku golden and matched forced-code generator decisions within the existing MLX reference band on 16 rows, but 35 of 128 seeded code sequences differed; no stochastic quality claim is made. A [one-process old/new screen](docs/results/2026-11-15-phase4-tinystories/generator-fused-norm-mlx-screen-v1.md) found candidate median 548/1972 ms versus baseline 621/2070 ms at batches 32/128, but batch-128 MLX peak/cache rose. The [nonblind five-seed quality retest](docs/results/2026-11-15-phase4-tinystories/generator-fused-norm-quality-v1.md) passed the frozen PPL and entropy criteria. Expanded fixed-code checks now cover 80 rows: all generator logits remain within the existing same-device MLX reference band and all checked decisions match; 1,132 generator/final-logit values remain outside the original CUDA diagnostic band. The full isolated `tools/check.sh --mlx` passed with 18 real-Metal cases (`reports/phase4-fused-norm-mlx-full-v1.log`), following the full CUDA check. The [three-process Python comparison](docs/results/2026-11-15-phase4-tinystories/generator-fused-norm-mlx-performance-v1.md) failed: Rust MLX peak exceeded Python at batches 1/4 and batch-128 p95 was 12.65 s versus 2.37 s, despite lower median latency and RSS. The candidate is not integrated; investigate the small-batch memory regression and preserve the unresolved tail.

- **Cooperative Metal sampler rejected:** the isolated [experiment](docs/results/2026-11-15-phase4-tinystories/mlx-categorical-cooperative-probe-v1.md) exactly matched the existing sampler on 16,640 controlled rows but was slower in one-process microbenchmarks (large shape 42.56 ms versus 2.46 ms). It was not integrated; these isolated timings neither explain the earlier tail nor establish end-to-end performance. Preserve the source and results for review.

- **TinyStories tail trace:** one bounded marked [Metal diagnostic](docs/results/2026-11-15-phase4-tinystories/mlx-tail-trace-v1.md) had no request above 2.11 s and cannot explain or resolve the prior stall. All 30 measured intervals aligned to reported durations within 0.005 ms and overlapped target GPU intervals; the first warmup begin marker was missing. The 2.4 GiB system trace stays private on the Mac. An owned controller was paused after the target to avoid interrupting a live save, leaving a post-workload CPU-telemetry gap; the recorder timeout was fixed. The offline aggregate was validated and copied without system-wide data. The verified follow-up aggregate found median 2009.72 ms through `sample_seeded`, 69.71 ms afterward, and 1802.12 ms of unioned target GPU overlap per normal request. The trace has no shader-timing rows and no slow request, so neither these overlaps nor the empty table identifies the cause or ranks kernels. Do not repeat merely to obtain an outlier.

- **Prepared rotary-table candidate rejected:** CUDA and Metal correctness checks passed, but one-process old/new measurements showed under 1% latency difference and about 0.24 MiB more active Metal memory; this does not establish a worthwhile win or resolve the large-batch tail. The candidate was removed, restoring source identical to `0b71793`; [retained evidence and limitations](docs/results/2026-11-15-phase4-tinystories/mlx-prepared-rope-v1.md). TinyStories Metal performance remains open.

- **Metal cache candidate:** exploratory explicit 4 GiB process-local cache runs substantially reduced the batch-32/128 cache snapshots while retaining promising single-process latency; [measurements and limits](docs/results/2026-11-15-phase4-tinystories/mlx-cache4g-probe-v1.md). This is not a hard cache bound, a paired performance result, or an explanation of the earlier p95 spike. The [source-bound three-process comparison](docs/results/2026-11-15-phase4-tinystories/mlx-cache4g-comparison-v1.md) reduced median cache at both batches and passed the measured latency direction at batch 32, but failed it at batch 128 (Rust/Python median 2343/2167 ms, p95 3993/2538 ms). Four Rust trials exceeded four seconds. An approximate activity-window analysis found sampled Spotlight activity overlapping the two slower Rust processes; exact trial and GPU timing are unavailable, so causation remains unproven. This candidate is not a performance gate pass; preserve all runs and investigate the unresolved tail.

- **TinyStories CLI candidate:** `sample-tinystories` exposes the existing one-batch seeded generator with explicit backend and inference-only loading; [usage and limits](docs/PHASE4-TEXT.md). On Linux, strict-cuDNN lint, input-boundary tests, `tools/check.sh --cuda` (`reports/phase4-tinystories-cli-cuda-check-v1.log`) and a 128-row comparison with the retained scored seed-42 batch (`reports/phase4-tinystories-cli-cuda-v2/result.json`) passed. No model or sampling math changed. On the clean Mac, strict MLX CLI lint, the 128-row comparison against the retained scored Metal batch (`reports/phase4-tinystories-cli-mlx-v1/result.json`) and `tools/check.sh --mlx` (`reports/phase4-tinystories-cli-mlx-check-v1.log`, 18 real-Metal cases) also passed. The CLI has not been benchmarked under the dedicated-process lifecycle policy.

- **Expanded Metal numerical evidence:** 80 fixed-code development rows match upstream CUDA in all sampled codes, IDs and head argmaxes, but 1,541 floats remain outside the existing band. All generator logits match the independent MLX adaptation within that band; a conservative near-tie analysis and limits are in [development observations](docs/results/2026-11-15-phase4-tinystories/mlx-development-v1.md). The cross-runtime numerical differences remain unresolved; the old band is diagnostic rather than a newly imposed TinyStories acceptance rule. A [two-pair batch-128 tail probe](docs/results/2026-11-15-phase4-tinystories/mlx-tail-probe-v1.md) did not reproduce the previous p95 spike; it does not erase that exception or identify its cause. Review that exception before declaring the Metal performance gate complete.

- **TinyStories Metal — active investigation:** Real-Metal sampler/sentinel and Q/K/V tests pass. The hybrid candidate's five official-seed sample files are byte-identical to the scored Metal retest; [quality](docs/results/2026-11-15-phase4-tinystories/mlx-quality-v1.md). Rust/Python MLX fixed-code generator values match within the existing band on 16 examples and all decisions match; the original Python CUDA comparison has sparse outside-band floats, not a numerical pass. The [default-policy run](docs/results/2026-11-15-phase4-tinystories/mlx-hybrid-performance-v2.md) has lower Rust median latency, RSS and MLX peak at all batches but higher immediate active/cache in several cells. An explicitly configured dedicated-process [lifecycle-policy comparison](docs/results/2026-11-15-phase4-tinystories/mlx-lifecycle-performance-v3.md) has lower Rust median latency and all reported memory counters, but batch-128 Rust pooled p95 is substantially worse and remains unresolved. Background CPU activity coincided with part of the slow cluster; causation is unproven. The first full Mac `tools/check.sh --mlx` failed on a needless borrow (`reports/phase4-tinystories-mlx-check-v1.log`); after the semantics-preserving fix, the full rerun passed 18 real-Metal cases (`reports/phase4-tinystories-mlx-check-v2.log`) and the pinned Sudoku generator golden passed, and four additional pinned development traces have been copied/hash-verified for expanded numerical checks. No Phase 4 acceptance exists yet. Preserve `.pi/`, retained evidence and the user's `/tmp` cleanup constraint.

- **Phase 3 complete — resumed final audit:** the user resumed after reboot. The full default `tools/check.sh` passed under normal umask `022`, including all 486 reporting tests; log `reports/phase3-publication-check-v3.log`. The earlier three umask-dependent failures and interrupted rerun remain in v1/v2 logs. The Metal audit and CUDA timing summary reproduced byte-for-byte; retained CUDA/MLX CLI reconstruction and inspection results matched, and both builds rejected unavailable devices without fallback. CUDA and Metal quality/performance evidence and limitations are in the [Phase 3 report](docs/results/2026-11-14-phase3-sudoku/README.md). No project-named `/tmp` artifacts remained when checked. Phase 4 is next; recheck Mac availability before later GPU work. The unrelated `.pi/` remains untouched.

- The following reboot handoff and pending-publication status record the state before this successful resume and do not supersede the completion result above.

- **Reboot handoff — 2026-11-15:** user requested immediate reboot and to save, commit and push the state. Stop ongoing work and wait for explicit resume. All CUDA and MLX hardware checks and Phase 3 quality/performance measurements passed, but the final full default publication check still needs to be rerun after reboot. Its first attempt `reports/phase3-publication-check-v1.log` failed three pre-existing diagnostic tests due to umask `022`; the test fixture was fixed to create each parent privately and all six affected tests passed under `022` in `reports/phase3-umask-regression-v1.log`. The full repeat `reports/phase3-publication-check-v2.log` was intentionally interrupted for reboot and is not a pass. Rerun `RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 ./tools/check.sh` under default umask; preserve both prior logs. If it passes, update Phase 3 completion status and audit the committed evidence before marking the goal complete.

- **Phase 3 hardware evidence — scoped greedy Sudoku on CUDA and MLX:** `tools/check.sh --mlx` passed 18 real-Metal operator cases; the separately run pinned synthetic generator golden passed. Metal matched pinned Python on all codes and final IDs across 6,000 boards; CUDA matched final IDs with one measured medium intermediate-code near tie. Three fresh processes per runtime/batch found Rust faster and lower in median process peak RSS and MLX peak at batches 1/4/32/128; allocator cache and activity caveats are in the [dual-platform report](docs/results/2026-11-14-phase3-sudoku/README.md). The [independent audit](tools/bench/audit_phase3_mlx.py) recomputes raw quality/timing evidence, checks tracked sources and rejects a changed summary. Metal raw outputs/activity are locally byte-verified in `reports/phase3-mlx-captured-v1`; licensed inputs stay outside Git. Python MLX is an adaptation qualified on 128 easy boards, not the original PyTorch runtime. Both local and Mac `/tmp` contained no `ncm-*` entries after validation. The unrelated `.pi/` remains untouched. Recheck Mac availability before later GPU runs.

- **Dependencies and retained data:** no required Phase 3 dependency or evidence is intentionally stored in `/tmp`; reboot may clear it. On-disk local dependencies and assets are under `.reference/`, `artifacts/`, `target/` and `reports/`, including `.reference/venv`, `.reference/upstream-ncm`, `.reference/phase3-final-timings-source`, `artifacts/canonical/sudoku-*`, `reports/phase3-linux-qualified-v1`, `reports/phase3-linux-final-timings-v1`, `reports/phase3-mlx-captured-v1`, and `reports/phase3-cli-reconstruct-v1`. Mac equivalents and native libraries remain under `~/projects/lexlapax/ncm-rs`; inspect before syncing and do not overwrite unrelated work. These ignored artifacts are not in Git and must be preserved. The source, compact results and this handoff are committed separately.

- The following reboot-handoff and pending-Apple entries are historical; they document prior state and do not supersede the completed measurements above.

- **Reboot handoff (2026-11-15):** user requested saving state for reboot and resumption. Stop here; do not launch further Mac jobs before the reboot. Afterward, recheck Mac availability and SSH using `ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1 -o ControlMaster=no -o ControlPath=none macstudio 'printf "ssh-ready\n"'`; inspect the remote checkout before fast-forwarding. The clean Mac checkout was last confirmed at `c56bcf5`. Linux checkout has only the unrelated untracked `.pi/`. Linux quality and performance evidence is complete for the current checkpoint; Phase 3 still needs real Metal golden, three full 2,000-board evaluations, paired Python MLX performance/memory and publication of both platforms' results. Do not claim the prepared Apple scripts have passed.
  - Mac already has the easy artifact, all 15 evaluation data files and the three Linux Python reference outputs; copied data/output hashes matched Linux. Medium and hard canonical artifacts were absent at the last inspection and must be copied to new paths, then verified, without overwriting existing work. Reference manifest SHA-256 is `3fa4abca7c28c92085c3a3ac57656f4283433499187d4f5a58545ff4bb1ba82f`.
  - On available Apple hardware, use the documented native MLX environment (`PATH=/opt/homebrew/bin:$PATH`, `MLX_SYS_PREBUILT=$PWD/.reference/mlx-native`, and `RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"`). Run `./tools/check.sh --mlx`, build the release CLI with `--features mlx` and `mlx_generator_bench` example, then run `tools/bench/run_phase3_mlx_quality.py` with `--reference-dir reports/phase3-linux-qualified-v1`, the reference manifest SHA above and a new output directory. It runs the ignored pinned synthetic golden before full quality. Run `tools/bench/run_phase3_mlx_bench.py` with the easy Python reference and its SHA `bd5ea581e7d230866637be71caab24aec7b2670164f25415e1db474e591655ca`; the script first requires exact 128-board reference qualification. Preserve any failure, activity evidence and differences; optimize and revalidate if Apple misses the user-required speed or memory direction.
  - Eight root-owned artifacts from earlier fake sudo tests remained under `/tmp` when checked. The exact grouped removal command was supplied to the user; after reboot recheck what actually remains, never remove unrelated `/tmp` files. Raw Linux captures remain in ignored `reports/phase3-linux-qualified-v1`, `reports/phase3-linux-final-timings-v1` and `.reference/phase3-final-timings-source`; licensed weights/data remain outside Git. The last completed CUDA suite and concurrent golden passed. No local cargo, rustc, nvcc or generator benchmark process was running when checked for this handoff. Do not infer Apple results from Linux.

- **Phase 3 performance/memory requirement:** user requires Rust to outperform Python and asks for a time/space audit of all Rust structs. Initial repeated CUDA baseline was slower at every batch; retained raw results and [summary](docs/results/2026-11-14-phase3-sudoku/linux-baseline-timings-v1.json) are historical. Latent reuse, existing fused RoPE and packed Q/K/V reduced batch-32 exploratory median to 447 ms vs Python 458 ms, with matching hash; a one-process four-batch probe found batch 128 still ~1% slower. Prepared geometry, typed weights, fused RoPE and bit-exact specialized CUDA layouts are under validation; a one-process batch-128 probe observed 1,653 ms Rust versus 1,767 ms Python. The optimized CUDA CLI exactly matches the earlier Rust codes and IDs on all 6,000 boards, with the same single Python code near tie and zero decoded-ID differences. CUDA regression checks and the concurrent generator golden passed after the latest concurrency fix; full quality and Apple gates remain pending. A shared-generator concurrency test exposed a driver stall with ordinary Candle kernels; model requests are now serialized and the concurrent golden passes. Scope and limits are in the [structure audit](docs/results/2026-11-14-phase3-sudoku/rust-structure-audit-v1.md). The later repeated measurement and structure-review scope are below. NVIDIA recommends reusing compatible cuBLASLt heuristic results, but local instrumentation measured only ~1.2 ms per batch-32 request, so no native cache was added. Source: https://docs.nvidia.com/cuda/cublas/index.html#heuristics-cache.

- **Repeated CUDA candidate measurement:** three fresh processes per runtime and batch, 30 synchronized samples each, show Rust/Python medians of 25.72/33.89, 63.23/68.04, 419.45/447.28 and 1644.83/1744.12 ms at batches 1/4/32/128, with matching output hashes. Median whole-process peak RSS is 597–613 MiB Rust vs 1,706–1,760 MiB Python; sampled GPU process peaks are also lower, subject to 1 Hz sampling. Desktop GPU processes were present. [Measurement and limits](docs/results/2026-11-14-phase3-sudoku/README.md). After request serialization, `tools/check.sh --cuda` passed, including 486 report tests and 18 real-device operator cases; the separate concurrent generator golden also passed. Fresh source-bound full quality is recorded below; Apple work remains pending.

- **Fresh Linux full-quality capture:** at `cb0162c`, all three 2,000-board evaluations reproduced the earlier result: no final ID differences from pinned Python, one medium intermediate code difference at a measured near tie, and solved counts 1,988 / 1,928 / 1,726 for easy / medium / hard in both. Source, checkpoint and dataset hashes were stable before/after. [Quality summary and reproduction](docs/results/2026-11-14-phase3-sudoku/README.md). Apple availability was requested; MLX checks and performance remain pending.

- **Apple comparison preparation:** the clean Mac checkout was fast-forwarded to `39a7e77` without starting GPU work. A new offline Python MLX Sudoku-easy generator adaptation has only passed Python syntax validation; it is not a qualified baseline until checked against the pinned original reference on real Metal. Mac GPU availability was requested; await the response before MLX workloads. The source-bound Linux GPU evidence is unchanged.

- **Apple harness preparation:** added a native MLX generator timing example and a paired-process runner that requires both MLX implementations to match 128 pinned Python CUDA boards before timing, records source/input hashes and process RSS, and captures nonprivileged CPU activity. Prepared scripts pass Python syntax checks; `cargo fmt --all --check` and the Linux default `cargo check -p ncm-backend-mlx` pass. These do not compile or validate the Apple-only example or establish MLX correctness, performance or memory. The Python adapter now pre-materializes host input outside timing and releases prior output arrays. Await Mac availability before real Metal work.

- **Apple reference provenance:** the prepared MLX adapter rechecks the actual reread configuration and generator payload against the existing pinned manifest; the comparison runner will fingerprint both the Python MLX wheel libraries and the Rust native MLX libraries. Python syntax and runner help checks pass. Real Metal execution and qualification remain pending Mac availability.

- **Mac state:** clean checkout fast-forwarded through `2e3885d`; no MLX workload was started. Previously absent ignored paths `artifacts/canonical/sudoku-data-phase3` and `reports/phase3-linux-qualified-v1` were populated with evaluation data and the three Linux Python reference outputs. All 15 dataset file hashes and the three reference-output hashes match Linux. Medium and hard canonical artifact directories remain absent; do not overwrite existing files. GPU availability and the grouped removal of eight root-owned repo `/tmp` artifacts were requested and have not yet been confirmed.

- **Apple full-quality preparation:** added an explicitly ignored real-Metal generator golden using the pinned synthetic hashes, and a quality runner that checks that golden actually ran before executing all three 2,000-board Rust evaluations and scoring against the stable, hashed Linux Python capture. Local CPU-only preflight verified all pinned artifacts, dataset files and references; a wrong reference-manifest hash was rejected. Python syntax, Rust formatting, and default Linux `cargo check -p ncm-backend-mlx` pass. None compiles or runs the Apple-only golden; Metal correctness and quality remain pending availability.

- **Throughput reporting:** the Linux results now include boards per second derived from measured batch latency; the prepared Apple runner reports the same quantity and pooled nearest-rank p95. Arithmetic and Python syntax were checked locally. These are serial batch-throughput estimates, not concurrent-request throughput; no Apple values have been measured.

- **Mac access update:** the clean Mac checkout was last confirmed at `3e69dc2`; the subsequent attempt to pull `5a13a25` and a separate `ssh-ready` probe both timed out. No Metal workload has run, and Mac GPU availability is still unconfirmed. Retry the documented `ssh macstudio` form after access returns, inspect checkout state, then fast-forward before starting the approved Linux-first Apple gate. Eight root-owned project `/tmp` artifacts remain pending the grouped user cleanup.

- **Mac access restored:** `ssh-ready` succeeded, the clean checkout fast-forwarded to `072c718`, and nonprivileged `ps` showed substantial system background activity. GPU availability remains unconfirmed; no Metal workload ran. The prepared benchmark now rejects any batch input that is not the corresponding prefix of the pinned 2,000-board reference; local source/input checks and Python syntax pass.

- **Structure review follow-up:** field-level triage now covers the 183 explicitly declared Rust structs in workspace and vendored source, with focused method review for material owners. BPE symbol/merge storage, Unigram lattice/trie ownership, BERT post-processing, regex cloning and streaming/pre-tokenization allocation candidates are documented in the [audit](docs/results/2026-11-14-phase3-sudoku/rust-structure-audit-v1.md); they remain unmeasured for new text workloads. Macro-generated types and external dependencies are outside this source inventory. No global-optimality claim.

- **Phase 3 in progress:** user authorized completion. Shared eager greedy Sudoku generator and explicit-device JSON CLI are implemented but not accepted. CUDA initial parity matches pinned Python for 644 codes/178 decoded IDs on two synthetic boards and for both codes/IDs on two official boards per difficulty; maximum synthetic per-level logit difference 4.25e-5. Easy first 128 official boards also match exactly. New explicit medium/hard artifact profiles and newly pinned dataset revision are documented in [artifact contract](docs/ARTIFACTS.md); canonical artifacts and licensed datasets remain outside Git. Initial full Linux comparison has zero decoded-ID disagreements across all 6,000 boards; one medium intermediate code differs at a measured near tie. [Initial evidence](docs/results/2026-11-14-phase3-sudoku/README.md). The required `tools/check.sh --cuda` passed, including 485 report-tool tests and 18 real-device operator cases; the separately invoked CUDA generator golden passed. Repeated synchronized Rust/Python timing and GPU-activity capture has completed; real MLX validation and both complete gates remain pending. [Plan](docs/PLAN.md#phase-3--end-to-end-sudoku-vertical-slice).

- **Pre-Phase 3 cleanup:** corrected stale Phase 2/3 plan status and the reproducible performance summary’s exception status and reproduction command. The CUDA/MLX timing, Python comparisons and memory data are unchanged. Final evidence replay and three targeted negative tests passed; the performance summary regenerates byte-for-byte. Removed obsolete local experimental build caches while preserving the retained evidence.

- **Phase 2 complete with a scoped user-accepted exception:** [final audit](docs/results/2026-11-11-phase2-optimization/phase2-final-audit-v1.md), [exception](docs/results/2026-11-11-phase2-optimization/apple-immediate-memory-exception-v1.md). Linux and Apple controlled timing/host-memory audits are 78/78, and fresh trace qualifications pass 642/2,500 each. The Apple immediate B4 snapshot is 1.058744× prior Rust and is accepted as an exception, not a 1.05 pass. The separate B4 encode MLX active-memory peak is 1.20127× and is disclosed, not part of that exception or host RSS. Fresh Linux tokenizer and finite CUDA lifetime checks are [recorded](docs/results/2026-11-11-phase2-optimization/final-linux-closure-checks-v1.md). Large licensed assets need separate publication authorization; Phase 3 is unstarted and requires separate authorization. The final audit reproduced byte-for-byte in a temporary clean worktree and its three negative tests passed. The local `tools/check.sh` passed, including 484 report-tool tests. Temporary Phase 2 worktrees were removed from Linux and Mac; dirty changes, ignored reports and histories were preserved under `.reference/phase2-cleanup-preserved` on each machine (build caches excluded). Historical entries below retain their contemporaneous status and do not supersede this result.

- **Metal trace launch bound:** a fresh 642-trace/2,500-decision capture records
  the executable, MLX/model sources and native libraries before/after; its
  qualification matches the earlier gated result byte-for-byte. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-bound-trace-capture-v2-result.md).

- **CUDA trace launch bound:** a fresh 642-trace/2,500-decision capture records
  the trace executable and both CUDA source hashes before/after; its qualification
  matches the gated result byte-for-byte. Current depfiles list both CUDA modules;
  their historical build provenance is not claimed. [Evidence](docs/results/2026-11-11-phase2-optimization/cuda-bound-trace-capture-v2-result.md).

- **Current source and evidence binding:** the [v2 checker](docs/results/2026-11-11-phase2-optimization/phase2-current-bindings-v2.md)
  validates the `f155794` candidate, both 78-cell audits, trained qualifications,
  144 Linux pool comparisons, and reused Apple benchmark/library identities;
  four negative cases pass. The [current summary](docs/results/2026-11-11-phase2-optimization/phase2-performance-summary-v2.md)
  reports timing, host memory and lifecycle evidence. A fresh temporary worktree
  without model weights reproduced both outputs and passed four negative cases.
  At that historical checkpoint Apple immediate memory remained unresolved; the later scoped user exception is recorded above.

- **Apple completion ordering reviewed:** pinned MLX event waits and Metal
  completion handlers are distinct; explicit stream synchronization waits for
  a command buffer but does not by itself prove prior-handler ordering. This
  supports a cleanup hypothesis, not allocation attribution or acceptance.
  [Source review](docs/results/2026-11-11-phase2-optimization/mlx-event-completion-review-v1.md).

- **Apple unconditional synchronization deferred:** three paired full-codec
  runs reduce immediate B4 block active bytes to 59,970,828, but B4 block
  medians regress 10–16% in every pair. No production change, float parity or
  gate pass. [Result](docs/results/2026-11-11-phase2-optimization/mlx-stream-sync-runtime-result-v1.md).

- **Apple stream synchronization diagnostic:** explicitly synchronizing the
  GPU stream after block evaluation reduced immediate B4 active bytes from
  69,702,948 to 59,970,828 in all three pairs. This is not a runtime change,
  timing result or gate pass; full-codec cost and float parity remain open.
  [Result](docs/results/2026-11-11-phase2-optimization/mlx-stream-sync-memory-result-v1.md).

- **Apple Q/K/V memory toggle:** separate slices reduce immediate B4 block
  active bytes by 376,832 consistently, but remain 1.053× the historical
  previous-Rust snapshot; no float parity or timing was measured. Shared CUDA
  packed-layout speed gains mean this alone is not a qualified fix. No runtime
  change or gate pass. [Result](docs/results/2026-11-11-phase2-optimization/mlx-qkv-memory-toggle-result-v1.md).

- **Apple projection memory toggle deferred:** six isolated processes found
  no consistent immediate B4 block improvement from replacing fused projection
  with separate matmul/add; the target median was unchanged. ID/latent hashes
  matched, but floating outputs and timing were not qualified. No runtime change
  or gate pass. [Result](docs/results/2026-11-11-phase2-optimization/mlx-projection-memory-toggle-result-v1.md).

- **Bounded CUDA workspace validated:** candidate `f155794` passes Linux
  initial and controlled 78/78 audits, trained quality, and native pool ratios
  1.044155–1.044357 versus prior Rust. On Mac the full MLX check, 642 traces,
  2,500 decisions and four tokenizer probes pass; rebuilt timing binaries and
  trace hash match the earlier Apple control. The later user acceptance covers the immediate snapshot; the separately higher encode peak remains disclosed.
  [Linux gate](docs/results/2026-11-11-phase2-optimization/cuda-bounded-linux-gate-v1.md),
  [Apple checks](docs/results/2026-11-11-phase2-optimization/cuda-bounded-mlx-check-v1.md).

- **CUDA workspace feasibility:** zero workspace and a 2.5 MiB heuristic cap
  produced mixed isolated timings. Retaining the 4 MiB algorithm preference but
  supplying 2.5 MiB gave numerically identical outputs and near-parity isolated
  timing on 12 shapes. The experiment still allocates 4 MiB and is not a gate;
  subsequent integration, fallback and pool measurement are reported above.
  [Results](docs/results/2026-11-11-phase2-optimization/cuda-workspace-followup-result-v1.md),
  [zero workspace](docs/results/2026-11-11-phase2-optimization/cuda-zero-workspace-poc-result-v1.md).

- **Historical Linux live-pool concern:** prior-candidate synchronized native CUDA
  pool-used snapshots were 1.07039–1.07071× prior Rust in every batch/scope,
  above the separate 1.05 live-memory band if that pool metric is applicable.
  The 4,219,984-byte increase matches a 4 MiB fused-projection workspace plus
  25,680 bytes of prepared state; Python allocator counters differ in coverage.
  The host-memory audit did not decide this; the later bounded candidate above
  reduces the measured same-pool ratio. [Review](docs/results/2026-11-11-phase2-optimization/linux-live-pool-review-v1.md).

- **Pinned algorithm alignment reviewed:** the supported codec's quantization,
  interpolation, decoder and tokenizer choices are mapped to the pinned Python
  source, including deliberate snapshot/reconstruction differences and FP32
  limits. This is not exhaustive parity or a new run. [Review](docs/results/2026-11-11-phase2-optimization/upstream-codec-alignment-v2.md).

- **Earlier closure binding verified:** guarded production/harness sources then
  matched gated candidate `5200924`; selected pins/fixtures match baseline, and the
  archived and published Linux/Apple 78-cell audits pass with unchanged limits.
  The Apple source review also matches all 315 reports and 210 Python copies.
  All 58 timing and 20 host-memory cells per platform are uniquely inventoried
  and their stored controlled bounds independently checked; qualification
  inventories, oracle policy, 2,500 decisions and four-profile tokenizer probe
  contents also match. Focused negative cases pass. [Platform summaries](docs/results/2026-11-11-phase2-optimization/phase2-performance-summary-v1.md)
  report per-scope timing, candidate throughput, host-memory ratios and
  observed first-load/first-execution ranges and separate lifecycle phases.
  The earlier Linux and Apple gate pages now link to the subsequent control
  without erasing their historical results.
  A clean Linux replay is byte-identical; Apple replay differs only in nine
  last-digit standard deviations. This does not resolve live memory or close
  Phase 2. Public README, plan, architecture and
  testing summaries and the results index now reflect the same state. [Checker](docs/results/2026-11-11-phase2-optimization/phase2-closure-bindings-v1.md).

- **Settled memory accounting reviewed:** the archived matched Apple diagnostic
  shows a 776-byte candidate delta in every workload and cross-process pairing
  at 10 and 100 ms; cached codebook norms have 768 logical payload bytes. This
  is consistent with prepared-state overhead, not proof of allocation ownership
  or live-graph comparability. The immediate B4 difference was later accepted as a scoped exception; exact allocation ownership remains unproven. The [completion checklist](docs/results/2026-11-11-phase2-optimization/completion-audit.md)
  and experiment ledger now reflect the passing 78-cell control, Apple CPU
  diagnostic and deferred fused interpolation. [Evidence](docs/results/2026-11-11-phase2-optimization/apple-gate18-results-v1.md).

- **Fused interpolation deferred after full-codec test:** Linux/CUDA and Metal
  checks and 642-trace / 2,500-decision qualification pass. B4 encode peak
  drops 5.7% versus the current runtime but remains above old Rust; paired
  model timings are mixed. No production change or acceptance claim. The
  1.05 live-memory criterion remains open. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-linear-fused-codec-result-v1.md).

- **Fused interpolation diagnostic promising:** six isolated Metal processes
  matched finite outputs bit-for-bit across ten shapes. B4 13→89 peak active
  allocation fell from 3,083,040 to 822,048 bytes and median time ratio was
  0.8689. Full-model, special-value, views, startup and safe integration remain
  untested; no production change or gate pass. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-linear-fused-poc-result-v1.md).

- **Apple 18-process control passed 78/78** after the initial B4 tail failure;
  quality passed, and the separate 45-process CPU comparison is validated.
  Matched live-memory probes show a 1.058744 immediate B4 block ratio, settling
  near parity after a barrier; B4 encode peak active bytes are 1.20127× prior
  Rust. Earlier matched evidence attributes a similar 26 MB peak increase to
  linear-gather interpolation, which also sped up B4 encode. The 1.05
  steady-state live criterion and resource tradeoff remain open; Phase 2 is not
  complete. [Evidence and activity](docs/results/2026-11-11-phase2-optimization/apple-gate18-results-v1.md).

- **Same-process unmasked attention deferred:** output comparisons pass but
  B4 block medians are mixed (0.985/0.977/1.007); no runtime change or gate
  pass follows. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-attention-toggle-result-v1.md).

- **GELU lifetime observation:** all backend/model drops return active MLX
  bytes to zero; first compiled/eager requests were 266/85 ms in sequential
  processes, with later first requests near 7 ms. Startup and fallible compiler
  initialization prevent integration as-is. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-gelu-lifetime-result-v1.md).

- **Shape-independent GELU diagnostic:** Metal shape/view/special-value
  regression passes; all nine resident-block medians improve. One first compiled
  reconstruction took 261 ms. Cache errors, startup/lifetime and acceptance
  remain open; no integration. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-gelu-shapeless-result-v1.md).

- **Same-process GELU diagnostic:** compiled exact GELU improves resident-block
  medians in all nine comparisons; reconstruction is near unchanged. One first
  compiled reconstruction took 213 ms. General cache/startup design and full
  acceptance remain open; no integration. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-gelu-toggle-result-v1.md).

- **GELU scalar reuse deferred:** finite outputs matched bitwise, but three
  isolated Metal processes showed no consistent material speedup; no backend
  state or runtime change was added. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-gelu-scalars-result-v1.md).

- **System-wide Metal trace captured an outlier:** one 1.617 ms B4 block
  overlaps 0.848 ms of other-process GPU intervals; the target main thread was
  blocked 1.450 ms. Three near-median trials have no recorded other-process GPU
  overlap; all 30 trials show WindowServer overlap only for the largest.
  This single trace does not prove causation or close the gate.
  [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-system-activity-v2.md).

- **Unmasked attention deferred:** Linux/CUDA and full Metal checks pass;
  trained qualification passes 642 traces and 2,500 decisions. Three paired
  Metal processes show mixed timings and sampled desktop activity, so no runtime
  change or gate pass is claimed. The ledger/checklist and per-process sampled
  activity are reconciled. The `macstudio` SSH host entry works; do not
  override its identity agent. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-attention-nomask-metal-v1.md).

- **Apple all-cell sensitivity:** both retained controls' 58 timing cells and
  20 memory cells are checked; post-hoc pooled timing intervals fit the limits,
  but neither failed gate is reclassified. A prospective 3+18-repeat option is
  selected by the user; an offline replay verifies 18-process
  summaries and the full audit reject a missing report. [Analysis](docs/results/2026-11-11-phase2-optimization/mlx-all-windows-exploratory-v1.md).

- **Full-codec GELU prototype deferred:** Linux CUDA and full Metal checks and
  trained qualification pass, but three full-codec pairs are mixed: B4 block
  improves while B2 block/reconstruction regress in two pairs. B2 encoder
  timings also swing widely despite no direct GELU call, so causation is open.
  Startup/cache
  risks remain; no integration or acceptance gate. The experiment ledger and
  completion checklist reflect the current control. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-gelu-codec-experiment-v1.md).

- **Exact GELU compilation diagnostic:** corrected finite checks pass in three
  isolated Metal processes, with lower warm B4 medians. First compilation cost
  is significant; full-model timing, trained parity and cache lifetime remain
  untested. No production change. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-gelu-compile-poc-v2.md).

- **System-wide Metal activity diagnostic incomplete:** target finished without
  a slow B4 trial, but the recorder exceeded a post-target deadline and the
  terminated trace cannot be exported. The runner no longer imposes that
  finalization deadline; the focused mocked regression passes. No repeat or
  GPU-overlap claim. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-system-activity-failed-v1.md).

- **Apple gate sensitivity:** post-hoc, window-stratified resampling of both
  retained B4 controls yields a Rust-comparison p95 interval upper of 1.098714;
  it was not predeclared and cannot close either failed gate. [Analysis](docs/results/2026-11-11-phase2-optimization/mlx-block-windows-exploratory-v1.md).

- **Marked Metal trace:** an isolated `ddde851` diagnostic with block signposts
  built and aligned 108 calls. One instrumented B4 slow trial had similar graph
  time, longer synchronous evaluation and higher recorded CPU-to-GPU latency;
  matched encoding ends at a similar offset to normal trials, but GPU begins
  later. Its submission record begins before the block marker.
  One trace does not prove the cause or gate outcome. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-block-signposts-v1.md).

- **Metal trace feasibility:** one bounded instrumented run produced 9,171 GPU
  interval records, but no slow B4 block trial; existing durations cannot align
  individual trials to the trace. Add explicit trial markers before attributing
  a captured outlier. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-tail-metal-feasibility-v1.md).

- **Apple quiet-window gate:** the separately declared unchanged-candidate
  matrix/control both yield 77/78; block p95 versus prior Rust remains open
  after control. Previous 76/78 remains evidence. User requested rethinking the
  Mac strategy; investigate slow-process attribution before another full gate.
  [Evidence](docs/results/2026-11-11-phase2-optimization/tiny-borrow-apple-quiet-gate-v1.md).

- **Dependency scope audit:** the checksum-pinned tokenizers archive and current
  vendored tree have 99 matching paths and only three changed files, each
  matching its reviewed integration commit. The 218-package lock inventory is
  unchanged except the approved local-source metadata; upstream/tokenizer pins
  and tracked fixtures match the frozen baseline. Static scope is not semantic
  or performance acceptance. [Audit](docs/results/2026-11-11-phase2-optimization/dependency-scope-audit-v1.md).

- **Upstream alignment review:** the clean pinned Python decoder and current
  Rust/MLX code have corresponding supported block order, with documented FP32
  evaluation differences and qualified traces. A separately built differential
  probe compared the prior/current TinyStories parsers on 57 synthetic payloads;
  outcomes match, but this is not a Python oracle or exhaustive proof. An initial
  shared-Cargo-target comparison was rejected as inconclusive. [Review](docs/results/2026-11-11-phase2-optimization/upstream-alignment-review-v1.md).

- **Evidence reconciliation:** the current 40-entry experiment ledger and
  completion checklist now distinguish historical runs from current Linux
  78/78 and Apple initial/control 78/78 and 76/78. Final acceptance remains
  open; the user was asked whether to choose a separately predeclared quiet
  gate or deeper profiling because the prior plan specified no retry. [Ledger](docs/results/2026-11-11-phase2-optimization/experiments.json), [audit](docs/results/2026-11-11-phase2-optimization/completion-audit.md).

- **Block tail diagnostic:** three instrumented Metal processes produced one
  B4 trial above 1.2× its process median; both host graph construction and
  synchronous evaluation rose. Evaluation includes submission/waiting and is
  not kernel-only, so the cause is unresolved. This perturbed diagnostic does
  not replace the 76/78 control. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-block-segments-v1.md).

- **RMS fusion deferred:** direct Metal regression, full MLX check and 642
  trained traces pass, but a three-pair diagnostic shows no target batch-4
  block improvement (median ratio 1.0055) and mixed other scopes. The
  experiment was not integrated; its limited results are not acceptance.
  [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-rms-affine-diagnostic-v1.md).

- **Latest Apple gate:** loader candidate `3555c67` passed the initial 78/78,
  but the predeclared all-workload control is 76/78: batch-4 block p95 versus
  Rust and Python remains open. TinyStories RSS/HWM pass. Wallpaper activity
  was recorded; it does not establish the outlier's cause. Full MLX checks,
  tokenizer probes, 642 trained traces and clean audit reproduction pass. Do not
  retry unchanged. [Gate](docs/results/2026-11-11-phase2-optimization/tiny-borrow-apple-gate-v1.md).

- **TinyStories loader candidate:** unescaped hex strings are borrowed only
  during parsing; escaped JSON retains owned handling and no borrow escapes.
  Full Linux CUDA check and four tokenizer probes pass. A small matched Linux
  diagnostic shows 24,060 fewer load allocation calls, 14% lower median load
  and lower observed HWM, with mixed decode timing. This is not acceptance.
  Initial and fixed controlled Linux gates pass 78/78; the Apple initial gate
  passes 78/78 but the fixed control remains 76/78, as recorded above. [Evidence](docs/results/2026-11-11-phase2-optimization/tiny-borrow-candidate-v1.md), [Linux gate](docs/results/2026-11-11-phase2-optimization/tiny-borrow-linux-gate-v1.md).

- **Previous Apple gate:** projection candidate `a323b16` is **74/78** initially
  and **76/78** after the fixed control. Only TinyStories host RSS/HWM remain
  open (peak ratio 1.12434); the block and Qwen timing cells pass. Do not retry
  unchanged; investigate loader memory without changing byte-BPE semantics.
  Source binding and clean audit reproduction pass. [Gate](docs/results/2026-11-11-phase2-optimization/mlx-addmm-apple-gate-v1.md).
  A fresh finite 384-request MLX memory observation returns active bytes to zero
  after all drops, with stable cache and rising host RSS. [Memory](docs/results/2026-11-11-phase2-optimization/mlx-memory-addmm-v1.md).

- **MLX projection-bias candidate:** pinned MLX addmm preserves the projection
  plus bias operation, with fallback for other shapes. Linux CUDA checks pass
  and five executables match the 78/78 candidate. Full Metal checks and 642
  trained traces pass. A three-pair block diagnostic improves B4 medians in
  all pairs, but p95 and other scopes were mixed; this was not acceptance. The
  fixed full Apple gate yielded 76/78 and memory was reassessed as above.
  Retain prior 75/78. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-addmm-candidate-v1.md).

- **Prior Apple memory observation:** after reboot, a source-bound diagnostic on the
  previous runtime completed 384 reconstructions across three model lifetimes.
  Active MLX bytes returned to zero after each drop; cache snapshots stabilized
  in this finite workload, while host RSS increased. This is not a general
  bound or a timing gate. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-long-lived-memory-v1.md).

- **Qwen ID-copy experiment deferred:** an isolated accessor avoided the final
  ID-vector copy while preserving the tokenizer pipeline. Tokenizer tests and
  four probes passed, but three paired Linux processes did not show a speedup
  (median ratio 1.016); the noisy diagnostic is not acceptance and the
  dependency change was not integrated. [Evidence](docs/results/2026-11-11-phase2-optimization/qwen-ids-copy-diagnostic-v1.md).
  The user confirmed Mac Studio is ready after reboot.

- **Latest Apple gate:** the affine candidate `2850331` reached **77/78**
  initially and **75/78** after its fixed all-workload control. Batch-4 block
  p95 versus Rust/Python and Qwen encode median versus Rust remain open. Chrome
  was absent from top-process samples; no thermal warning was recorded. Both
  source inventories and clean-export audit reproduction pass. Do not retry the
  unchanged candidate; investigate substantive fixes and preserve the pinned
  upstream semantics. Long-lived MLX cache evidence remains open. [Gate](docs/results/2026-11-11-phase2-optimization/mlx-affine-apple-gate-v1.md).

- **MLX affine candidate:** the accelerated decoder normalization now passes
  matching affine parameters into the pinned fast primitive, with fallback for
  other shapes. Linux CUDA checks pass and five binaries match the 78/78
  candidate. Full MLX checks and 642 trained traces pass. A three-pair Metal
  diagnostic improves batch-4 block medians but has mixed p95/other-scope
  results; the fixed Apple gate is now 75/78 as recorded above. Retain the
  prior 72/78 and unchanged upstream/tolerance contract. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-affine-candidate-v1.md).

- **Previous Apple gate:** SSH restored and Mac checkout fast-forwarded. That
  candidate passes the full MLX check, 642 trained traces/2,500 decisions and
  four tokenizer probes; its initial Apple matrix is **69/78** and fixed
  all-workload control **72/78**. Batch-4 block p95 (two comparators),
  TinyStories encode/load p95 and GPT-2 RSS/HWM remain pending. Chrome appeared
  during control; activity and limitations are recorded. Source inventories
  and the separately omitted core trait are bound. Do not rerun unchanged;
  investigate substantive fixes, with Linux first.
  [Apple evidence](docs/results/2026-11-11-phase2-optimization/fused-apple-gate-v1.md).

- **Fused projection approved and integrated:** the user approved v2 with a
  request to stay aligned with upstream Python. Strict FP32, existing pins and
  tolerances remain; fresh CUDA check, 642 trained traces/2,500 decisions and
  four tokenizer probes pass. A cross-stream regression passes after the full
  check. The integrated codec binaries match the independently tested v2;
  Linux is **78/78** after its fixed all-workload control, reproduced from a
  clean export. Pinned upstream decoder layer ordering matches the integrated
  model; all 24 captured projection-output traces meet existing budgets. The
  Mac/Tailscale setup was changed and SSH restored. The Apple benchmark omits
  the changed core trait file from its source inventory; it has been separately
  bound to the Mac build as recorded above.
  The completion checklist reflects current Linux evidence; the
  32 MiB workspace on newer GPUs is unmeasured.
  [Linux gate](docs/results/2026-11-11-phase2-optimization/fused-linux-gate-v1.md). [Integration evidence](docs/results/2026-11-11-phase2-optimization/fused-integration-v1.md),
  [review](docs/results/2026-11-11-phase2-optimization/fused-projection-review-v1.md).

- **TinyStories decode candidate:** reserve at most 4 KiB from input ID count
  before byte-BPE decode. Long, skipped, malformed and invalid-ID regression,
  four real tokenizer probes and full CUDA check pass. Three-process Linux
  diagnostics show decode/skip median ratios 0.79144/0.82950 with equal output
  hashes; not acceptance. CUDA codec benchmark is byte-identical to the trained
  qualified, 78/78 candidate. Fresh Linux controls are 71/78 and, after user
  quieted desktop, 75/78: three codec p95 intervals remain open despite passing
  TinyStories. Do not rerun unchanged. Apple remains gated; batch-4 Metal block
  still needs a substantive fix. One profiled block run shows two copies, six
  adds and 55 event creations per call, but no large outlier or proven cause;
  event tracking remains enabled. [Linux gates](docs/results/2026-11-11-phase2-optimization/tinystories-linux-gates-v1.md).
  [Evidence](docs/results/2026-11-11-phase2-optimization/tinystories-reserve-v1.md).

- **CUDA convolution candidate:** convolution bias addition now reads the
  transposed result directly, eliminating an intermediate contiguous copy.
  Focused CPU/real-CUDA tests, 642 trained traces and 2,500 decisions pass;
  full `tools/check.sh --cuda` passes with private umask `077` after a
  formatting failure and three clock-fixture failures under umask `022`.
  Three-process diagnostics show decode medians 4.8–5.3% lower and encode
  medians 3.2–3.8% lower, with matching IDs/logits; this is not acceptance.
  Fresh Linux gate: 75/78 initial and **78/78** fixed all-workload control;
  nine lifecycle runs and clean-export reproduction pass. Apple initial and
  all-workload control were 75/78 and 74/78 with substantial Chrome activity.
  After Chrome closed, the fixed quiet-window initial/control are 72/78 and
  **75/78**: batch-4 block against Rust/Python and TinyStories decode against
  Rust remain open. Do not rerun unchanged; investigate substantive fixes.
  [Quiet Apple gate](docs/results/2026-11-11-phase2-optimization/conv-bias-apple-quiet-gate-v1.md);
  [Apple gate](docs/results/2026-11-11-phase2-optimization/conv-bias-apple-gate-v1.md). [Candidate](docs/results/2026-11-11-phase2-optimization/conv-bias-cuda-v1.md);
  [Linux gate](docs/results/2026-11-11-phase2-optimization/conv-bias-linux-gate-v1.md).

- **Qwen decoder dependency patch approved:** the user approved the isolated
  one-buffer byte-level decoder for integration. It preserves tested token
  fallback/invalid UTF-8 behavior and passes
  full CUDA/MLX checks and four tokenizer probes. Qwen decode diagnostics
  improve 26–31% across aggregate results and improve in every paired process;
  Mac GPT-2 encode varies adversely and sampled HWM is documented. Full gates
  remain necessary. Integrated candidate `211df63` passes Qwen decode in a
  fresh Linux matrix, but the initial audit is 71/78 and its one fixed control
  is 70/78 due to other timing uncertainties. After the user quieted the
  desktop, a small fixed diagnostic showed fewer spikes. A separately
  predeclared quiet-window full gate and all-workload control then yielded
  77/78 initial and **74/78** controlled; four codec p95 intervals remain
  above the unchanged limit. Offline inspection found matching stream-span
  spikes in all 65 >1.2× wall outliers across those cells, but the spans also
  include host submission/idle and do not identify the cause. Three separate
  instrumented processes found 24 such outliers with at most 0.031 ms thread
  run-queue wait versus at least 0.281 ms excess wall time; other causes remain.
  A single traced batch-1 decode outlier coincided with a slow CUDA event
  creation call; profiling perturbs timing and does not prove the cause of
  gate failures. Event tracking protects cross-stream lifetimes and remains
  enabled. The diagnostic and its limitations are in the quiet-gate report.
  The earlier
  failures remain retained. The later convolution candidate passes Linux as
  recorded above; Apple validation is next.
  [Quiet gate](docs/results/2026-11-11-phase2-optimization/decode-quiet-gate-v1.md), [earlier gate](docs/results/2026-11-11-phase2-optimization/decode-final-linux-gate-v1.md), [quiet diagnostic](docs/results/2026-11-11-phase2-optimization/decode-quiet-diagnostic-v1.md).
  [Review and evidence](docs/results/2026-11-11-phase2-optimization/qwen-decode-review-v1.md).

- **GPT-2 dependency patch approved:** the user approved the tested byte-level
  identity shortcut for integration. An isolated prototype
  shortcut preserves tested splits/offsets and passes full CUDA/MLX checks;
  matched GPT-2 encode diagnostics improve 4.5% Linux / 6.5% Mac; all 12
  diagnostic source inventories and output hashes verify from clean exports.
  The reviewed patch is integrated as `1f7d2a8`; Linux passes fresh **78/78**
  on the initial matrix, tokenizer probes pass, and byte-identical CUDA codec
  binaries retain trained parity. All six changed code/test files match the
  reviewed worktree. SSH access returned and the Apple initial and fixed
  control audits both pass **76/78**: GPT-2 now passes, but Qwen decode and
  skip-special median intervals remain above 1.05. Investigate a substantive
  Qwen improvement; do not rerun the unchanged candidate. The decoder change
  above is approved and being integrated. [Review](docs/results/2026-11-11-phase2-optimization/gpt2-bytelevel-review-v1.md), [Linux gate](docs/results/2026-11-11-phase2-optimization/bytelevel-final-linux-gate-v1.md), [Apple gate](docs/results/2026-11-11-phase2-optimization/bytelevel-final-apple-gate-v1.md).

- **Completion checklist reconciled:** current Linux and Apple checks, gate
  outcomes, lifecycle evidence, source reproduction, and targeted MLX
  normalization/attention disposition replace stale historical rows. Current
  Apple codec reports also show zero active MLX bytes and unchanged
  cache at the first and post-trial drop across 24 processes; long-lived
  inference cache behavior remains unproven. [Checklist](docs/results/2026-11-11-phase2-optimization/completion-audit.md).

- **Prior Apple gate fails:** candidate `884d8f6` passes full MLX checks and
  trained parity, but remains **77/78** after one fixed GPT-2 control: encode
  median interval upper 1.06574 exceeds 1.05. The decoder block regression is
  resolved and reconstruction gains are retained. The approved GPT-2 patch
  changes the candidate; do not repeat the prior unchanged gate. All 150 native
  report inventories and 225 shared Python report copies across current Linux
  and Apple matrices/controls match clean source exports. [Current Apple gate](docs/results/2026-11-11-phase2-optimization/blocknorm-apple-gate-v1.md), [source audit](docs/results/2026-11-11-phase2-optimization/raw/blocknorm-source-audit-v1.py).

- **Historical Apple gate:** combined candidate `2df27c3` has a fresh
  69/78 initial audit, including batch-4 resident-block ratio 1.35310 versus
  accepted Rust; fresh trained Metal parity and tokenizer probes pass. The
  initial failure reproduces from clean export. Isolated RMS-only and fast
  attention alternatives pass parity but do not resolve the regression. Final
  latent shapes, strides and float hashes match the accepted implementation;
  an Apple-only unpacked Q/K/V alternative also fails to resolve the regression.
  Fresh-process block diagnostics pass at similar speed with identical actual
  latents; the gap returns after the full workload and survives cache clearing.
  A single-gather alternative passed trained parity but did not consistently
  improve the workload and was not promoted. Synchronized boundary profiles
  implicate Q/K materialization but perturb execution; unpacking alone did not
  resolve the unsynchronized gap. Isolated decoder normalization passes
  trained parity and improves sampled block timings, but encode timings vary
  and show regressions. A refined normalization/attention candidate also
  passes trained parity. Production candidate `621c565` now passes the Linux
  CUDA check, trained parity and 78/78 gate after one fixed follow-up; its
  corrected evidence archive reproduces the audit from a clean export. Its
  first Apple check failed Clippy on test-only integer casts; after correction,
  the full Apple check passes. [Check evidence](docs/results/2026-11-11-phase2-optimization/raw/blocknorm-mlx-test-fix-v1.tar.xz).
  [Linux gate](docs/results/2026-11-11-phase2-optimization/blocknorm-linux-gate-v1.md), [state diagnostics](docs/results/2026-11-11-phase2-optimization/mlx-block-state-v1.md), [Apple initial gate](docs/results/2026-11-11-phase2-optimization/final-mlx-initial-v1.md), [alternatives](docs/results/2026-11-11-phase2-optimization/mlx-block-alternatives-v1.md).

- **Comparator source audit:** all 42 final-Linux and 24 historical-Apple
  Python comparator report inventories match their clean tested-source exports;
  declared upstream pins and recorded candidate/baseline heads agree. This is
  offline source/metadata evidence, not the pending final Apple measurement.

- **Historical Apple audit reproduced:** a clean `cd49989` source export and
  archived reports reproduce the expected 77/78 failure byte-for-byte; all 48
  native report source inventories match candidate or baseline. Separately,
  all six gather diagnostic inventories, output IDs and trained qualification
  verify from clean exports. Neither is a passing final gate for the newer
  combined candidate. [Details](docs/results/2026-11-11-phase2-optimization/mlx-linear-poc-v1.md#historical-apple-audit-clean-export-reproduction).

- **Combined candidate Linux gate passes:** `4ab4d3a` passes 78/78 after one
  fixed codec/GPT-2/Qwen follow-up (initial 74/78); full CUDA check and four
  tokenizer probes pass. A clean source export reproduces the audit byte-for-
  byte, with all 84 native report source inventories verified. All 99 vendored
  crate files match the verified archive except the two approved source patches;
  the license is unchanged. Apple final gate
  is next, but the 1Password SSH agent again refused to sign; availability and
  competing work must be rechecked after access returns. [Evidence](docs/results/2026-11-11-phase2-optimization/final-linux-gate-v1.md).
- **MLX compilation deferred:** a pure explicit-input normalization graph saved
  roughly 0.03–0.04 ms warm but first calls reached 509 ms; full-model benefit,
  bounded cache and trained parity remain unproven. [Details](docs/results/2026-11-11-phase2-optimization/mlx-compile-poc-v1.md).

- **MLX fast normalization deferred:** isolated fast layer/RMS normalization
  passes Metal tests and trained parity, but matched full-model diagnostics
  show batch-2/4 encode and batch-4 reconstruction regressions despite faster
  resident blocks. Production remains unchanged. [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-fastnorm-poc-v1.md).

- **BERT dependency patch approved:** an ASCII-only BERT
  normalizer path skips identity work; pinned version and non-ASCII path remain
  unchanged. Matched diagnostic encode medians improve 4.7% Linux / 3.9% Mac;
  exact alignment regression, four profile probes, and full Linux CUDA / Mac
  MLX checks pass. The user approved this additional vendored source change;
  neither final performance gate is implied. The completion audit reflects the
  current Linux and Apple status. [Review and evidence](docs/results/2026-11-11-phase2-optimization/bert-ascii-review-v1.md).
- **Area resampling experiment:** isolated padded gather improves larger Metal
  shapes but slows the smallest and increases peak allocations. A bounded
  hybrid passes trained parity, but full-model timings are confounded and peak
  allocations rise further; it is deferred. Production area remains unchanged.
  [Details](docs/results/2026-11-11-phase2-optimization/mlx-area-poc-v1.md).

- **BERT encoding experiment rejected:** a temporary consuming tokenizer-ID
  accessor reduced matched Linux encode median only 0.4%; no dependency change
  was promoted. The earlier Apple BERT p95 interval failure remains open.
  [Details](docs/results/2026-11-11-phase2-optimization/bert-ids-move-poc-v1.md).

- **Apple gate and linear-resampling candidate:** trained Metal parity passes
  642 traces / 2,500 decisions, and the Mac check passes with Linux-only report
  modules explicitly excluded. Initial Apple audit 76/78; one predeclared BERT
  follow-up yields 77/78, with BERT encode p95 interval upper 1.11420 > 1.10.
  Isolated gather interpolation matched six shapes and reduced representative
  evaluated time 1.0656 to 0.2966 ms, with higher measured MLX peak memory.
  The production candidate passes expanded Metal tests, trained parity and the
  Linux CUDA check. Matched full-model reconstruction improves substantially,
  with higher MLX allocator peaks and noisy resident-block timings. Both final
  candidate gates and the open BERT p95 comparison remain.
  [Evidence](docs/results/2026-11-11-phase2-optimization/mlx-linear-poc-v1.md).

- **Linux Phase 2 gate passes:** candidate `582e8fe`, including the approved
  pinned BPE allocation optimization, passes 78/78 after one predeclared
  codec/Qwen follow-up; the initial audit was 75/78. The Qwen checked-load
  follow-up median ratio versus accepted Rust is 0.95573. CUDA check passes
  under umask 077; an earlier umask 022 fixture failure is retained. Next is
  Apple/MLX: `macstudio` accepted the public key but the 1Password SSH agent
  refused signing, so availability and remote state remain unverified.
  [Details](docs/results/2026-11-11-phase2-optimization/vendored-linux-gate-v1.md).

- **Packed Q/K/V pool follow-up:** an offline audit of both retained archives
  shows the isolated B2 reservation first rises in reconstruction, not
  necessarily in the following block. In the fresh combined matrix, accepted
  and candidate reserved pool peaks match in every scope; candidate
  reconstruction used peaks are lower, while decode/block peaks are higher.
  Counters remain distinct from host RSS; no acceptance result changed.
  [Details](docs/results/2026-11-11-phase2-optimization/packed-qkv-v1.md#offline-pool-counter-comparison).

- **BERT dispatch isolation rejected:** separating BERT parsing into a
  non-inlined helper did not improve Qwen or BERT load in three matched
  processes per binary; the patch was reverted. Tokenizer tests and real-asset
  probes passed. The earlier combined candidate was 77/78 on Linux; the reviewed
  dependency patch is now approved and integrated as described above.
  [Details](docs/results/2026-11-11-phase2-optimization/bert-dispatch-v1.md).

- **Qwen BPE builder proof of concept:** an isolated temporary copy of pinned
  `tokenizers` 0.22.2 reserves exact merge-string capacity. Three matched Qwen
  processes per binary show median load 66.101 to 63.420 ms; one allocator
  capture shows reallocations 60,376 to 128. Tokenizer tests and four real-asset
  probes pass. Offline comparison confirms all 99 pinned crate files match the
  registry source and only the builder file changes in the PoC; the license is
  unchanged. The user approved and the production integration now passes the
  Linux gate as described above; Apple validation remains pending.
  [Details](docs/results/2026-11-11-phase2-optimization/qwen-bpe-builder-poc-v1.md).

- **Qwen load investigation:** accepted/candidate Rust allocation counts were
  identical in one instrumented capture. A separate phase profile measured
  parsing and pinned BPE construction at 26.352/32.615 ms. An in-place Unicode
  preprocessing candidate passed parity but regressed matched checked load
  65.693 to 77.133 ms; it was reverted and the release binary restored. The
  earlier combined candidate was 77/78 on Linux. No adaptive acceptance retry
  was run. [Details](docs/results/2026-11-11-phase2-optimization/qwen-load-investigation-v1.md).

- **Combined candidate Linux gate:** `6c95d21` passed `tools/check.sh --cuda`
  and completed a fresh 45-process matrix plus one predeclared 27-process Qwen
  follow-up. Both initial and controlled audits are **77/78**: B1 block and
  BERT checked load pass, Qwen decode's initial uncertainty resolves, but Qwen
  checked-load follow-up median ratio **1.05535** exceeds 1.05. All nine paired
  Qwen process medians favor accepted Rust. The failed audit reproduces exactly;
  this is a historical failure; the newer candidate passed as described above.
  [Details](docs/results/2026-11-11-phase2-optimization/combined-linux-gate-v1.md).

- **Packed Q/K/V candidate:** shared decoder packs Q/K/V in one layout before
  slicing; focused CPU/real-CUDA batch-independence checks and trained CUDA
  qualification pass, with unchanged native trace hash. In three matched
  processes per binary B1/B2/B4 resident-block median ratios are
  0.8273/0.8895/0.9093. Used pool peak rises by 182,272/364,544/729,088 bytes;
  B2 benchmark reserved peak rises 64 to 96 MiB, while a separate ten-drop
  lifetime capture matches historical B2/B4 reserved values and returns live
  use to zero. Both platform gates and full resource acceptance are pending.
  [Details](docs/results/2026-11-11-phase2-optimization/packed-qkv-v1.md).

- **Fixed decoder paths rejected:** precomputing decoder weight paths with
  compile-time literals showed no material improvement in a three-process CUDA
  comparison; the known B1 timing clusters appeared in both binaries. The patch
  was reverted and evidence retained. The local release `codec_bench` binary
  was rebuilt for the packed Q/K/V candidate. The BERT
  candidate and both final performance gates remain pending.
  [Details](docs/results/2026-11-11-phase2-optimization/b1-weight-paths-v1.md).

- **BERT checked-load candidate:** authenticated WordPiece vocabulary parsing
  now reserves the fixed 30,522-entry profile capacity and delegates all model
  behavior to pinned `tokenizers`. Three matched CPU processes per binary
  measured median checked load 3.9094 to 2.4690 ms; one allocator capture
  reduced requested bytes 7,776,931 to 5,613,868. Tokenizer tests, Clippy,
  real-asset goldens/rejections and benchmark output hashes pass. Source
  fingerprint inventories include the new parser. Fresh full Linux/Apple gates
  remain pending; prior 76/78 is for the previous candidate.
  [Details](docs/results/2026-11-11-phase2-optimization/bert-wordpiece-v1.md).

- **Fresh Linux performance gate:** candidate `a6290db` completed the 45-process
  three-way matrix and one predeclared nine-process-per-runtime follow-up for
  codec, BERT, Qwen and TinyStories. The initial audit is 71/78; the controlled
  audit is **76/78**, with B1 resident-block latency and BERT checked load still
  failing against accepted Rust. No limits changed; the archived audit reproduces
  byte-for-byte. Apple and the separately documented reporting failure remain
  pending. [Details](docs/results/2026-11-11-phase2-optimization/finite-scan-v1.md#fresh-linux-performance-gate).

- **Grouped finite scan candidate:** model loading now checks little-endian FP32
  exponent bits in groups, preserving complete preflight before upload. Focused
  edge/unaligned and invalid-weight tests pass. A 30-trial CPU scan diagnostic
  improved 4.359 to 2.995 ms; three matched CUDA processes per binary measured
  checked-load medians 192.200 to 189.226 ms, with twelve matching output-ID
  hashes. The combined `tools/check.sh --cuda` stopped before CUDA after one of
  482 reporting tests failed in a CPU-only lifecycle child observation; the CUDA
  portion passed separately, including 18 operators. CUDA trained qualification
  passes 642 traces/2,500 decisions, zero ID disagreement, two near ties; resource
  and both 78-cell gates remain pending.
  [Details](docs/results/2026-11-11-phase2-optimization/finite-scan-v1.md).

- **Streamed artifact hashing rejected:** hashing each artifact during its bounded
  read retained all checks but increased matched CUDA checked-load process median
  from 191.25 to 201.98 ms (three processes each). The patch was reverted; raw
  evidence is retained. The benchmark binary has since been rebuilt for the
  finite-scan candidate. The earlier Linux audit was 72/78; the public
  raw-upload API decision is pending.
  [Details](docs/results/2026-11-11-phase2-optimization/streamed-hash-v1.md).

- **Performance work resumed:** decoder trace-label elision was prototyped and
  measured on CUDA against the saved pre-change candidate in three matched
  processes each. Output-ID hashes matched but no material speedup was measured;
  the patch was reverted and negative evidence retained. Its benchmark binary
  has since been rebuilt. The earlier Linux audit was 72/78; the user's
  decision on the proposed public raw-upload API is pending.
  [Details](docs/results/2026-11-11-phase2-optimization/decoder-trace-label-v1.md).

- **Observer startup evidence:** a CPU-only stress probe reproduced the
  `collector_command` failure at job 1 on attempt 54; the owned child still had
  the kernel `PF_FORKNOEXEC` flag during the follow-up. Fake reset and subreaper
  restoration succeeded. A vfork-based spawn alone is not an observation-ready
  barrier. The observer remains fail-closed and the approved collection rules
  are unchanged; launch and hardware gates remain open.
  [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#follow-up-fork-without-exec-flag).

- **Evidence ownership recheck:** bounded evidence metadata comparisons now include
  UID and GID explicitly, with a focused regression. This strengthens the existing
  same-owner reader only; mixed root/user evidence, authenticated entry and actual
  GPU restoration remain unresolved. Three focused tests pass. The default check
  hit the sandbox's sccache restriction; `RUSTC_WRAPPER= CARGO_BUILD_JOBS=4
  ./tools/check.sh` failed one of 482 reporting tests on the recurring
  `collector_command` mismatch at job 6. Fake reset succeeded within deadline and
  subreaper state was restored. Log: `/tmp/ncm-evidence-owner-check-no-cache.log`,
  SHA-256 `0f05934a28e213bf80ac917ca42e2ee5cd122eedfce25f166500d81a54469635`.

- **Grouped fake handoff suite:** a checksum-bound CPU-only suite combines the
  actual armed-handoff validator, fake restoration supervisor, privilege drop,
  user-owned identity writer and offline receipt binding. It checks both an
  accepted handoff and a deliberately altered command. Both the ordinary-user
  self-test and the grouped user-run root-mode suite passed; the root-owned
  summary and hash were verified. The test-only outer subreaper and unauthenticated fake declarations
  are not a production entry or hardware restoration proof. [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#grouped-fake-handoff-suite).

- **Fake parent-loss follow-up:** a versioned CPU-only test now kills its own fake
  controller after a worker starts, then checks that the surviving supervisor
  cancels the worker and resets fake state while a separate test child survives.
  The ordinary-user self-test passed; it did not drop privileges. The user-run
  root-mode test passed all three fake cases, including parent-loss restoration;
  its root-owned summary and digest were verified. The test-only subreaper is not the proposed
  live coordinator ancestry. [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#fake-parent-loss-follow-up).

- **Fake root-supervisor/user-worker test:** a checksum-bound CPU-only harness now
  combines the existing restoration supervisor, fake set/reset file and a worker
  that drops to the invoking user. An ordinary-user self-test passed normal and
  deadline cases without a privilege drop. The user then ran the sudo test: the
  root supervisor, dropped-user worker and fake reset/cleanup passed normal and
  deadline cases; the deadline transaction failed as intended. The root-owned
  summary and hash were verified outside the sandbox. The versioned script is
  retained in the results tree. Neither run
  establishes a real clock operation, coordinator wiring, parent-loss handling,
  GPU restoration or readiness. [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#fake-root-supervisor-and-user-worker).

- **Inherited capability fix:** the user-run CPU-only root test identified
  nonzero inheritable capabilities as the only failed identity comparison.
  `clock_drop_privileges.py` now clears all thread capability sets with Linux
  `capset` after the UID/GID switch and still independently verifies all four
  sets empty. Thirty-six focused tests and 481 default reporting tests pass,
  including ABI/error tests and a real disposable ordinary-user `capset` call.
  The user-run revised fake root test passed: the normal-user drop and empty capabilities survived fake exec, root
  ID regain and root-file access were denied, and the child was reaped. This is
  user-reported CPU-only evidence, not privileged handoff or hardware restoration.
  [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#clearing-inherited-capabilities).

- **Identity-aware finalizer (schema 2):** the clock finalizer now rejects old
  schema-1 inputs and requires a declared worker identity plus the separate
  identity record. It validates receipt/PID/timing consistency, record digest and
  clock-domain agreement before the unchanged final frozen-file recheck, and
  publishes a separate identity result. Input and completion schemas explicitly
  advance to 2; target/producer authentication and continuous identity remain
  false. Fifteen focused synthetic/CPU tests pass, including missing, mutated,
  transplanted and overclaiming identity cases. Default checks passed 477
  reporting tests, before the last focused-only fault additions. This is not a launch entry or a
  hardware gate. [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#identity-aware-finalization-schema-2).

- **Worker identity evidence:** the worker writes a new private, bounded version-1
  `worker-identity.json` record before sequence setup, and a separate offline
  validator binds its declared identity and clock/PID to supplied worker and
  sequence receipts. Thirty-two focused tests pass, including a real CPU-only child
  writer/reader transport. Default checks passed 475 reporting tests; the later
  real `no_new_privs` FFI test passed separately. The initial fixture used fractional timestamps and
  failed strict integer validation; corrected fixture passes. This is a new
  artifact. The separately versioned schema-2 finalizer now consumes it, but
  neither target nor producer is authenticated. Root/user output ownership,
  real privilege drop, launch entry, policy restoration and hardware gates remain.
  [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#worker-identity-evidence).

- **Worker privilege-drop wiring:** the import-only worker wrapper refuses mixed
  or root credentials without an explicit target, checks its armed handoff, then
  drops root and verifies the resulting IDs/groups/capabilities before building
  observers or starting jobs. Ordinary workers are also sampled before work.
  Twenty-one combined focused tests pass; the new wrapper tests stub the actual
  drop and sequence. Default checks passed 464 reporting tests, begun before the
  final focused test additions. Target identity authentication, a root-owned launch snapshot,
  user-owned outputs, authenticated identity declarations, role-entry integration and
  an integrated root transition remain pending. The user-run CPU-only sudo test exited
  125 during verification. A stage probe showed the prctl, setgroups, setresgid
  and setresuid calls succeeded. A field-level probe identified nonzero
  inheritable capabilities. The revised fake root drop/exec test passed; live
  handoff and hardware restoration remain unproved.
  [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#worker-privilege-drop-wiring).

- **Dedicated-worker privilege drop:** `clock_drop_privileges.py` provides an
  import-only Linux `-S` child transition: reject unexpected descriptors, set
  `no_new_privs`, switch supplementary groups and all GIDs/UIDs, clear thread
  capability sets, and sample credentials. Any post-transition error terminates that child. Thirteen combined
  focused tests pass, including mocked failures and a real extra-descriptor check.
  Default checks passed 459 reporting tests (started before the extra-descriptor
  test was added). The checksum-bound CPU-only sudo test exited 125 during the
  transition; stage diagnostics identify the failure at identity/capability
  verification; the field-level probe identified inherited capabilities. The
  revised fake root drop/exec test passed. This changes
  the candidate worker's inherited security state and is not wired to a launch
  entry; trusted identity/source, restoration, handoff, performance effects and
  hardware gates remain unproved. [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#dedicated-worker-privilege-drop).

- **Collector-command investigation:** an instrumented CPU-only stress batch
  reproduced the mismatch on run 28, job 11. Follow-up still saw the collector's
  command and executable; procfs reported `running`, so the syscall could not be
  identified. Fake reset and reaper restoration succeeded. Kernel-source and local
  Python inspection make an exec-transition race plausible, but do not prove it.
  A new regression keeps collector-command mismatches fail-closed; all 13 focused
  observer tests pass. No diagnostic acceptance or timing changed. [Evidence](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#collector-command-investigation).

- **Worker credential sampling:** `clock_unprivileged_identity.py` verifies a
  supplied non-root UID/GID and exact supplementary-group multiset against
  `/proc/self/status` and process APIs, including saved/filesystem IDs and zero
  inheritable/permitted/effective/ambient capabilities. Seven focused tests pass,
  including this process; duplicate sandbox-mapped group IDs are preserved. It
  is import-only and not wired to a privileged entry; identity declarations,
  continuous state, safe privilege drop, handoff and launch remain unproved.
  Full default check failed 1 of 452 tests on the recurring CPU handoff mismatch;
  its new classification is `collector_command` at job 15, with fake reset
  succeeding. The run started before the final seventh focused test was added.
  [Details](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#sampled-worker-credentials).

- **Observer mismatch diagnostics:** the existing guard still immediately fails
  every command mismatch, but records a fixed non-sensitive category: other
  declared phase, collector command, or unclassified. Twenty-nine focused tests
  and default checks with 445 reporting tests pass; an added actual-mismatch
  category assertion passed separately. No unexpected argv/environment bytes are
  serialized. [Scope](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#observer-command-mismatch-classification).
  A later full check reproduced a mismatch at job 15, classified as the
  collector command; earlier job-10/job-12 categories remain unknown. This does not
  relax observation or establish hardware readiness.
- **Worker role dispatch:** `clock_worker_role.py` binds the supplied quiet/role
  declaration, private snapshot, live restorer handoff, original absolute deadline
  and observed 18-job sequence. Eleven focused wrapper/binding tests pass with the
  new actions stubbed. A separate 16-test combined run failed the recurrent real
  CPU handoff fixture at job 12 (`live child command mismatch`); fake reset/cleanup
  succeeded, cause still unknown. A subsequent full check passed 444 reporting
  tests without explaining that failure. [Scope](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#worker-role-dispatch).
  No observer guard, timing or GPU policy changed. Authenticated entry, privilege,
  restoration, fresh quiet window and both performance gates remain pending.
- **Observer role dispatch:** `clock_observer_roles.py` wraps the existing sibling
  observer supervisor/telemetry collector after supplied quiet-window and sampled
  role binding, snapshot re-read and source-digest checks. Ten original focused
  tests and default checks with 441 reporting tests passed. A later nine-test
  focused run includes real CPU0 parent/observer-role transport through the live
  binding into a stub dispatch; no telemetry/GPU work occurred.
  [Scope](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#observer-branch-role-dispatch).
  Caller authentication, privilege/restoration entry, real telemetry and quiet
  window remain pending; the wrapper is not a launch authorization.
- **Read-only launch-preparation preflight:** `clock_preflight.py` composes fresh
  file and GPU metadata, compares reviewed hashes/records, repeats Git checks,
  and rehashes declared files after GPU queries. It retains the current GPU record
  by digest. Nineteen initial focused tests and default checks with 436 reporting
  tests passed before GPU-record publication and output-isolation refinements;
  ten updated focused tests pass afterward with stubbed producers. An actual
  read-only preflight at clean `6f118be` compared fresh snapshots and GPU metadata,
  passed late Git checks and rehashed 329 files / 4,020,276,376 bytes and 56
  aliases. It started no collection and leaves readiness/permission/restoration
  false; later HEADs require fresh preparation. [Evidence](docs/results/2026-11-11-phase2-optimization/linux-clock-readonly-preflight.md).
  This is not a launch entry and cannot establish producer authentication, role
  ancestry, privilege, restoration, quietness or acceptance.
- **GPU metadata comparison:** `compare_clock_gpu.py` checks supplied read-only
  records' GPU/utility identities, exact query receipts, row counts and capture
  digests without IO; it also binds the exact utility identity to a supplied file
  snapshot. Eight focused tests pass; default checks passed 431 reporting tests
  before the binding addition, which was then exercised in the focused tests.
  Actual retained file/GPU v1 records bind,
  but remain explicitly unauthenticated and not launch-ready. [Scope](docs/results/2026-11-11-phase2-optimization/linux-clock-gpu-preparation.md#comparing-supplied-gpu-records).
  No fresh producer authentication, GPU-lock permission, enforcement or
  restoration follows from comparing records.
- **Read-only GPU metadata component:** `prepare_clock_gpu.py` declares two
  bounded utility queries, validates expected UUID/PCI/name and a 1710 MHz
  application-clock-list row, and rechecks utility identity. Three new stubbed
  tests and 16 combined focused tests pass; default checks pass 429 reporting
  tests. [Scope and commands](docs/results/2026-11-11-phase2-optimization/linux-clock-gpu-preparation.md).
  The committed component also ran outside the sandbox: actual identity matched,
  114 application-clock rows include 1710 MHz, and readiness remains false.
  Retained hashes and exact command are in the linked report. Lock support/
  permission, enforcement, policy, quietness and restoration remain unproved.
- **Launch-time file-snapshot comparison:** `compare_clock_snapshots.py` checks
  the fixed 18-job plan, declared recheck counts and worktree state, then compares
  stable file/loader/upstream/environment/timezone fields while allowing probe
  receipts/times and private session-entry counts to vary. Eighteen focused tests
  pass; default checks pass 426 reporting tests. Retained v1 self-comparison is
  only parser/consistency evidence. [Scope](docs/results/2026-11-11-phase2-optimization/linux-clock-file-snapshot.md#comparing-a-reviewed-snapshot-with-a-fresh-preparation).
  The current record's producer and role identity remain unauthenticated, with no
  hardware permission, restoration, quiet-window or launch claim.
- Read-only outside-sandbox GPU identity matched the historical RTX 3060 Ti UUID
  and PCI; supported graphics-clock listing includes 1710 MHz (application-clock
  list, not proof of GPU-lock permission or enforcement). Sandbox cannot reach the
  driver. [Commands and retained list](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#read-only-gpu-identity-and-supported-clocks).
  Identity must be rebound during final preflight; no clock change or benchmark.
- **Sampled live role binding:** `clock_role_binding.py` checks a private snapshot
  digest, recomputed command, role/parent/coordinator image, environment, cwd, CPU0
  affinity and sampled ancestry without executing any diagnostic role. Five tests
  pass, including an actual CPU0 parent/child `-S` Python transport plus negative
  snapshot, parent, environment and ancestry cases. [Scope](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#sampled-role-binding).
  Caller-supplied coordinator declaration/snapshot origin, safe earlier ancestry,
  continuous liveness and dispatch remain unauthenticated. Default checks pass 438
  reporting tests after the cwd addition. Read-only noninteractive sudo policy listings permit both exact
  proposed commands for the historical UUID, while the outside-sandbox user has
  no effective capabilities. Actual permission/enforcement and safe cleanup of
  privileged children remain unproved; the current plan invokes nvidia-smi directly.
  A single harmless `sudo -n /usr/bin/sleep 3` under the bounded-child primitive
  exited immediately with “a password is required”; no privileged sleep ran and
  cleanup behavior was not tested. User requested a sudo prompt and ran `sudo -v`
  in their own terminal; the agent's `sudo -n true` and the user's same-terminal
  detached fake test both still required a password. No privileged sleep ran, no
  password was shared and no GPU clocks changed. A reviewed interactive privilege
  and cleanup design remains necessary. The user ran the checksum-bound CPU-only
  root fake-restoration test: normal and deadline cases each reset the fake state
  and completed owned-child cleanup. Root-owned summary and hash verified; this
  does not prove unprivileged-coordinator wiring or hardware restoration. A
  same-terminal process-group probe did not identify the root child. A stronger
  fake marker probe confirmed a root child started and received SIGTERM; it does
  not prove complete privileged cleanup, parent-loss handling or restoration.
  Local no-TTY probes failed to authenticate. A fresh CPU-only 18-job handoff
  test passed once, followed by five and then twenty consecutive passing CPU-only
  runs; these do not resolve prior intermittent mismatches. The user-run root/ordinary-user
  access probe found the fake root obligation, parent environment and parent
  executable inaccessible to UID 1000; parent command and children remained
  readable. Root/user handoff, evidence ownership and direct-child bindings
  therefore need a redesigned and tested privilege boundary. [Probe evidence](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#potential-terminal-owned-process-group-probe).
  A separate unprivileged PTY probe found that a new process group retained its
  controlling TTY while a new session did not; this does not prove sudo cache or
  privileged cleanup. [Evidence](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#unprivileged-tty-topology-observation).
  The local sudo manual notes per-terminal caching by default, unrelayable SIGKILL,
  and policy-gated `sudo -T`; these are not proof of active configuration. A safe
  privileged-child timeout/cleanup design remains unverified. A five-minute
  default sudo cache would also be shorter than the 600-second lock window; no
  actual local expiry is inferred. [Topology constraint](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#credential-lifetime-and-supervisor-topology).
  No real clock command was run.
  [Permission evidence](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md#noninteractive-permission-investigation).
  No launch or GPU action occurred.
- **Clock role declaration plan:** `clock_role_plan.py` constructs fixed role
  argv, CPU placement and existing capture/deadline budgets without executing or
  authenticating anything. Nineteen focused tests and 420 default reporting tests
  pass; the full run began before the private-copy/UUID argument refinement, which
  passed its focused tests afterward. The future entry must be
  in a fresh snapshot; v1 predates it. [Plan and limitations](docs/results/2026-11-11-phase2-optimization/linux-clock-role-plan.md).
  No live role handlers, authenticated ancestry, permissions, hardware restoration
  or fresh quiet-window confirmation exist; no GPU/policy/benchmark action ran.
- **Preparation-only snapshot assembler:** `prepare_clock_snapshot.py` composes
  native, upstream, Python/startup, preload, loader, worktree and tool/utility
  records, independently validates all 18 fixed codec jobs, and rechecks supplied
  files before exclusive private output. Sixteen focused tests and default checks
  with 418 reporting tests pass; focused tests stub the producers and cover invalid
  commands and failed rechecks. [Scope and commands](docs/results/2026-11-11-phase2-optimization/linux-clock-file-snapshot.md).
  The first real metadata-only assembly at clean `ff8d370` rechecked 317 files /
  4,020,208,680 bytes and 56 aliases, with 18 planned jobs; the collection path
  was not created and readiness remains false. Snapshot/hash and limitations are
  in the linked report. It must be refreshed against a later committed HEAD. No
  launch entry, GPU identity, permissions, hardware restoration or quiet-window
  claim; Linux still 72/78.
- **Precollection declared-file recheck:** `verify_clock_preparation_files.py`
  inventories all admitted file types/identities/sizes before hashing, verifies
  digests and then rechecks visible metadata and exact alias resolution. Five
  temporary-file tests and 25 combined focused tests pass, including replacement
  and alias mutation. The first fixture's error-text mismatch was corrected; no
  guard changed. [Scope and commands](docs/results/2026-11-11-phase2-optimization/linux-clock-inventory-composition.md#precollection-file-recheck).
  The first 413-test full run failed the previously intermittent CPU-only
  clock-handoff and read-only lifecycle integrations. The former records a live
  child command mismatch at job 10; fake reset/cleanup succeeded. Both tests pass
  in a single focused rerun, and a full rerun passes 413 reporting tests; cause
  remains unestablished and the failed run remains retained.
  This is not yet wired to a fresh authenticated entry and makes no completeness,
  atomicity, hardware, policy or launch claim; final postflight remains required.
- **Upstream inventory binding:** the pure clock inventory composer now requires
  an upstream snapshot record, validates its pinned commit/scope/source inventory,
  and includes all source plus Git executable/config/index identities. Actual v4
  record gives 42 identities, four empty sources; old mixed-record compatibility
  admits 269 files but remains incomplete and not fresh. Nine focused tests and
  default checks with 408 reporting tests pass;
  [scope and evidence](docs/results/2026-11-11-phase2-optimization/linux-clock-inventory-composition.md#upstream-producer-binding).
  This checks supplied records, not producer authenticity or current bytes. Fresh
  full preparation, roles and hardware gates remain; no GPU/policy/benchmark run.
- **Reusable upstream preparation:** `prepare_clock_upstream.py` turns the
  earlier pinned-tree inspection into an import-only fresh-output component,
  requiring a reviewed Git configuration hash and rejecting ignored entries.
  Three temporary-Git tests cover clean/empty files, drift, untracked/ignored,
  wrong pin/configuration and output containment; seven focused tests pass.
  Default checks passed 406 reporting tests; started before the added explicit
  configuration-hash guard, which was then validated separately in focused tests.
  An actual metadata-only v4 snapshot confirms the pinned commit, 39 files /
  877,235 bytes, no ignored entries and `launch_ready=false`. A missing-hash
  review finding was fixed before v4. [Evidence and scope](docs/results/2026-11-11-phase2-optimization/linux-clock-upstream-verification.md#reusable-preparation-component).
  Fresh full preparation, authenticated role entry, permission/restoration and
  fresh quiet-window gates remain; no GPU/clock/benchmark action occurred.
- **Inventory composition follow-up:** `clock_preparation_inventory.py` combines
  supplied native/Python/loader/preload declarations and explicit other groups
  with existing admission checks, without IO or launch-readiness claims. Retained
  records exposed four valid empty upstream files and twelve loader aliases with
  `..`; frozen-file declarations now accept empty regular files while executable
  identities stay nonempty, and loader aliases retain their original spelling
  and must resolve to their pinned targets. Symlink-sensitive regression, 28
  focused tests and default checks with 403 reporting tests pass. The retained
  old-record composition admits 267 files /
  55 aliases but omits newer tools, so is not fresh or complete preparation.
  [Details, initial failures and limitations](docs/results/2026-11-11-phase2-optimization/linux-clock-inventory-composition.md).
  Fresh authenticated preparation, entry/role binding and hardware gates remain;
  no GPU command, policy action or benchmark. Linux 72/78; Apple pending.
- **2026-11-13 continuation:** user approved raising only the clock diagnostic
  postflight count from 256 to 512 and confirmed default GPU clocks with no
  custom lock or competing controller. `verify_clock_files.py` now uses 512;
  unchanged byte/alias/collection/telemetry/deadline/acceptance limits still apply.
  Focused file/finalizer tests pass 17 cases with `umask 077`, including 261/303
  admitted, 512 boundary and 513/post-only overflow rejected. An initial run
  with the inherited broader umask failed three private-directory fixture tests;
  no security check was relaxed. `umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh`
  passes all default checks, including 397 reporting tests. Details are in
  [the admission follow-up](docs/results/2026-11-11-phase2-optimization/linux-clock-preparation-admission.md#approved-count-increase).
  This confirmation is not a fresh quiet window or proof of permissions/restoration;
  complete frozen preparation/entry and live gates remain pending. Linux 72/78,
  Apple pending.
- **Earlier diagnostic authorization (supersedes historical choices below):** user
  approved read-only investigation option 1, followed by clock-controlled option 2
  only if option 1 provides insufficient data. Read-only v1 ran once after explicit
  fresh quiet-window confirmation and failed at the fixed capture limit; no retry.
  Clock fallback is still unrun; the user has since confirmed default-policy/
  no-controller state as above. Permissions, tested hardware restoration and a
  new quiet window remain needed; continuation establishes none of those facts.
  [Selected protocol and durable preparation evidence](docs/results/2026-11-11-phase2-optimization/linux-readonly-trace-v1-proposal.md)
  cover fixed sequence/quotas, report receipts, trace joins/output qualification,
  frozen file/environment preflight, loader/Git guards and the worker supervisor.
  Prior metadata checks match 40 pinned files and 14 canonical libraries; clean
  worktrees verified at `156d976`/`3307dff`. Byte/receipt consistency is not live
  provenance; DT_NEEDED listings do not cover lazy driver/JIT libraries. Source/
  lock remain anchored to `d44beaa`, manifests to measured v2 `fa040ed` (retaining
  the approved codec_memory example). Raw Git status logs remain private.
  Telemetry/coordinator/validator prerequisites are implemented and tested with
  synthetic observers and archived v2 CPU compatibility. Ordinary sequence failure
  preserves the full sampler lifetime; observer failure cancels in-flight work.
  The sampler starts with affinity {0,2}, then confirms {0}; other observers use
  CPU0. Explicit environments differ from inherited v2. All 901 CPU samples,
  counter/schedule arithmetic, GPU units/UUID/timezone and 12 receipt brackets are
  checked offline. Gaps remain explicit, not per-call clocks or continuous coverage.
  Exact earlier tests, failures and limitations are in the selected protocol.
  `observe_readonly_process.py` now samples only owned waitable children: executable
  inode, expected command/environment equality, affinity and executable mappings.
  Optional hooks connect bounded capture, sequence and coordinator; observation
  failure stops later jobs. Eleven CPU-only tests include all 12 fresh children;
  default checks pass **196 reporting tests**. Initial failures exposed differing
  kernel-map/path device numbers (28/29): both are now retained, with kernel mapping
  symlink and path-inode checks, not conflated. Direct map_files stat was denied;
  no permissions changed. No mapped-byte/whole-lifetime proof is claimed, and no
  GPU/model run occurred. Follow-up tests fixed silent factory disabling and
  non-atomic exec/exit snapshots: incomplete samples are counted, never promoted;
  later complete workload observation remains required. Startup access denial is
  recorded as not-ready; persistent denial cannot qualify. Detected mismatches
  cannot be erased by a later child exit. Fixed five-case fake-child batches retain
  all four earlier race failures and final five passes. Logs and
  mapping limitations are in the selected protocol.
  Latest: `verify_readonly_processes.py` cross-checks all 12 process records against
  receipts, declared commands/environments, image/injection identities and native
  libraries. Its postflight function rechecks all prepared/declaration/loader/library
  files and current mapped paths, keeping additional libraries explicitly post-only.
  It preserves kernel/path device distinctions and never claims immutable mapped
  bytes. Nine tests include actual CPU-child writer/reader compatibility; default
  checks pass **205 reporting tests**. No GPU/model collection or policy change.
  `prepare_readonly_launch.py` now composes a preparation-only metadata manifest:
  jobs, tools/utilities, loader/Git guards, GPU identity, IANA timezone and limits.
  It requires `python -S`, never creates the collection directory or starts sampling,
  and explicitly leaves readiness/availability/collection false. Eleven offline tests
  pass; default checks pass **216 reporting tests**. GPU/loader/Git probes were mocked
  in these new tests; the real preparation CLI has not run. Source/library rechecks,
  path-alias rejection and partial private evidence are covered. Actual frozen
  manifest/current-clean-state verification and final launch entry remain pending.
  `diagnostic_clock.py` now records explicit UTC/monotonic lifetime envelopes;
  `validate_readonly_lifecycle.py` binds all child/process samples, readiness,
  parent/PID chains and CPU/GPU observer cleanup to supervisor receipts. Six tests
  include twelve actual CPU-only process observations under the independent
  supervisor with five-second synthetic observers; default checks pass **222 tests**.
  An initial generated-fixture NameError and its debug evidence are retained; the
  fixture was corrected without changing production limits. A deterministic
  clock-recording/deadline regression also failed before moving that stamp ahead
  of the final startup guard; the red/green logs are retained. New fields do not
  retrofit historical receipts or prove live provenance.
  `finalize_readonly_trace.py` now composes telemetry, lifecycle, process, trace/output
  and frozen-file validators after a single trusted runner returns. It binds 64 raw
  evidence files and five derived reports, preserves failed attempts and refuses
  reruns. Ten fresh-interpreter tests use real validators over archived outputs and
  synthetic telemetry/maps/traces; full checks pass **232 reporting tests**. Updated
  hashes do not hide bad logits; incomplete/mismatched evidence blocks frozen rehash.
  This is import-only orchestration, not independent live-reaping proof: the final
  entry must still supply reviewed commands and an actual exclusive-reaper runner.
  `readonly_trace_entry.py` now supplies fixed -S/CPU0 roles, live parent-command/
  image checks, exact isolated environments, mandatory native/injection dependency
  observation, actual exclusive-reaper supervision and the composed finalizer.
  Outer cancellation gives the supervisor up to 10 seconds TERM grace before KILL;
  native job limits/cleanup remain unchanged. Ten entry tests include real role
  transport/cancellation with workload/preflight/qualification stubs; full checks
  pass **243 reporting tests**. Initial fixture failures, missing-injection-dependency
  red/green and an error-message regression are retained. No CUDA abort guarantee.
  [Readiness checklist, exact commands and evidence](docs/results/2026-11-11-phase2-optimization/linux-readonly-trace-v1-readiness.md)
  maps requirements to the implementation. Published tooling: `b7f61b5`; two actual
  metadata preparations and launch-time comparison passed at that clean HEAD.
  [Read-only v1 failure and evidence](docs/results/2026-11-11-phase2-optimization/linux-readonly-trace-v1-results.md):
  explicit “quiet window now”; 2026-11-12 00:28:13–00:43:14 UTC, one attempt.
  AP/CP completed; AT exceeded 1 GiB capture, killed/reaped; jobs 4–12 unstarted.
  All 901 CPU samples/full 900-second collection completed; 900 GPU rows. Existing
  separate CPU/GPU content validators pass, not the twelve-job gate. Supervisor
  cleanup complete; eight recorded child PIDs later absent. No retry/quota change.
  Finalizer correctly failed before frozen postflight; no completion/entry result.
  Truncated trace: 1,433/2,008 elapsed queries, no traced report/qualified joins;
  insufficient attribution data, not a clock-cause finding. Initial offline alias
  counting error retained/corrected using existing ELAPSED names. Compact 36-member
  archive published; full 1 GiB capture retained locally by hash, not public raw
  reproduction. Clock-restoration integration and policy/controller/permission
  checks remain before new availability; no clock intervention is ready or performed.
  Supervisor SIGKILL/host failure cannot guarantee cleanup. Linux remains 72/78,
  Apple pending; no GPU-policy/production changes or completion. Post-attempt
  `CARGO_BUILD_JOBS=4 ./tools/check.sh` passes 243 reporting tests; all 36 archive
  hashes and telemetry revalidation pass in clean `git archive b7f61b5` scratch.
  Checks/log hashes and raw-trace reproduction limitation are in the result.
  Latest preparation: `supervised_restoration.py` adds an import-only independent
  set/one-worker/reset transaction, obligation written before set, owned/adopted
  cleanup, parent-loss/interrupt handling and at most one cleanup-only retry.
  No GPU command/CLI is supplied or run. Sixteen fake-policy CPU subprocess tests
  pass, including actual escaped-child reaping before reset, unrelated survivors,
  partial set failure, deadline/overflow, interrupted/timed-out reset, serialization
  exceptions and setup failures. Staged review's reset-timestamp regression failed
  before guarding metadata recording so it cannot bypass reset; red/green retained.
  That checkpoint's 16 focused tests/default checks passed 259 reporting tests.
  Follow-up now separates absolute reset and earlier work deadlines, persisted
  before set: default 600 seconds with a 10-second cleanup reserve. A new owned
  post-Popen timestamp must bound first-reset submission within the deadline;
  late/missing evidence fails even after exit-zero reset. Restoration reports are
  schema 2; bounded-child receipts add a spawn-time field, no historical retrofit.
  Red missing-argument/field tests retained; **34 focused tests and 263 reporting
  tests pass**, including delayed spawn and post-spawn clock-error cleanup.
  [Clock proposal](docs/results/2026-11-11-phase2-optimization/linux-clock-diagnostic-v3-proposal.md)
  records commands/log hashes and limitations: no hard real-time guarantee or
  hardware restoration proof. Next: integrate fixed deadline settings, telemetry,
  enforcement/frozen worker bindings; obtain still-pending default-policy/no-
  controller confirmation and permission evidence, then a fresh window. No real
  GPU command, lock/reset, permission test or benchmark ran in this follow-up.
  Offline runner audit at `c82d847` (then source-identical to v2) confirmed the existing
  previous/Python/candidate cyclic order: 18 codec jobs, not a new runtime order.
  CPU0-only startup is rejected by the required CPU2/CPU0 eligibility checks.
  Source review finds worker-time preflight hashing/probes, inherited environment,
  unbounded per-job calls, internal monitors and misleading no-clock-change metadata
  must be separated/bound before integration. Details/source hash/probe log are in
  the clock proposal. That audit changed no code or live policy.
  Latest: extracted pure `plan_jobs` from the existing runner and made its CLI
  consume the same plan. Three independent fake-CLI command/cwd/order/copy tests
  passed before and after extraction; planner tests reject filesystem IO/probes.
  Six new planner + six affinity tests pass; default checks
  pass **269 reporting tests**. Legacy-green/new-helper-red logs and new runner
  source hash are recorded in the proposal; frozen historical pins are unchanged.
  This is command construction only, not a frozen launch manifest or GPU parity.
  Next: bind the plan into an isolated bounded worker with full telemetry and
  restoration; the existing CLI is still not suitable as that worker. Policy/
  controller confirmation, permission evidence and later fresh window remain pending.
  Further offline preparation adds a trusted nonblocking external cancellation
  hook to restoration: observer loss/errors latch failure, including after a
  complete worker receipt, but never suppress reset. No actual telemetry monitor
  is supplied by the hook. Three new fake-policy tests cover healthy/transient/
  invalid/error/late-loss cases; **22 restoration tests and 272 reporting tests
  pass**. Red/green logs and additive schema-2 fields are in the clock proposal.
  No GPU command or benchmark ran. Remaining work is integrated bounded jobs,
  bound readiness/coverage, enforcement and outer supervision; live policy/
  controller confirmation and permissions remain required before a new window.
  Latest offline component: `clock_codec_sequence.py` executes the fixed 18-job
  pure plan through bounded owned captures, explicit isolated environments,
  mandatory observers and the restorer-supplied absolute work deadline. It stops
  on first failure without retry; Python report copies are deferred. Prospective
  bounds (1 MiB capture/job, 16 MiB post-exit report read, 1 GiB free headroom) and
  limitations are recorded in the proposal; read-only v1 bounds are unchanged.
  Nine real CPU-only/fake-workload tests pass, including all 18 children and
  cancellation/prevalidation failures; default checks pass **281 reporting tests**.
  No GPU/model run or clock command occurred. Still not launch-ready: frozen
  preparation/entry, live observation binding, telemetry/restoration composition,
  final qualification and the unanswered policy/controller/permission gates remain.
  Four new composed CPU-only tests run the real restoration supervisor and bounded
  worker together with fake policy/workloads: exact deadline handoff, 18 successes,
  failed-worker exit, observer loss and deadline abort, owned cleanup before reset,
  and unrelated-child survival. Default checks pass **285 reporting tests**.
  Source audit found an outer-wrapper blocker: read-only TERM grace is capped at
  10 seconds, insufficient for two 10-second reset attempts plus cleanup. Do not
  reuse it for clock restoration without a reviewed/tested shutdown budget.
  Tests use shortened limits, not live CUDA or a production entry. No policy,
  benchmark, permission or grace-bound change in that checkpoint.
  Latest: `clock_restoration_capture.py` adds an import-only 660-second outer
  capture/60-second cleanup grace; generic explicit grace now permits 60 seconds,
  while native default zero and read-only fixed ten remain unchanged. A red test
  exposed stdout backpressure during grace; capture now keeps draining under the
  cap, including after writer failure. A real fake-policy supervisor completes
  two six-second reset attempts and >128 KiB final output after cancellation;
  failed collection/first reset stay failed. Review also caught a selector-error
  path skipping owned reaping; its red/green regression is retained. Final
  **22 focused tests and 292 reporting tests pass**. Proposal/HANDOFF retain red/green logs and limitations.
  No GPU/policy/permission/benchmark run. Still need frozen role/budget binding,
  full telemetry/restoration composition, final qualifications and the unanswered
  default-policy/no-controller confirmation before readiness and a new quiet window.
  Latest offline integration: `clock_diagnostic_coordinator.py` gates restoration
  on telemetry readiness and now requires a separate observer-supervisor branch.
  Readiness PID must be its sole live child in sampled procfs relationships; not
  independent provenance or continuous liveness. Nine clock-coordinator CPU tests
  include two surviving sibling supervisors after coordinator SIGKILL, telemetry-
  worker SIGKILL/session-escaped sampler reaping, unrelated-child survival and
  prior readiness/failure/TERM cases. Generic adoption-and-KILL must not surround
  the restorer. The frozen production entry must still bind this tested role tree.
  A lost warning-pipe regression initially skipped final restoration evidence;
  warning errors now remain failed while role cleanup/result writing proceed.
  An initial test's exact exit-1 assumption failed with Python exit120 after parent
  loss; it now requires nonzero plus actual retained cleanup evidence. All failures
  retained in the proposal. **40 focused tests and default checks with 302 reporting
  tests pass**. Synthetic shortened telemetry/fake policy only; no GPU command,
  permission test or benchmark. Read-only defaults unchanged. Still owed: frozen
  roles/deadlines, actual per-job observation and full telemetry/output qualification;
  default-policy/no-controller confirmation, permissions and a later fresh window.
  New `clock_process_observation.py` prebuilds 18 independent real process observers
  from copied declarations, with per-runtime image/library identities, exact job/env
  binding and one-time ordinal issuance. It preserves Python argv symlink spelling
  separately from ELF identity and performs no construction-time IO/hashing/probes.
  Five new CPU-only tests exercise actual `/proc` observation across all 18 fake
  jobs, live mismatches, missing libraries and prelaunch drift. **25 focused tests
  and default checks with 307 reporting tests pass**. A broader run exposed an
  existing startup fault-injection race; deterministic red/green coverage fixes
  only the fixture, preserving production sampling rules/assertions. Logs/hashes
  are in the clock proposal. Frozen hardware/entry binding and postflight remain
  pending; no GPU command, policy action or benchmark. Linux stays 72/78; Apple pending.
  Latest: `clock_worker_handoff.py` validates a bounded private obligation, fixed
  600/590/10-second budget relationship, reviewed commands and actual worker/parent
  argv/image/env/CPU0/ancestry before returning the original absolute deadline.
  Schema-2 obligations now add supervisor/parent PID fields; no historical rewrite.
  Four CPU-only test methods exercise all 18 jobs with real observations under
  default restoration budgets, plus nineteen rejected handoff cases with reset
  and zero workload submissions. **32 focused tests and default checks with 311
  reporting tests pass**; exact evidence/limitations are in the clock proposal.
  No GPU/policy/permission/benchmark run. This is a tested handoff component, not
  the frozen preparation/production entry or full qualification; those and the
  unanswered policy/controller/permission/fresh-window gates remain pending.
  [Offline Python preparation audit](docs/results/2026-11-11-phase2-optimization/linux-clock-python-preparation-audit.md)
  verifies all 110 v2 members, six Python metadata sets, three historical source
  pins and current 36 distribution versions. No historical interpreter/library
  byte equivalence is established; five current-only file hashes remain snapshots.
  Torch distribution version 2.15.0 differs from runtime build 2.15.0+cu132; retain
  both. Current pip is extra to the benchmark lock; the existing .pth startup shim
  must be accounted for, not disabled. Preserve intrinsic harness hashes/Git checks
  while forbidding added collector probes/early rehash. Documented offline
  reproduction passes; no Torch import, GPU command or benchmark. Docs-only work;
  default checks not rerun (prior fef3b08: 311 reporting tests). Next: explicit
  current-only Python/loader/startup inventory in frozen preparation, then entry
  binding and final qualification. Linux 72/78 and all live authorization gates remain.
  Latest: import-only `prepare_clock_python.py` now binds the seven historical
  records, three source pins, current venv/ELF/link chain, distribution metadata,
  requirements lock, static Torch build tag and explicitly reviewed startup files.
  It executes no interpreter/package/probe and keeps current-only/provenance gates
  false. Seven synthetic tests cover drift, unsupported inputs, inventory races and
  bounds; **12 focused tests and default checks with 318 reporting tests pass**.
  Three actual file-only component preparations agree: 3 historical sources, 43
  current-only files, 37 distributions (extra pip retained), 3 interpreter links;
  launch_ready=false. A retained red/green case now rejects same-byte interpreter
  inode replacement during preparation; final checks bind device/inode/path too.
  Exact commands/hashes are in the Python audit. No Torch import,
  GPU query, clock action or benchmark. Remaining: Python/native loader inventory,
  current upstream check, frozen full entry/role bindings and final qualification;
  policy/controller/permission/fresh-window gates remain unanswered or unverified.
  Latest: [standalone loader preparation](docs/results/2026-11-11-phase2-optimization/linux-clock-loader-preparation.md)
  adds `prepare_clock_libraries.py`: all-six-target prechecks, bounded glibc listings,
  canonical roots/library bounds and final device/inode/path rechecks. Seven new
  tests; **22 focused and 325 reporting tests pass**. Initial fixture-path failure
  retained. Actual preparation: six complete utility receipts, 35 current-only
  libraries/3,547,116,642 bytes; native accepted/candidate resolution agrees.
  Native/Python differ on two cuBLAS SONAME paths. Reviewed Torch startup explicitly
  preloads CUDA libraries, so standalone listings are NOT its runtime mapping
  contract; no environment/library changes or performance attribution. Next:
  explicit preload-candidate inventory/required-map review, current upstream check,
  frozen entry and final qualification. No Torch import/GPU query/clock/benchmark;
  launch_ready=false, Linux 72/78, Apple pending; live policy gates unchanged.
  Latest: file-only `prepare_clock_preloads.py` inventories the three reviewed Torch
  preload layouts in one site-packages root without importing/executing candidates.
  Six new tests; **20 focused and 331 reporting tests pass**. Actual inventory:
  16 source entries, 15 ELF candidates/1,956,767,857 bytes; optional NVTX pattern
  absent here, no ambiguous matches. Twelve identities overlap the standalone
  inventory; cuSOLVER/nvJitLink/NVRTC-builtins add three current-only files (38-file
  union across preparations, not a new simultaneous snapshot). Runtime selection,
  complete sys.path/closure and launch readiness stay false. Loader document now
  records a proposed mandatory core-map subset versus frozen optional candidates.
  Next: implement/test required-map and known-candidate binding, current upstream
  check, frozen entry/final qualification. No GPU query/Torch import/benchmark or
  policy change; live authorization/availability gates and Linux 72/78 remain.
  Latest: known-candidate sampling rejects first-sample identity drift without
  requiring optional presence; observed fields cannot override declarations.
  `clock_library_contract.py` now purely joins prepared records and selects native
  required mappings plus interpreter/_C/core-Torch mappings. No IO/hashing/probes.
  Retained-record compilation: required 11/11 native, 10 Python; 40 known identities,
  50 path bindings; all eighteen commands construct observers without live work.
  Latest: [clock process-record validator](docs/results/2026-11-11-phase2-optimization/linux-clock-process-validation.md)
  independently checks eighteen receipts, mapping digests/identities, mandatory and
  optional candidates, sequence-local clocks/deadlines and declared injection absence.
  Six new tests include 40 negative variants and persisted writer/reader evidence
  from eighteen real CPU-only fake jobs. **23 focused and 344 reporting tests pass**.
  This is not producer authentication, restoration/telemetry/output qualification
  or final file rehash. Latest: `verify_clock_files.py` rechecks supplied frozen
  files, known libraries/images, alias targets and current mapped paths, with unknown
  libraries explicitly post-only. Nine new tests include real eighteen-job CPU-only
  observer→validator→file-check transport; **31 focused/353 reporting tests pass**.
  The first full check failed an existing read-only lifecycle sequence; cleanup and
  telemetry completed, but its assertion omitted sequence details. That diagnostic
  is now included; one focused and one full rerun pass, cause unestablished/red log
  retained. No guard/budget changes. [Scope and evidence](docs/results/2026-11-11-phase2-optimization/linux-clock-process-validation.md#final-current-file-component)
  distinguish a file-check component from enforced finalization ordering.
  Latest: [restoration-record validation](docs/results/2026-11-11-phase2-optimization/linux-clock-restoration-validation.md)
  independently checks the armed obligation, 600/10/10 budgets, original deadline,
  set/worker/reset command/clock order, first-reset spawn and cleanups. Six new tests
  include 52 negatives and persisted default-budget fake-policy/eighteen-CPU-job
  transport; **18 focused/359 reporting tests pass**. Initial focused discovery
  duplicated six imported tests; module import corrected it, original log retained.
  Latest: [sibling lifecycle validation](docs/results/2026-11-11-phase2-optimization/linux-clock-restoration-validation.md#sibling-lifecycle-follow-up)
  binds that transaction to coordinator/observer-supervisor/telemetry parent chains,
  readiness clocks/digest, outer 660/60 capture, CPU-sample brackets and eighteen job
  receipts. Five new tests include 40 negatives and real sibling fake-policy/default-
  budget transport with five-second synthetic telemetry/stub process observations;
  **26 focused/364 reporting tests pass**. No live ancestry/provenance, mapped-byte/
  output quality or hardware restoration claim. Latest: [clock telemetry contents](docs/results/2026-11-11-phase2-optimization/linux-clock-telemetry-validation.md)
  reuses unchanged CPU/GPU parsers for all 901 samples, binds raw receipts/readiness
  and brackets the restorer plus eighteen jobs. Six new tests cover 43 negatives;
  **21 focused/370 reporting tests pass**. Retained red/green readiness-time cases
  reject local time before sampling and a first GPU sample after readiness. Virtual-
  clock serialization/sparse GPU fixtures are not real collection or enforcement;
  gaps/unavailable values remain explicit. Latest: [clock report qualification](docs/results/2026-11-11-phase2-optimization/linux-clock-output-qualification.md)
  checks all six reports/runtime in fixed 1–3/4–6 halves with the unchanged codec
  checker and frozen Python reference. Current Python outputs compare directly to
  frozen Python, avoiding tolerance chaining. Three new tests include eighteen
  hash-verified historical report members/synthetic receipts and 43 altered cases;
  **7 focused/373 reporting tests pass**. Retained first failure was a no-op order
  mutation in the fixture, corrected without weakening guards. This is not fresh
  collection, capture integrity, producer authenticity or performance acceptance.
  Latest: [sequence evidence-file binding](docs/results/2026-11-11-phase2-optimization/linux-clock-evidence-binding.md)
  reads 74 private files through bounded no-symlink descriptor traversal, binds
  frozen jobs/deadline, embedded/individual records, capture hashes and report
  pathname identities, then rechecks visible metadata. Eight new tests include
  real eighteen-CPU-job transport and isolated archived outputs through the actual
  qualifier; **17 focused/381 reporting tests pass**. Capture inode provenance is
  explicitly unavailable (same-byte pre-read replacement allowed); no atomic/file-
  producer/whole-run proof. Latest: [control-side binding](docs/results/2026-11-11-phase2-optimization/linux-clock-evidence-binding.md#control-side-follow-up)
  adds 21 files (95 total in one bounded reader), six declared role receipts/capture
  digests, CPU report inode and GPU digest checks. Observer-supervisor stdout stays
  explicitly post-only; coordinator/observer-supervisor/GPU launch bindings remain
  owed by the frozen entry. Six new tests include predeclared real sibling fake-policy
  transport through the lifecycle validator; **25 focused/387 reporting tests pass**.
  No new producer authentication or live policy proof. Latest: [same-evidence finalizer](docs/results/2026-11-11-phase2-optimization/linux-clock-finalization.md)
  invokes one trusted restoration-aware runner, then gates those 95 files through
  full telemetry, sibling/restoration, process and eighteen-report qualification;
  receipt digests/clock domains agree, metadata rechecks precede frozen rehash last.
  Exclusive private completion marker appears only after all gates/output writes.
  Four new test methods use actual validators with archived outputs/synthetic clocks,
  maps and policy; 15 gate failures/two publication failures retain partial evidence.
  **18 focused/391 reporting tests pass**. Import-only composition is not a frozen
  entry or independent reaping/provenance/hardware proof. Latest: [current upstream snapshot](docs/results/2026-11-11-phase2-optimization/linux-clock-upstream-verification.md)
  at `f536960` verifies raw pinned commit `c702742`, replacement objects disabled,
  HEAD/index and 39 tracked files/877,235 bytes; zero untracked/ignored entries.
  Accepted/candidate Git guards pass, production/manifests unchanged; candidate's
  four allowed session entries stay private (counts only). Ten upstream probes and
  six native guards completed; 25-member compact archive consistency/privacy checked.
  **15 focused/391 reporting tests pass**. V1 retained before stronger v2 commit/object
  checks; no import-closure/historical-library/atomic-snapshot or launch claim.
  Next preparation audit found a [file-count admission blocker](docs/results/2026-11-11-phase2-optimization/linux-clock-preparation-admission.md):
  current tool paths plus retained declarations form 303 unique planned files; even
  a 261-file subset exceeds the clock postflight's 256-file cap. This is path-count
  evidence, not fresh target hashes or complete inventory. **9 existing file tests
  pass**; default suite not rerun (prior 391). Requested approval to raise only the
  clock postflight count to 512, preserving 8 GiB total/2 GiB per-file and every
  collection/deadline/telemetry/acceptance limit. No change applied; do not omit files
  or split batches to evade the cap. Need that decision AND the unanswered default-
  clock/no-custom-lock/no-controller answer; continuation establishes neither.
  Safe offline progress while those answers remain pending: [early fail-closed admission](docs/results/2026-11-11-phase2-optimization/linux-clock-preparation-admission.md#early-fail-closed-admission-follow-up)
  now shares pure declaration checks between finalizer and postflight. A retained red
  test showed 257 files reached the callback; now oversized/conflicting declarations
  reject before filesystem inspection, output creation or collection. No file IO/
  hashes added to this early check; actual file rehash remains last. Four new tests
  cover exact count/byte/alias boundaries, no-IO admission and seven finalizer faults;
  **17 focused/395 reporting tests pass**. No limit change: the 303-file plan still
  fails the 256-file cap. Fresh complete preparation/entry and live readiness gates
  remain pending. No quiet window requested, GPU/benchmark work or acceptance closure;
  Linux 72/78/Apple pending.
  Further offline safety work: [coordinator local-subreaper admission](docs/results/2026-11-11-phase2-optimization/linux-clock-diagnostic-v3-proposal.md#coordinator-local-subreaper-admission-follow-up)
  rejects an existing process-local child-subreaper role before lifecycle/outputs/branches;
  query failure also stops, and unsupported platforms never query. Two new tests retain
  a red real-flag/mocked-lifecycle regression; **32 focused/397 reporting tests pass**.
  First full check failed a separate default-budget handoff fixture (worker exit 1,
  fake reset/cleanup successful); log retained, cause unestablished, now includes
  bounded sequence/worker diagnostics on recurrence. No timing/expectation relaxation.
  Local flag checking does not inspect ancestors or prove safe launch topology. Both
  user answers, fresh preparation/entry and live gates remain pending; limits unchanged.
  A [fixed six-attempt CPU-only handoff reproduction](docs/results/2026-11-11-phase2-optimization/linux-clock-diagnostic-v3-proposal.md#fixed-cpu-only-handoff-reproduction-batch)
  at `1ec6aaa` retained all 108 jobs and six successful fake resets/cleanup records.
  All attempts passed; the original failure did not reproduce and remains unresolved.
  Independent file audit verifies original deadlines, observations/reaping records and
  a private 566-file/1,930,063-byte inventory. Not clock preparation or a cap bypass.
  No timing/source/policy/limit changes; default suite not rerun for this evidence-only
  checkpoint (prior 397 pass and failed full run both retained). No further batch;
  await requested decisions and complete frozen preparation/entry before live gates.

- Public repository: https://github.com/lexlapax/ncm-rs, `main`, remote `origin`.
- **Phase 2 is reopened for the additional five-step optimization plan approved
  on 2026-11-11.** The user requested the revised plan be committed/pushed before
  implementation. Phase 3 remains unstarted and blocked on this additional work.
  The earlier remediation is complete: production candidate `083c1e5`, evidence:
  `docs/results/2026-11-10-phase2-acceptance/README.md`. Both platforms pass all
  78 comparisons against previous Rust and qualified Python under unchanged
  limits. Final fresh CUDA, then Metal captures pass 642 traces/2,500 decisions
  each, zero ID disagreements; full real-device checks and tokenizer probes pass.
  CUDA codec latency and Apple checked-load/host-memory deficits are remediated.
  Original/intermediate failures remain historical evidence, not overwritten.
  User authorized implementation, research and commit/push for this extension.
  Phase 3 is not started and needs the next work authorization; no generator/CLI
  work was done. The active thread objective now includes completing the expanded
  Phase 2; planning publication alone does not complete it.
  Large licensed release archives remain separately pending authorization; this
  is a scoped codec/tokenizer result, not full-model inference or a release.
- [Approved optimization plan/research](docs/PHASE2-OPTIMIZATION-REVIEW.md):
  (1) freeze/refresh accepted baseline, (2) independent lower-risk experiments,
  (3) Linux gate then Apple gate, (4) latency/CPU/memory/startup/teardown evidence,
  (5) conditional profile-backed escalation and final completion audit. Step 4
  applies throughout. Source hashes/static estimates are under
  `docs/results/2026-11-11-phase2-optimization-review/`; new runtime evidence is
  pending. Baseline is `3307dff` (production `083c1e5`) plus qualified Python;
  unchanged 1.05 median/memory and 1.10 p95 limits apply against both comparators.
  Candidate ledger must cover prepared state, reconstruction reuse, MLX resampling,
  tokenizer APIs/structures, loading copies and conditional higher-cost ideas.
  Preserve historical acceptance/negative evidence and all existing contracts.
  Revised `docs/PLAN.md`, `docs/PHASE2.md`, review and status pages distinguish
  accepted earlier work from pending new gates. No production edits or new GPU
  runs in this planning revision. Default `CARGO_BUILD_JOBS=4 ./tools/check.sh`
  passes (24 report-tool tests); research JSON/hashes/arithmetic and local links
  checked. Plan published as `013f745`. Ask on consequential ambiguity/blockers;
  do not mark the active objective complete. Preserve `.pi/`.
- Additional optimization baseline checkpoint:
  [report](docs/results/2026-11-11-phase2-optimization/README.md), locked
  `acceptance-v1.json` and `baseline.json` pin current accepted Rust/Python before
  production edits. New clean worktree `.reference/phase2-optimization-baseline`
  at `3307dff`; saved binaries in adjacent `-bin` exactly match accepted benchmark
  hashes. Three-process rotating Linux matrix collected, CPU 2, with telemetry;
  all output/point/memory checks pass but diagnostic audit is **73/78**, not an
  acceptance pass (five GPT-2/Qwen process envelopes). Identical binaries show
  ~9% B1 reconstruction placement/process difference; no speedup is claimed.
  Fresh `tools/check.sh --cuda` passes (29 report-tool tests), trained capture
  passes 642 traces/2,500 decisions and cleared-environment tokenizer probes pass.
  Nine-process lifecycle and fresh CUPTI (1,312 kernels) retained. New tested
  `tools/bench/run_process_resources.py` records whole-child CPU diagnostics,
  not request CPU/allocation counts; v1/v2 preserve the inherited pre-exec RSS
  counter finding and explicit v2 labeling/streamed launcher hashing. No global
  settings or unrelated processes changed. No production edits or Apple run yet.
  Clean-checkout baseline summary/audit reproduction and archive/hash checks pass.
  Published baseline checkpoint: `cc965d8`. Its five uncertainty cells remain
  historical evidence, not erased by subsequent runs.
- Provisional lower-risk Linux candidates:
  [report](docs/results/2026-11-11-phase2-low-risk/README.md), source/archives in
  `evidence.json`; complete opportunity ledger at
  `docs/results/2026-11-11-phase2-optimization/experiments.json`.
  Diagnostic-only System wrapper/example counts Rust allocation requests (not
  native allocations or RSS); two explicit example tests pass. HF `encode_fast`
  removes 1,344 BERT allocation requests/89 strings, no GPT-2/Qwen reduction.
  TinyStories in-place all-occurrence merge compaction removes 259 requests/trial;
  exhaustive overlap/rank regression covers 2,047 binary strings plus long cases.
  Neither tokenizer change has an established material latency gain yet.
  Untraced reconstruction reuses encode's latent/immutable snapshot storage via a
  shared decoder tail; traced reconstruction retains independent decode and all
  observations. Extended tiny/trained tests verify untraced snapshots after another
  decode; trained logits/IDs/snapshots match traced values exactly. Fresh CUDA
  checks and 642 traces/2,500 decisions pass; tokenizer goldens/rejections pass.
  Matched B1/B2/B4 medians 3.0003/4.0104/5.1257 -> 2.2853/2.6712/3.3667 ms;
  diagnostic kernels 1,312 -> 918. Fresh lifecycle retained. Matrix is **74/78**:
  unresolved process envelopes at B1 resident block, Qwen decode, BERT/GPT-2 load;
  all pooled points/memory pass. No waiver or final Linux gate; no Apple run yet.
  Clean-checkout summaries/negative audit and evidence hashes verified. Default
  report tests remain 29; 17 tokenizer unit tests and 2 allocation-example tests
  pass. Next: prepared state/loading-copy investigations, matched uncertainty
  controls and full Linux gate before Mac availability recheck and MLX work.
- Prepared-state checkpoint (still provisional):
  [report](docs/results/2026-11-11-phase2-prepared-state/README.md) preserves two
  incremental matrices versus `bcf2be4`, not substitutes for final accepted-Rust
  comparisons. Shared positions/codebook norms and CUDA bounded geometry pass
  fresh 642-trace/2,500-decision qualification; final `tools/check.sh --cuda`
  passes. Three cache unit tests and a dedicated real-CUDA owner/drop/isolation
  test pass; live used-pool memory stabilizes and returns to baseline. Cache is
  backend-owned, clones start empty, logical bounds 64 entries/1 MiB. Profile
  kernels 918 -> 892, then geometry removes 70 H2D copies/24,912 bytes. Latest
  B1 block median ratio **1.1106 versus bcf2be4** remains unresolved; no final
  keep/acceptance claim. Both matrices/profiles/failed attempts are archived;
  clean-source summary/profile reproduction and hashes verified. Earlier 74/78
  audit remains historical, not a gate for this newer candidate. Raw-upload PoC
  tests pass; 58 tensors read back exactly, isolated medians 13.512509/7.379914 ms
  decoded/raw. **Public WeightBackend little-endian byte-upload method needs user
  approval; production loading remains unchanged.** No RSS/full-load claim.
  Next: API decision, hot-name/trace attribution, final resource/lifecycle evidence,
  controlled timing investigation and full Linux gate before Apple. No new Apple
  work, Phase 3 or completion claim. Snapshot binaries at
  `.reference/phase2-geometry-snapshot-bin`; original intermediate positions/norms
  executable hashes are recorded but those binaries are not retained. Preserve
  `.reference/phase2-low-risk-baseline{,-bin}` and all older baselines.
- Linux controlled follow-up remains **incomplete, 72/78**:
  [report](docs/results/2026-11-11-phase2-linux-controls/README.md),
  [working completion audit](docs/results/2026-11-11-phase2-optimization/completion-audit.md).
  Fresh full matrix is 74/78; predeclared 30-process controls for codec/BERT/GPT-2/
  Qwen retain all 360 processes. Six native-comparator cells remain open (B1 block
  median uncertainty; three GPT-2 execution p95 cells; codec/Qwen load p95).
  No noise waiver; all Python/matrix-memory cells pass, not an overall pass.
  Audit adds explicit full-codec control opt-in and binary/Python environment
  binding without changing limits; 36 report tests pass. Fresh CUDA checks,
  642 traces/2,500 decisions, tokenizer probes, nine-process lifecycle and
  45-process CPU/resource output validation pass. Six whole-model probes verify
  60 drops to zero live pool usage, 25,680-byte stable prepared state, and net
  retained-output savings ~1.07/2.16/4.35 MB B1/B2/B4 (not a host-RSS gain).
  Three unchanged-binary GDB captures establish 168 direct model format calls /
  7,992 requested bytes per reconstruction; trace/typed-name plumbing deferred.
  CUDA kernel/layout/graph/table/ownership decisions and tokenizer deferrals are
  recorded in the ledger; MLX-specific work and final dual-platform gates remain.
  A shared-target diagnostic baseline build caused stale release artifact reuse;
  a cache test failed and binary hashes identified old accepted binaries. Local
  CUDA/model release outputs were cleaned/rebuilt; fresh checks pass and frozen
  candidate hashes match again. Earlier measurements predate this incident and
  are binary-bound. **Use isolated targets for future cross-worktree builds.**
  Preserve `.reference/phase2-memory-baseline` (diagnostic example added locally)
  and `.reference/phase2-memory-bin`; no historical baseline overwritten.
  Clean staged-checkout reproduction matches both summaries and all three negative
  audits exactly; source/archive hashes, archive scope and local links pass.
  Offline inspection of every affected native process finds GPT-2/Qwen/codec-load
  slowdowns in both implementations around 05:47–05:49 UTC, plus a later Qwen spike.
  This is temporal association, not established causality or grounds to exclude
  samples. GPU locking is not established as necessary/sufficient for CPU-only
  failures; prioritize a quiet window and read-only CPU scheduling/frequency
  telemetry. The report records ranges/timestamps; all native ranges/peak-repeat
  identities are verified against recorded report hashes, and 36 report-tool
  tests pass. No new runs or policy changes.
  Do not change clocks/power/services or stop unrelated work without approval.
  No Apple run yet.
- **Stopped for handoff at the user's request; expanded Phase 2 is not complete.**
  Read [continuation notes](docs/results/2026-11-11-phase2-optimization/HANDOFF.md)
  before resuming. Latest addition is the read-only Linux CPU sampler and six
  tests (42 report tests total). Ambient observation identifies CPU 2/3 as SMT
  siblings; CPU-3 auditor overlap confounded the earlier resource-only run, not
  acceptance timings. One predeclared resource refresh used the same 45 jobs/
  binaries/inputs with observer CPU 0 and no concurrent analysis/build/debugger
  work. All outputs qualify, and 301 telemetry samples cover the entire run.
  New raw archive/report is linked from the Linux controls report; v1 is retained.
  No policy, production, public loader API or acceptance-gate change. Linux stays
  72/78; MLX remains blocked. No background collection is left running. `.pi/`
  remains intentionally untracked. Next: quiet/read-only-instrumented Linux
  investigation, then remaining completion-audit requirements; do not mark complete.
- Continuation: offline diagnostic preparation resumed after the handoff; no new
  acceptance, resource, GPU or Apple measurements. Rechecked HEAD/origin at
  `c4bcb69`, only `.pi/` untracked before edits; all 64 directly bound inputs in
  the retained final Linux audit match their hashes, still **72/78**.
  [Fixed diagnostic v2 protocol](docs/results/2026-11-11-phase2-optimization/linux-diagnostic-protocol-v2.md)
  proposes one 54-process run with read-only telemetry, **not yet run** and not an
  acceptance replacement. Quiet-window confirmation remains pending. Runner adds
  opt-in `--observer-cpu`: reject workload SMT siblings, pin launcher/GPU observers,
  retain explicit workload CPU, restore launcher affinity. Prior unpinned observers
  are a possible confound, not an established cause. Six tests include real Linux
  child inheritance/override; all **48 report tests** and default
  `CARGO_BUILD_JOBS=4 ./tools/check.sh` pass. Exact local log/hash in the protocol.
  No production/API/pin/tolerance changes. Next: confirm quiet window, bind/check
  all preflight identities, execute the fixed diagnostic once, then decide a
  defensible next intervention; Linux gate still blocks MLX and completion.
- **Diagnostic v2 completed once after explicit quiet-window confirmation**:
  [results](docs/results/2026-11-11-phase2-optimization/linux-diagnostic-v2-results.md).
  Launch HEAD/origin `fa040ed`; exact binary/source/Python/payload identities rechecked
  and manifest saved before launch. All 54 processes qualify; 901 samples/900 seconds
  bracket every command (17:01:56–17:06:47 UTC on 2026-11-11). All scopes/processes,
  telemetry and frozen pre-run protocol retained. Five CPU/load-tail issues did not
  recur, but native B1 block clusters persist: median ratio 1.10076, interval upper
  1.12231. CPU3 median/max activity 0/0.9901%; one-second GPU clocks cannot identify
  block clocks or establish cause. Outcome **unresolved recurrence**; Linux remains
  **72/78**, no control transplant, repeat, policy/production change or Apple work.
  Analysis/identities/coverage and clean-checkout summary reproduction verified;
  all 48 reporting tests and `CARGO_BUILD_JOBS=4 ./tools/check.sh` pass.
  No new full CUDA/Metal check is claimed. Evidence
  archive excludes binaries/weights/large traces; `.pi/` remains untouched.
  Next: separately predeclare a discriminating investigation. A bounded fixed-GPU-
  clock experiment is a hypothesis requiring explicit temporary-policy approval,
  capability/original-state checks, restoration and a new quiet window; not yet
  approved/performed, not proven necessary/sufficient. Read-only finer-grained
  observation is an alternative. No background collection remains running.
- Offline follow-up proposal (not approved/run):
  [clock diagnostic v3](docs/results/2026-11-11-phase2-optimization/linux-clock-diagnostic-v3-proposal.md)
  specifies one 18-process codec-only investigation, optional 1710 MHz request,
  reset-to-default policy caveat, 600-second cleanup deadline and 900-second CPU
  capture. Read-only capability query lists 1710 MHz; lock support/permission and
  restoration are NOT proven. Supervisor implementation/failure-path tests remain
  required before launch; no executable lock wrapper exists yet. User must confirm
  default policy, approve the temporary change and provide a new quiet window.
  Docs-only links/diff checks and 48 reporting tests pass; no runtime/policy change.
  The completed v2 is not rerun; Linux remains 72/78 and Phase 2 incomplete.
- Offline resident-block source inspection at `41ab9e5` is recorded in the
  [completion audit](docs/results/2026-11-11-phase2-optimization/completion-audit.md#offline-resident-block-timing-inspection).
  Benchmark source/CUDA ops and inspected block/evaluate sections match accepted
  `3307dff`. The block synchronizes before end-event enqueue: event spans include
  submission/idle gaps, not just kernel duration. Matching wall/event clusters
  therefore do not establish a device-clock cause. Verified with direct source
  inspection and byte/section comparisons; docs diff/link checks and all 48
  reporting tests pass. No new benchmark, hardware test, policy change or waiver.
  Next read-only protocol must discriminate kernel duration from submission gaps
  and label instrumentation overhead; investigation choice/new quiet window remain
  pending. Linux 72/78.
- [Offline v2 pool/trace inspection](docs/results/2026-11-11-phase2-optimization/linux-diagnostic-v2-results.md#offline-pool-counter-and-retained-trace-follow-up)
  checks all twelve archived native reports: B1 pool counters are constant within
  each runtime across fast/slow repetitions (64 MiB reserved, 128 MiB threshold).
  This does not exclude allocation latency or identify a cause. Older CUPTI
  captures are reconstruction/B4 scopes, not v2 B1 trial traces; no retrospective
  attribution. Published standard-library reproduction verifies archive/member
  hashes and all rows; ran successfully. Docs diff/new-link checks pass. No new
  measurements, settings changes or acceptance waiver; Linux remains 72/78.
  Next: predeclare a bounded trace investigation if the read-only path is chosen;
  actual execution still needs a fresh quiet window.
- [Read-only trace proposal](docs/results/2026-11-11-phase2-optimization/linux-readonly-trace-v1-proposal.md)
  now specifies 12 fixed native plain/traced processes, frozen v2 binaries,
  time/output caps and strict B1 window attribution. Source-derived queries
  110–145 cover B1 block first/warm/retained calls before stage profiling; no
  full-harness trace confirms that mapping yet. Offline checks on three old B4
  captures verify 36 two-event/query windows each, not current B1 causality.
  Docs links/diff and schedule arithmetic/source checks pass. No new run/policy
  change; parser/runner and required tests are not implemented. Next: user choice,
  implement/test the selected diagnostic, then obtain a fresh quiet window and
  bind launch prerequisites. Linux remains 72/78; no acceptance or Apple progress.
- Offline prerequisites under `tools/bench/`: `cupti_trace_boundaries.py`,
  `cupti_kernel_windows.py`, `bind_cupti_trials.py`, `qualify_diagnostic_codec.py`,
  `bounded_child.py`; 64 tests covering host boundaries, kernel/copy/memset correlation,
  overlap arithmetic, source/hash/PID declarations, ordinal joins and report-output
  qualification through the unchanged codec validator/frozen v2 Python references. Synthetic full-layout coverage
  is not a real current B1 trace. Each old B4 capture maps 972 kernels, 396 copies
  and 72 memsets to 36 windows; retained-B4 first/30-trial timing joins match
  exactly. Warmup timings were not serialized and stay null. Byte declarations do not prove same-child
  launch provenance or output quality. Bookkeeping/overhead stays unattributed;
  actual launch/report provenance, completeness and integrated collection remain
  pending. The bounded child primitive passes fake CPU success/failure/limit/
  interrupt/cleanup tests, keeps the leader waitable until group cleanup, and
  records failed/unconfirmed outcomes. Two red-to-green tests exposed setup-time
  cancellation/deadline gaps, now checked before attempting a spawn. Fake child
  environments are minimal; the original environment-bearing failure log is
  restricted/unpublished. It is not a full launcher; aggregate quotas,
  frozen manifest/order, telemetry/affinity and supervision across collector loss
  remain unresolved. No benchmark or policy change. The output qualifier passes archived regression fixtures,
  not a new capture, oracle run or acceptance attempt. User choice/quiet window still block launch; no new measurements/settings. `CARGO_BUILD_JOBS=4 ./tools/check.sh`
  passes with 112 reporting tests. Supported memory activities are matched in both
  directions; unknown/ambiguous/cross-window records fail, overlap is not counted
  twice, and equal-timestamp host records sort deterministically. Initial
  boundary-suite check failed inherited-RSS isolation; archive work now runs in
  a child, preserving the existing resource assertion. Kernel prototype initially
  missed RUNTIME launches; retained fixtures exposed both variants, now tested.
  Checks/log hashes and limitations are in the read-only proposal; docs diff/link
  checks pass. Linux remains 72/78; Apple and completion remain pending.
- Within EVERY phase: finish Linux/CUDA work and validation, then macOS/MLX.
  Both gates before advancing. Phase 3 generator/CLI work is out of this goal.
- Phase 1 is complete on both platforms ([acceptance](docs/PHASE1.md)). Phase 2
  implements shared FP32 Sudoku codec semantics and TinyStories/GPT-2/Qwen/BERT
  text tokenizers, **not full text-model inference**. No generator, inference CLI,
  official task-quality result, generalized importer or inference release exists.
- Production execution stays Python-free. Offline conversion/reference/measurement
  and native CUDA/MLX libraries are expected. Never silently change pins, schemas,
  established goldens or numerical tolerances; unsupported options must fail.
- Local checks replace GitHub build CI: `tools/check.sh`, opt-in `--cuda`/`--mlx`.
  Never restore build CI without approval. Metadata-only upstream watcher remains.
- `ssh macstudio` is authorized. Recheck availability/competing work before later
  runs, inspect worktree, then pull reviewed pushed commits into
  `~/projects/lexlapax/ncm-rs`. Never overwrite unrelated work or global tooling.
  Announce the first Phase 2 MLX run only after Linux acceptance.
- Update this map before EVERY commit, including amend/fixup. No agent attribution
  or co-author trailers; preserve upstream licenses and technical provenance.
- Persistent goal state is untracked `.pi/`; leave it untouched and unstaged.

## Historical Apple measurement acceptance

- `docs/PLAN.md` and `docs/PHASE2.md` now distinguish the accepted original
  baseline from the reopened performance-remediation extension.
- Independent Python MLX baseline passes all 642 unchanged trained traces and
  2,500 decisions against the CUDA oracle. Three-process codec/tokenizer evidence,
  raw telemetry and baseline qualification are published in
  `docs/results/2026-11-10-phase2-mlx/`. The extension fixes the summary's hidden
  dependency on ignored baseline files using checksum-pinned committed evidence.
  A clean-checkout test reproduces all historical metrics exactly; only current
  summarizer source hashes differ. Historical reports are not rewritten.
- Host-only artifact loads reproduce high native RSS; diagnostics remain at
  `reports/phase2-mlx-load-diagnostic`. No loader fix or efficiency claim.
- Final summary-tool tests pass locally. On the M1 Ultra candidate worktree,
  `./tools/check.sh --mlx` passed on 2026-11-10 with the pinned MLX prefix,
  including real-Metal components/operators and 11 report-tool tests; the exact
  measured source hashes were verified equal to the candidate files before this
  check. MLX reference asset tests also pass in `.reference/mlx-venv`.
- Preserve `.pi/` and the Mac candidate worktree's unrelated/uncommitted state.

## Task routing

| Need | Load |
| --- | --- |
| Phase 2 contract / acceptance / remediation | [final dual-platform acceptance](docs/results/2026-11-10-phase2-acceptance/README.md), [docs/PHASE2.md](docs/PHASE2.md), [plan](docs/PLAN.md), [research and performance extension](docs/PHASE2-PERFORMANCE.md), [initial research/PoC](docs/results/2026-11-10-phase2-performance-research/README.md) |
| Approved additional Phase 2 optimization | [five-step plan, research and pending evidence checklist](docs/PHASE2-OPTIMIZATION-REVIEW.md) (implementation authorized; no new speedup result) |
| Shared runtime and model | [core](crates/ncm-core/src/lib.rs), [config](crates/ncm-core/src/config.rs), [model](crates/ncm-model/src/lib.rs), [blocks](crates/ncm-model/src/blocks.rs), [codec](crates/ncm-model/src/codec.rs) |
| CUDA adapter / known native fixes | [adapter](crates/ncm-backend-cuda/src/codec.rs), [vectorized resampling](crates/ncm-backend-cuda/src/resample.rs), [pool policy](crates/ncm-backend-cuda/src/runtime.rs), [operators](crates/ncm-backend-cuda/src/ops.rs), [remediation](docs/results/2026-11-10-phase2-cuda-remediation/README.md) |
| Phase 2 oracle and complete CUDA results | [oracle audit](docs/results/2026-11-10-phase2-oracle/README.md), [codec report](docs/results/2026-11-10-phase2-codec/README.md) |
| Tokenizer identities, safety and parity | [tokenizer report](docs/results/2026-11-10-phase2-tokenizers/README.md), [source lock](tools/reference/tokenizer-sources.lock.json), [canonical lock](tools/reference/tokenizers-v1.lock.json) |
| Fixtures / offline tools | [fixtures](tests/fixtures/README.md), [reference README](tools/reference/README.md) |
| Artifact trust/limits | [docs/ARTIFACTS.md](docs/ARTIFACTS.md), [ncm-io](crates/ncm-io/src/lib.rs), [runtime checks](docs/results/2026-11-10-phase2-runtime/README.md) |
| Architecture / upstream caveats | [architecture](docs/ARCHITECTURE.md), [research](docs/RESEARCH.md) |
| Checks / reporting / compatibility | [testing](docs/TESTING.md), [reporting](docs/REPORTING.md), [upstream policy](docs/UPSTREAM.md) |
| MLX codec / current parity evidence | [adapter](crates/ncm-backend-mlx/src/codec.rs), [Apple parity report](docs/results/2026-11-10-phase2-mlx-codec/README.md), [shared trained capture](tests/support/trained_codec.rs) |
| Phase 2 Linux acceptance / benchmarks | [Linux gate](docs/results/2026-11-10-phase2-linux/README.md), [summary tool](tools/bench/phase2_summary.py), [runner](tools/bench/run_phase2_linux.py) |
| Historical baselines | [results index](docs/results/README.md), [Apple Phase 1](docs/results/2026-11-10-mlx-phase1/README.md) |

## Pins, machines and caches

- Upstream NCM: `c70274273f489abc73832b2b98584a8483b6b203`.
- HF checkpoints: `hazyresearch/ncm-checkpoints` at
  `d342ce8aaea58830eef3b8ad473664673e41f692`. OWT weights remain absent.
- Candle `=0.11.1`; default workspace needs no GPU SDK. Explicit CPU diagnostic
  and CUDA paths never silently fall back. Rust floor **1.99.1** is tested;
  previous 1.88 failed Candle AVX-512 intrinsics, not all intermediate versions.
- Linux: RTX 3060 Ti 8 GiB, driver 615.78.08, CUDA 13.4. Command:
  `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.
  Default GCC 16 is not the tested CUDA host compiler.
- Linux Torch reference: `.reference/venv`, Python 3.14.9, Torch 2.15.0+cu132;
  `tools/reference/requirements-linux.lock`. Clean source: `.reference/upstream-ncm`.
- Sudoku source: `artifacts/upstream/sudoku/easy`; canonical inspected bundle on
  both machines: `artifacts/canonical/sudoku-easy-qualified`. Its name alone does
  not confer inference qualification. Original file/tensor hashes are committed
  under `docs/results/2026-11-10-sudoku-artifacts/raw/`.
- Tokenizer source: `artifacts/upstream/tokenizers-v2`; canonical:
  `artifacts/canonical/tokenizers-qualified`. Separate `.reference/tokenizers-venv`
  uses `requirements-tokenizers.lock` (Transformers 4.57.6, Tokenizers 0.22.2,
  regex 2025.9.18 / Unicode 16). Do not replace the Torch environment with it.
- Rust HF Tokenizers `=0.22.2`, only Oniguruma feature; no HTTP/remote code. Candle's
  pre-existing Tokenizers 0.23.2 remains separate. Tiny byte BPE uses locked
  fancy-regex 0.17.0 and Unicode-16 regex-syntax tables. Asset licenses are retained.
- Mac: M1 Ultra 64 GiB, macOS 26.7.3, Xcode 27.2, Rust 1.99.1. Noninteractive SSH:
  `export PATH=/opt/homebrew/bin:$PATH`. Optional Xcode Metal compiler is missing;
  do not change global Xcode. Tested project-local prebuilt native MLX route works.
- MLX bindings: mlx-rs 0.34.0 / mlx-sys 0.7.0 Git
  `adf439bf5361d8a02955de7aaf651b0c45900945`; native MLX 0.32.2, mlx-c
  `a341b4925024b88b2c593468f16e12f5e17315da`. `.reference/mlx-venv` is offline-only;
  native include/share/lib are copied into `.reference/mlx-native`.
- Mac candidate testing used a separate worktree `.reference/phase2-mlx-candidate`
  with the main checkout's native prefix and target cache. Main checkout is updated
  only by pulling pushed commits. Preserve unrelated work; existing CPU work was
  left running, builds limited to two jobs. Benchmark acceptance must assess noise.
- Mac build: `MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"`,
  `RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"`,
  `RUSTDOCFLAGS="$RUSTFLAGS"`, `CARGO_BUILD_JOBS=8`, then `./tools/check.sh --mlx`.
  Tested wheel needs macOS 26.2+; absolute rpath is a spike setup, not packaging.

## Evidence and boundaries

- Phase 1: all 18 fixed operator cases pass CPU/CUDA/MLX at unchanged
  `atol=1e-5, rtol=1e-4`; all 238 checkpoint tensors read back byte-exact.
  Metadata-only manifest integrity is NOT authenticity or semantic validation.
  Root/leaf symlinks are rejected, but trusted parents/no concurrent hostile
  mutation remain required. Input byte limits are NOT peak-RSS bounds.
- Historical Linux checked-load median Rust/Python ~202/241 ms. Apple ~939/282 ms
  with repeated-load host peak RSS ~4.40/0.41 GiB: Rust slower/higher RSS. Native MLX
  active arrays return to zero, cache retains ~155 MiB; counters do not explain
  total host RSS. Investigate in Phase 2 before any efficiency claim.
- Phase 1 composed resampling was slower than PyTorch. Compiled Torch FP16 loses
  rounding on one fixture: compiled mixed precision remains unsupported, no waiver.
- Phase 2 reference budgets were declared before implementation in `docs/PHASE2.md`.
  Canonical trained oracle is CUDA; tiny oracle is CPU. Original-Python trained
  CPU/CUDA diagnostic retains one failing blank-board level-zero projection value
  (22,784-term dot, channel 113, ~0.001054 difference). Never use this to exempt
  candidate GPU failures or silently relax tolerances.
- Config/loading: strict initial codec family, exact 58 F32 runtime tensors,
  checked IDs/options, full host preflight before device upload; training buffers
  are separate. Both tiny and trained inventories match. No unsafe Send/Sync.
- Codec `n_layers=4` means head count, not decoder depth. Decoder is bidirectional,
  Q/K RMS-normalized and exact-erf GELU. GroupNorm spans positions/grouped channels.
  Preserve exact area bins, half-pixel interpolation and independent snapshots.
- Candle fixes: CPU batched transposed convolution misindexes a multi-position
  second batch, so only CPU dispatches separate batch slices. CUDA remains native
  batched. Native CUDA argmin tie order was wrong; selection now reduces values,
  then masked indices on-device. Tests cover wide/signed-zero ties and both fixes.
- Complete tiny codec: all 236 traces pass CPU/CUDA; snapshot/uninstrumented checks
  yield 252 floating comparisons / 112,212 elements; all IDs exact.
- Trained CUDA codec: all 642 traces and 2,500 code/output decisions pass. Two
  mixed-board near ties (levels 1/2) remain visible. Worst normalized error uses
  **94.2%** of the budget at blank-board level-zero channel 113. Do not infer Apple
  parity from Linux's narrow margin. Two fresh native traces are byte-identical.
- Reconstruction matches Python on 445/445 tokens and 5/5 sequences; recovery of
  original inputs is 437/445 (eight mixed-board differences), not losslessness or
  Sudoku-solving quality. Uninstrumented outputs match traced outputs.
- Historical native trace: `reports/phase2-codec/cuda-v2`, SHA-256
  `a3776cac0ba75eaeb9ebc9a30ebd03c131f0b4470f12f136b7ccc95fc25e0fa0`.
  Python reference traces: `reports/phase2-oracle/v1/`. Durable Phase 2 archive
  publication remains pending. Phase 1 archive exists at research prerelease
  `phase1-fixtures-v1`, not an inference release.
- Tokenizers: all four pinned profiles pass 356 encode and 2,314 decode comparisons,
  including bytes/Unicode/specials/postprocessing; Sudoku goldens unchanged.
  Tiny pickle conversion interprets bounded data opcodes, never unpickles or calls
  constructors. Canonical HF JSON comes from offline trusted-code-disabled wrappers.
  Four real-asset mutations/profile are rejected; fresh conversions are identical.
- Historical Linux measurements: `reports/phase2-linux-v1`; all 30 fresh-process raw reports,
  telemetry and first logits are committed under `docs/results/2026-11-10-phase2-linux`.
  Three processes, five warmups, thirty trials; matching output hashes/counts and
  44,856 native benchmark logits pass. B1 reconstruct Rust/Python 18.890/4.668 ms;
  warm codec load 184.49/267.61 ms; host HWM ~559–562/1530–1540 MiB. Native pool
  counters exclude context/cuBLAS allocations; not interchangeable with Torch's
  allocator or device-global snapshots. Occupied workstation/clock variation disclosed.
- Historical instrumented level spans identify intermediate composed resampling as the main
  native cost. Original-Python fullgraph Inductor preflight fails on NumPy-based
  Phi module indexing; no compiled timing baseline is qualified. Tokenizer timings
  retain both skip modes and special/non-special throughput; Qwen native loading
  is slightly slower in pooled medians. Original-source vs canonical load formats
  differ. No model/operator/tokenizer math, dependency pin or golden changed.
- MLX now implements the existing backend contract with opaque immutable arrays
  and an explicit GPU stream (restores a CPU-scoped caller); no model duplication.
  All 18 operators, 140 components and complete tiny codec pass; 642 trained traces
  and all 2,500 decisions pass. Two fresh captures are identical, SHA-256
  `f20ffe0f36a82c80a81e91a61b61950968821369b6b41d0f13cbf9e732875a0f`.
  Same two near ties; reconstruction 445/445 versus Python, input recovery 437/445.
- Initial MLX full-span convolution failed two blank-board projection values despite
  exact IDs. Retained in the Apple report. Equivalent native FP32 matmul for
  zero-padding/full-span convolution fixes parity; no precision/oracle/budget change.
  Worst component still uses 94.28% of budget. Large v1/v2 traces remain ignored
  under `reports/phase2-codec/` on both machines; Mac also has `mlx-v2-repeat`.
- The shared trained-capture harness refactor preserves the CUDA trace hash exactly.
  Mac tokenizer probe passes all four profiles, including rejection cases and a
  cleared environment without Python. Canonical tokenizer assets are now on Mac.
- Historical Linux `tools/check.sh --cuda` passes: 30 base Rust tests, real-CUDA codec
  suite, all 18 unchanged operators, fmt/Clippy/docs/syntax. Twelve report-tool tests
  and all 18 offline reference tests pass. Tokenizer probe previously also passed
  with cleared env and no Python on PATH. Logs: `.reference/phase2-linux-check.log`,
  `.reference/phase2-linux-reference-tests.log`; reports retain exact fingerprints.
  Shared-harness/final Linux checks: `.reference/phase2-mlx-shared-linux.log`,
  `.reference/phase2-mlx-final-linux.log`; all 18 offline tests also rerun.
  Mac `tools/check.sh --mlx` passes with four new codec tests; logs copied locally:
  `.reference/phase2-mlx-v2-check.log`, `.reference/phase2-mlx-v2-components.log`.

## Provenance and next actions

- Phase 1 closure `650cc7d`; Phase 2 oracle `97a3125`, checked runtime `8492c27`,
  components `a4a6344`, complete codec `bb77fef`, tokenizers `0396b56`, Linux runtime
  gate `fd1b482` (pushed). Preserve old reports/source hashes;
  do not relabel historical runs. Benchmark outputs are synthetic fixtures, not
  task-quality data or an inference release.
- **CUDA remediation milestone:** vectorized FP32 gathers/area reduction, pinned
  optional `candle-nn = 0.11.1` fused CUDA norms/softmax, immutable views and explicit
  default-pool retention. No shared model/loader/tokenizer math or goldens changed.
  CUDA default release threshold is now 128 MiB; explicit per-process shared-pool
  API allows zero/custom policy. Not preallocation or a hard cache limit. Tests
  cover real-CUDA policy/opt-out, broad resampling, stable ties and snapshots.
- User authorized stopping `llama-server.service` (inactive, not disabled), closed
  Chrome, then authorized continuing with Hyprland running. No other app was stopped
  or global tooling/power changed. Local CUPTI sample traces identify launch count
  16,756 → 1,312 for load + two B1 reconstructions. B4 bimodality was allocation:
  fast/slow cuMemAllocAsync totals 0.636/20.497 ms with identical kernel work;
  retention gives 0.642 ms. Extra reusable reservation is disclosed (96 vs 64 MiB).
- Predeclared bands are in the research `acceptance-v1.json`, written before
  production changes. Final interleaved old/new/Python reports are archived with
  source/binary hashes, environment and telemetry in the remediation report.
  B1/B2/B4 reconstruction old/new/Python medians: 18.165/3.285/4.659,
  26.681/4.003/5.081, 28.949/5.134/5.859 ms. All codec median/p95 Python bands
  pass descriptively. Warm load 195.88/194.45/286.25 ms; native host HWM ~580 MiB
  remains favorable. Failed vectorized B4 and earlier comparison rounds retained.
- Correctness: final `reports/phase2-codec/cuda-remediated-v1` passes all 642
  trained traces/2,500 decisions with unchanged budgets; same two near ties and
  437/445 input recovery. Native trace SHA-256:
  `d180a52c011c8af15812d10941de8574b1671c2cd23de67e9abc828f7c5ecba4`.
  360 synthetic CUDA rows / 753,984 values pass against pinned Torch CUDA (maximum
  budget fraction 0.628). FP64-coordinate alternative still fails 228 values;
  this GPU check is distinct from the earlier Torch-CPU comparison.
- Validation: final `tools/check.sh --cuda` passed (log
  `.reference/phase2-cuda-remediation-final-check.log`, also published). Final
  `./tools/check.sh` passes, including 20 report tests; 21 offline reference tests
  pass. A clean staged checkout with no ignored inputs exactly reproduces all six
  matrix summaries (historical summarizer provenance excepted), six CUPTI profiles
  and the control summary; evidence/source fingerprints and doc links verified.
  Logs: `.reference/phase2-remediation-publication-check.log` and
  `.reference/phase2-remediation-reference-tests.log`. Staged code/archives reviewed;
  `.pi/`, executables, weights and large model-trace archives are not staged.
- **Extended Linux acceptance passes; next is Apple remediation.** Fixed 30-triplet
  TinyStories and nine-triplet GPT-2 affinity controls close execution uncertainty.
  Qwen's first nine-triplet control still failed uncertainty; a fixed 30-triplet
  identity/path control (only allowed for byte-identical native binaries) passes.
  All negative runs are retained; no tokenizer optimization or proven path cause
  is claimed. Whole-matrix audit passes 78 comparisons with unchanged bands and
  qualified traces. Fresh-process startup/load/execute/drop/exit evidence is now
  separate from steady-state timing. See the Linux acceptance report for metrics.
- Latest `./tools/check.sh --cuda` passes, including 22 report tests; log
  `.reference/phase2-linux-acceptance-check.log`, also published. Audit:
  `reports/phase2-linux-acceptance-v2.json`; controls are archived in the new report.
  A clean staged checkout exactly regenerates acceptance from committed archives
  and passes all 22 report tests; staged scope/diff checked before publication.
  Shared lifecycle harness compiled against both old/current CUDA and exact solved
  outputs agree across all nine processes. No production change in this follow-up.
- Mac availability/state rechecked after `7aee602`: main was clean and safely
  fast-forwarded; old binaries saved in `.reference/phase2-mlx-baseline-bin`, clean
  old source in `.reference/phase2-mlx-baseline` (`5dc9626`). Historical candidate
  worktree inspected and untouched. First remediation diagnostics announced/run.
  Fresh old load ~944 ms and host HWM 4.81 GB reproduce the deficits. Allocation
  stacks identify `bounded_read → RawVec::finish_grow → realloc`; VM maps show
  freed large malloc regions. CPU sample: 2,187/2,222 main-thread samples in SHA-256.
- Shared candidate: reserve checked read length + EOF byte; deserialize authenticated
  BPE/WordPiece tokenizer models directly into the same runtime wrapper; enable
  pinned sha2 0.10.9 `asm` only on macOS/AArch64 (explicit sha2-asm 0.6.4 addition).
  No checksums/preflight/path checks, payloads, precision or algorithms removed.
  Details/evidence: `docs/results/2026-11-10-phase2-shared-loading/README.md`.
- Linux revalidation precedes Apple candidate tests: full `tools/check.sh --cuda`
  passes (`.reference/phase2-shared-load-final-check.log`), all 642 traces/2,500
  decisions and real tokenizer goldens/rejections pass; 21 offline reference tests
  pass. Fresh three-way matrix and 30-triplet controls yield 78 passing comparisons
  (`reports/phase2-linux-shared-load-acceptance-v2.json`). Qwen load 194.80/107.75/
  172.34 ms old/new/Python. Linux codec/tokenizer benchmark binaries remain byte-
  identical after the Apple-only feature addition; Linux SHA features unchanged.
- Shared change pushed as `74c2ed3`, pulled on Mac; full `tools/check.sh --mlx`
  passes. First codec diagnostic: 180.06 ms / 293.63 MB HWM versus old 944.17 ms /
  4.81 GB and Python 270.58 ms / 309.10 MB. Qwen is faster (93.79 ms) but still
  634.52 MB HWM; allocation stack identifies its separate read-buffer reallocations.
- Follow-up reserves checked tokenizer file length + EOF byte, with boundary tests.
  Returned through Linux first: full CUDA checks, real tokenizer goldens/rejections,
  fresh whole matrix and 30-triplet controls pass all 78 comparisons. Negative
  audits retained. Explicit optional `load_controls` applies unchanged paired-process
  uncertainty to checked loads too; malformed-trial regression passes. Evidence
  and exact configs are in the shared-loading report's `raw/` directory.
- SSH restored by user; Mac availability/worktrees rechecked, clean main pulled
  through `2c195ed`. Full Metal checks pass. Tokenizer-buffer candidate improves
  Qwen to 86.97 ms / 455.25 MB HWM, but fresh Python is 145.97 ms / 288.59 MB:
  memory still fails. No unrelated desktop work, tooling or credentials changed.
- User requested reconsidering Rust structures for memory AND performance. New
  candidate packs TinyStories bytes/u32 offsets, stores merge output IDs without
  redundant ranks, borrows reverse lookups and moves valid UTF-8 output storage.
  HF loading avoids full-vocabulary clones for size validation and untagged merge
  trees, keeping the pinned runtime BPE and complete tokenizer visitor unchanged.
  First packed decoder regression retained; ownership fix passes controlled timing.
- Linux first: full CUDA checks, all tokenizer goldens/rejections, complete matrix,
  30-triplet TinyStories/BERT controls and nine full-codec load-control triplets
  pass all 78 comparisons. Qwen load 223.22/89.90/196.78 ms old/new/Python; HWM
  194.49/81.56/240.10 MB. New checked-load control validates every codec report;
  Apple controls explicitly have no affinity claim and require activity telemetry.
  New MLX lifecycle/three-way tools await real Apple execution. Full details and
  all failed/final evidence: `docs/results/2026-11-10-phase2-compact-tokenizers/`.
- Compact candidate pushed/pulled as `45743d0`; full Metal checks and real
  tokenizer goldens pass. Qwen improves to 65.82 ms / 326.83 MB HWM but still
  exceeds Python's memory band. Retained as failed memory diagnostic. Follow-up
  reserves canonical BPE vocab/merge containers from pinned profile capacities;
  tests enforce bounds. Full Linux CUDA checks and fresh matrix + 30-triplet
  controls again pass all 78 comparisons; unchanged-codec load control is explicitly
  reused with source binding. Evidence appended to the compact-tokenizers report.
- Capacity follow-up pushed/pulled as `083c1e5`. Full Metal checks and real
  tokenizer goldens pass; 642 trained traces/2,500 decisions qualify unchanged,
  zero ID disagreement. Fresh Apple matrix: codec load 945.87/179.99/272.21 ms
  old/new/Python, HWM 4749.16/293.91/308.69 MB; Qwen 189.29/63.65/144.78 ms,
  HWM 633.88/69.81/290.98 MB. Repeated-load probes plateau with zero active MLX
  bytes after drops. All latency point estimates and host-memory ratios pass;
  75/78 cells meet conservative process envelopes. GPT-2 encode and Qwen
  encode/decode versus old Rust still need fixed additional process controls.
- Separate `.reference/phase2-mlx-lifecycle-baseline` at `5dc9626` has ONLY copied
  lifecycle example/support files; baseline build and remote lifecycle command
  succeeded. **SSH agent became unavailable again before report transfer.** Retry
  denied; user asked to unlock. Mac lifecycle report/final check logs remain remote;
  do not infer lifecycle figures or final acceptance. Next use a process-local
  persistent SSH connection after access returns, without global configuration edits.
- Published qualified Apple measurements and explicit pending diagnostic:
  `docs/results/2026-11-10-phase2-mlx-remediation/README.md`. Audit now binds Python
  control sources as well as native ones; mutation regression and local checks pass,
  latest Linux gate still passes 78 comparisons under the stronger audit. A clean
  staged checkout without ignored artifacts exactly reproduces Linux acceptance
  and both Apple summaries; all 24 report-tool tests pass.
- Final follow-up: SSH restored; availability rechecked and a process-local
  persistent connection used (no global SSH edits). Mac main pulled `42caaec`.
  Retrieved/validated lifecycle and final logs. Fixed 30-triplet GPT-2/Qwen controls
  resolve all three uncertain cells: all 78 Apple comparisons pass. Linux also
  passes 78 under the final audit. All negative rounds remain archived.
- Final `tools/check.sh --cuda`, then `--mlx`, pass; fresh native trained traces
  captured CUDA first then Metal both pass 642/2,500 unchanged. Cleared-environment
  tokenizer probes pass on both. Evidence/configs/logs are in the final report.
  Apple lifecycle medians: old/new/Python full process 1050.72/287.30/468.01 ms;
  startup 39.84/39.68/106.10 ms. Not disk-cold or steady-state timing. Clean staged
  checkout exactly reproduces both 78-cell audits and all four matrix summaries;
  all 24 report tests and evidence hashes pass. Control binaries match matrices.
  All 21 offline reference-tool tests and the final default publication check pass.
  Plan, tracker, extension contract, root README, testing/architecture status and
  results index now consistently record completion without a generator/release claim.
- **Historical handoff:** the earlier Phase 2 scope was closed at this point;
  the subsequent approved optimization work and its current open Apple gate are
  recorded at the top of this map. Preserve historical worktrees, goldens and
  `.pi/`. No service, unrelated process, clock/power policy or global tooling was
  changed for those measurements.
