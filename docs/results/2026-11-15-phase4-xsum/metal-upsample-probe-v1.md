# XSum Metal first-upsample diagnostic

The [same-batch cascade comparison](level1-cross-backend-diagnostic-plan-v1.md)
found identical CUDA/Metal codebook values but differences at the first
upsampling convolution. Its stride and kernel are both three, so this isolated
probe compares the existing native Metal transposed convolution with two
algebraically equivalent FP32 matrix-product formulations. It uses the same
32-row forced codebook input and original weights, comparing row 4937 to the
captured CUDA and Metal outputs. Different reduction orders are expected;
neither changed precision nor a relaxed tolerance is implied.

The input was prepared from the pinned artifact and the retained 32-row code
history; the selected codebook row is bit-identical to both captured backends.
Its SHA-256 is
`4654060f823fb5a12d32c668eaf5621ff68f589d970a16928fb4bddce9fa33db`.
A local CPU calculation showed FP32/FP64 matrix-product variants match fewer
CUDA values than the existing Metal convolution on this row, so those CPU
orders are not proposed for runtime use. The standalone Metal probe passed
Python syntax checking; its real-device result is pending. This is a bounded,
previously inspected diagnostic, not a quality or performance gate.

The row-4937 Metal probe at `685b829` reproduced all 4,608 previously
captured Metal values. Native convolution matched 3,299/4,608 CUDA values;
the flattened FP32 matmul matched 4,608/4,608, while three separate matmuls
matched 730/4,608. The result SHA-256 is
`070bfecce2a5e97f227498c3b9e006e50c351f894dbf9d4f71ec1c88fc41ac05`;
the local copy matches the Mac. This is one previously inspected row, not a
model-level fix or a quality gate.

For a broader bounded comparison, pinned PyTorch CUDA evaluated the same
32-row `x`, weight and bias with deterministic cuDNN, TF32 disabled and stride
3. Its row 4937 is bit-identical to the captured CUDA row. The full reference
trace SHA-256 is
`a44f20b092f44d91c3f4271171f996b181c63aac7eb58eca0639956ef70a14ed`.
The probe can now compare every element of that full batch after validating the
reference hash and selected row. The full-batch Metal result remains pending;
this direct operation is diagnostic, not an independent model reference.

The full 32-row Metal diagnostic at `622ae97` matched the CUDA direct
convolution in all 147,456 elements with the flattened matmul. Native Metal
matched 106,826 elements, while separate matmuls matched 25,889. The result
SHA-256 is `e801a2931781d7c99b0b68f048fd1e0dabdd3553893ab496bb3d7d260fa9274b`;
the local copy matches the Mac. This remains a single inspected batch and a
single stage, not an end-to-end or independent held-out result.

An explicit experimental Metal backend is prepared to use the flattened
matmul only for the shape-checked level-zero XSum upsampling operation; the
default backend is unchanged. A deterministic Metal layout/bias regression and
forced-history comparison are planned after Linux CUDA checks. Full-quality,
performance and raw-ID gates remain pending, with no tolerance changes.
The full local CUDA check passed, including 18 operator cases; log SHA-256
`0b4c44c9512532ac7f826655eb2d236b3182c97e52c81fc9781ca57c5096e3f5`.
This does not compile or validate the macOS-only candidate; the Metal check,
layout regression and forced-history trace remain pending.

The experimental backend subsequently passed `tools/check.sh --mlx`, including
18 Metal operator cases and the deterministic layout/bias regression; the
copied check log SHA-256 is
`5fae06ec61534c949e1c067d278f8ebfff20012d2cae76f53bbdfa2a68ad1019`.
On the original forced 32-row history, row 4937's first upsample now matches
all 4,608 CUDA values, but its level-1 position-4 code crossing remains.
For row 4883, all forced-history raw IDs now match CUDA; the original Metal
trace differed at position 40. The candidate row-4937 and row-4883 trace
SHA-256 hashes are
`ffead2f64871334f3f10097d6ffcf689e915d34be7b7d7d5489e7e94fe8891ac` and
`ad08ba5f1b90ac9d1977d861f417414cdcda63a9f71db2d21fcba9e96e26816c`.
These are forced-history diagnostics, not free-generation or full-quality
results; the exact Metal gate remains failed and the default is unchanged.

The same already-built experimental Metal binary (SHA-256
`0c51547caf12970f8755b1542e8aa96ac2278c059549ce1af044257503d6cd04`)
was also run on the three retained 32-row forced histories with known head
crossings. Against the retained CUDA traces, row 9813 now has no head argmax
differences; row 10009 still differs at level 5 position 15 (CUDA 3758,
Metal 3032), and row 10015 at level 8 position 18 (CUDA 2298, Metal 605).
All selected final IDs match CUDA in these forced histories. Inputs passed
SHA-256 checks after the runs, and copied output hashes match the Mac. The
respective safetensors hashes are
`b44663fa75ab50cbdfd9dfd399d20fcc857d970a19f7338126cbb175380d7b74`,
`a7dbde3f03384b09b10d33477be570266c7fdc277167268ac7857a171dfdbb59`,
and `8944ca1c2bc67eb2f67b6943f30ce6396e88bf4211169ad69447014b2e1b35ed`.
The ignored captures are in `reports/phase4-xsum-upsample-remaining-v1/` on
Mac and `reports/phase4-xsum-upsample-remaining-captured-v1/` locally. These
are inspected rows with forced CUDA histories, not a full free-generation pass.

A separate opt-in diagnostic combines the experimental upsampling operation
with the previously tested experimental normalization. Neither operation is
selected by the default backend. Linux checks precede the same forced-history
Metal comparison; results are pending. The full `tools/check.sh --cuda`
passed 18 CUDA operator cases; log SHA-256 is
`08ad8c45929811e7056ff25f944b12953f82dec3611cab67e557016e6e3605a2`.

At pushed commit `5b4d3d1`, the combined diagnostic built on the Mac using
the documented prebuilt MLX configuration. Initial build attempts without
that environment failed; their logs are retained as
`reports/phase4-xsum-combined-build-v{1,2,3,4}.log` on the Mac. The successful
binary SHA-256 is `f62b2344d0e0654ab5bb48f2e7e3e1bc2a6e5ae1154c5b15751ec06b4ab6ed81`.
Input checks passed after both forced-history runs. Row 4883 matches all CUDA
head choices and selected IDs. Row 4937 still differs at level 1 position 4:
CUDA chooses 3182 and Metal 3690, although forced final IDs match. Capture
SHA-256 values for rows 4883 and 4937 respectively are
`8d086d9f24f5ced81e01bada0441a1518540df2cb4801685d33a7ac7b7d5e125` and
`d3598bf2d90d60920f6ee83858693744c01a74cbe6d88d377208e6050c8a1a07`;
local copies match. These targeted forced-history results do not establish
free-generation parity or a phase gate pass.

The combined backend was also tested on the three additional retained forced
histories using the same binary. It retains the head crossings for row 10009
(level 5 position 15, CUDA 3758 versus Metal 3032) and row 10015 (level 8
position 18, CUDA 2298 versus Metal 605). It also restores the row 9813
crossing that upsampling alone removed (level 7 position 17, CUDA 2854 versus
Metal 3206). All forced-history final IDs match. The respective captured tensor
hashes are `74ff1cec3d1ca59a299ca148c749d9cce3c71f2151968ed9964e3f1604b39190`,
`cd4b612c0fca153046facff59e7894c8b5be53908bb842afa1b6e49c9ad63049`, and
`f9fd4027eed3fc28e72bd5604a29ea3bb6223ac06473e4cbe3b58869e89d26a6`.
Inputs remained hash-identical, and local captures match Mac hashes. This is
evidence against selecting the combined operation based on these inspected
rows; it is not a full corpus or quality test.

Comparing the retained CUDA cascade trace to the upsampling-only row-4937
trace identifies the next captured difference: the first convolution agrees
in all 4,608 values, while the following group normalization differs in 1,575
of 4,608 values (maximum absolute difference `4.7684e-7`). The second
convolution differs in 13,098 of 18,432 values (maximum `9.5367e-7`). These
are comparisons of one selected row in the same forced 32-row history, not
proof that group normalization causes the downstream choice crossing. They
identify group normalization as the next bounded operation to investigate.

On Linux CUDA, PyTorch `group_norm` with eight groups and epsilon `1e-5` on
that exact retained first-convolution input reproduced all 4,608 retained CUDA
normalization outputs bit-for-bit. Explicit PyTorch reductions over the same
grouped input disagreed in 1,902 values with square-root/division and in 1,734
values with reciprocal-square-root/multiplication (maximum differences
`4.7684e-7` and `7.1526e-7`, respectively). This single-row diagnostic
indicates that reduction arithmetic matters; neither decomposition is a
qualified replacement and no Metal result or causal fix is implied.

A standalone Metal diagnostic `tools/reference/xsum_groupnorm_metal_probe.py`
is prepared to compare the current composed operation, MLX fast layer norm,
and a safe-math Welford reduction on those exact eight groups of 576 values.
The payload under `reports/phase4-xsum-groupnorm-probe-input-v1.safetensors`
has SHA-256 `787774bc0753e424963efe7d17460105f3c3f4c88d6809f2edf20fd275372d9d`;
preparation checked that CUDA and upsampling-only Metal first-convolution
values are bit-identical. Python syntax checking passed locally. A real Metal
run is pending; this probe is not production inference or gate evidence.

The probe ran on Metal at pushed commit `690e412`; the input hash matched
before and after execution. Of 4,608 CUDA values, the composed operation
matched 3,033 exactly, MLX fast layer norm matched 2,781, and the safe-math
Welford candidate matched 2,070. All had maximum absolute difference
`4.7684e-7`; mean absolute differences were `1.8466e-8`, `1.9416e-8`, and
`3.0744e-8`, respectively. The result SHA-256 is
`24133c5c206c59f6a82d56760898348c3c131dff0b32e45c2a763bfef5ab2ba5`;
local and Mac copies agree. Neither alternative improves this inspected row's
exact count; no production or quality conclusion is drawn.

A follow-up probe adds a separate 512-thread, four-value-chunk reduction
matching the layout of the existing CUDA reference kernel. Metal and CUDA
compiler math and reciprocal-square-root rounding can still differ. Python
syntax checking passes; Metal execution is pending. This does not change the
model backend or qualify an output gate.

At pushed commit `eea366d`, the reference-layout candidate matched 4,117 of
4,608 CUDA values exactly on the same row, versus 3,033 for the unchanged
composed operation. Its maximum absolute difference remained `4.7684e-7`
and its mean absolute difference was `6.6625e-9`. The earlier three probe
results repeated exactly. The result SHA-256 is
`db5fee7e02a2f48431878e359456c1f8064928a02f923f099bf014a86f995c32`,
and local and Mac copies match. This previously inspected row motivates a
forced-history model test; it does not establish exact output or quality parity.

An explicit diagnostic backend now combines the first-upsample candidate with
a group-normalization kernel using the reference layout only on the validated
`[batch, 12, 384]`, eight-group shape. The default backend remains unchanged.
Full Linux `tools/check.sh --cuda` passed, including 18 real-device operator
cases (`reports/phase4-xsum-groupnorm-candidate-cuda-check-v1.log`). Rust
Metal compilation and forced-history behavior remain pending. The Rust kernel
binding does not expose the Python probe's safe-math option, so equivalence to
that probe must be measured, not assumed.

The first Metal forced-history run failed because the shared tracer still
expected 36 tensors after three diagnostic tensors were added. The failed log
is retained in `reports/phase4-xsum-groupnorm-model-v1/` on Mac. After the
guard was corrected to 39, the full CUDA check passed (18 device cases) and
the corrected Mac capture completed at `ddb0aa1`. Row 4937's first convolution
is exact versus the retained CUDA trace; group normalization differs in 491 of
4,608 values, the same count and maximum `4.7684e-7` as the standalone probe.
At level 1 position 4, CUDA still selects code 3182 and Metal selects 3690;
the forced final IDs match. The capture SHA-256 is
`6d81a2e4afc828f5c9384f878db85d91e4497ceaf33e29042bd44612d0c1324e`,
and the copied file matches the Mac. Matching aggregate probe statistics does
not prove the outputs are bit-identical. The candidate does not resolve the
known crossing and does not qualify full generation or the Metal gate.

On that forced row, grouping the 4,608 normalization values by the original
eight channel groups shows seven groups bit-identical to CUDA: all 491
differences occur in group 5, with maximum `4.7684e-7`. After the second
convolution, 9,798 of 18,432 values differ (maximum `7.1526e-7`). The two
competing level-1 logits are exactly the same in the original Metal,
upsampling-only and group-normalization candidate traces: 9.411467552 for
code 3182 and 9.411472321 for code 3690. These inspected values do not prove
causality; they narrow a subsequent normalization-arithmetic probe to group 5.

At `1105884`, the standalone reference-layout reduction was rerun with
explicit safe, relaxed and fast Metal math modes. Safe reproduced 4,117/4,608
exact matches, with all 491 differences in group 5. Both relaxed and fast
matched 2,523/4,608, with differences in seven groups. The three hash-bound
receipts remain under `reports/phase4-xsum-groupnorm-math-*-v1.json` on both
machines; their respective SHA-256 values are
`4abebfca74b51fb9cc52c38dc19792f2c7789efc1156777158658282acc7f071`,
`f765ed53627d3c3daf1cc1bee45008ebd91c76a9283394b6c4485e34f2c37f96`, and
`9e8d5effd2fcc4665c1c3373878c837922bcb1fd01e354b484f553f9ef31955b`.
These are one-row diagnostic results, not production or quality acceptance.

The reference-order Metal probe at `bdddeb5` exposed per-group statistics;
pinned CUDA `native_group_norm` reproduced the captured CUDA output exactly.
All eight means and seven inverse standard deviations match; group 5's Metal
inverse is one FP32 step higher (1.620216846 versus CUDA 1.620216727).
Receipts hash to `299d32ab825a4296081302189966f86fda780766ee350dab7a30a374786e3d9c`
(Metal) and `06d9d303f5000210e7782f890b716ec65c2810c5f31c07ea327bc90b494405da`
(CUDA). At `3691803`, replacing reciprocal square root with `1 / sqrt` in the
standalone Metal probe corrected group 5 but introduced differences in groups
2, 4 and 6; 3,036/4,608 values matched exactly. Its result hash is
`a019fd79077082d6484586369a01744be82b290d45e2b2e25696700f7c960b2a`.
This one-row diagnostic is not evidence for changing the runtime operation.

At `7d82987`, an FMA-based reciprocal-square-root refinement fixed group 5
but made group 4's inverse one FP32 step lower than CUDA. The output matched
4,094/4,608 values exactly, with all 514 differences in group 4 and maximum
absolute difference `4.7684e-7`. The local and Mac receipt SHA-256 is
`6201b7800fa0c21e441bd8df0ef219d75d6da1bbaa20d29aeda4f0893e1f2994`.
This inspected-row result is not a general CUDA approximation or runtime fix.

At `b926ff6`, the safe reference-layout probe additionally captured each
Metal variance-plus-epsilon input (receipt SHA-256
`365ee930752eb667c52e4cf62cbe81e908bcb5c89b7a005c5f5cccd51774aa2f`,
with matching local and Mac copies). Computing `1 / sqrt` in CPU double
precision and rounding to FP32 agrees with CUDA's recorded inverse for seven
groups, including group 5, but gives the lower neighboring value for group 4.
CUDA's pre-inverse variance was not captured, so these comparisons are
conditional on Metal's rounded variance being identical to CUDA's; they do
not prove CUDA's inverse is incorrectly rounded or justify a runtime change.
