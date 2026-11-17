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
