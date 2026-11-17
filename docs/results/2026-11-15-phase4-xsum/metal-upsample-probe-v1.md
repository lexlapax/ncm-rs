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
