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
