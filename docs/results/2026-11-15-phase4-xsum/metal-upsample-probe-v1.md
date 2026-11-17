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
