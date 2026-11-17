# XSum Metal reference-order layer-normalization probe

The [first-block comparison](level1-cross-backend-diagnostic-plan-v1.md) found
that CUDA and Metal receive identical 512 prefix vectors and first differ at
layer normalization. This isolated experiment compares the existing MLX fast
normalization and a diagnostic Metal Welford reduction against the captured
CUDA output for those exact inputs and learned affine weights. The candidate
follows the CUDA probe's 128-thread, four-SIMD-group reduction order, but Metal
compiler, division, contraction and reciprocal-square-root rounding can still
differ. The probe must report actual agreement and must not be treated as a
production backend or as proof that normalization caused the near tie.

A local CPU preparation checked bitwise equality of all 512 captured prefix
inputs and packed them with the original CUDA first-normalization output and
the pinned artifact's first-block weight/bias. The resulting payload SHA-256
is `aa3a937dd0529938bb9c341b84b48123ce19ace95d21e28ffbaf082e027b75fa`;
it is ignored under `reports/phase4-xsum-norm-probe-input-v1.safetensors`.
The CUDA row capture and full CUDA regression underpinning it are documented
in the linked comparison. Python syntax checking passed. The authorized Mac
will receive the hash-checked payload under ignored reports and execute the
committed standalone probe in its existing MLX environment. No model behavior,
artifact, threshold or quality gate changes; Metal result remains pending.
