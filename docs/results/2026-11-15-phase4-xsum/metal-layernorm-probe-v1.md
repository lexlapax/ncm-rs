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

On the authorized Mac at `cb27b14`, the probe completed on the 512 identical
prefix rows. The existing fast layer norm matched 234,772 of 393,216 CUDA
values exactly, with maximum absolute difference `1.1921e-7` and mean absolute
`7.9592e-10`. The safe-math Welford probe matched 363,025 values exactly;
maximum difference was also `1.1921e-7`, mean absolute `3.8816e-10`. The
result SHA-256 is
`1178738b930dedffd748b59280b83b528e4ba97fa28cfcbb14a4d2d0a26aa916`;
the local copy matches the remote file. It is a single captured batch, uses
previously inspected data, and leaves many differences. It does not establish a
model-level improvement or resolve the full Metal gate.

A follow-up explicitly checks for 32-lane SIMD groups and compares Metal safe,
relaxed and fast compiler math modes on this same diagnostic payload. These
variants remain diagnostic, and neither a numerical tolerance nor the runtime
backend is changed. Their results are pending.
