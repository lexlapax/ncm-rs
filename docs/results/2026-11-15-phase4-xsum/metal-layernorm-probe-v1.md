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

At `f317cf3`, the explicit 32-lane check passed. Safe math reproduced the
363,025 exact matches above; both relaxed and fast modes matched 196,960 of
393,216, worse than the 234,772 for the existing fast norm. All three had
maximum absolute difference `1.1921e-7`; mean absolute differences were
`3.8816e-10` for safe and `1.3516e-9` for relaxed/fast. Respective result
SHA-256 values for safe, relaxed and fast are
`da89792aca08405015aad5c69b8c855a43b6cccf75217ea1c8e74c4479902a42`,
`793e940eed01d619969c14791e6430f77aa351535882cb58ce65ade1a66c7f2e`, and
`7643b823a5d53c2a41c8ffb393e92bf7e4b25f3afc25856e3a24836b47499c9c`.
The copied files match remote hashes. These comparisons still do not establish
model-level behavior or acceptance; the safe variant warrants a bounded
forced-history investigation before any production integration.

An explicitly selected experimental Metal backend and forced-history example
are being prepared to test the safe Welford operation in the complete model.
The normal backend remains unchanged. Before any Metal run, local CUDA checks
must pass; the first model-level test will compare row 4937 with retained CUDA
and default Metal traces. The experiment is not a qualified profile and will
not change the frozen acceptance rules. A 32-lane SIMD assumption is checked by
the kernel; unsupported widths produce nonfinite diagnostic results and fail
capture rather than silently falling back.

The Linux `setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda`
passed, including 18 real CUDA operator cases; log SHA-256
`c1b4378b1e7704ec4bf47ff08ce9b1b32fe7ca3c9767021f27d1b65b88cf4feb`.
This validates the shared Linux regression, not compilation or execution of
the macOS-only experimental backend, which remains pending.

The first Mac check at `e6973f0` stopped at compile time because the new audited
kernel launch lacked the explicit `unsafe_code` allowance required by the
project's lint policy. No candidate inference ran. The failed check is retained
on the Mac under `reports/phase4-xsum-norm-candidate-mlx-v1/check.log`; the
allowance is now scoped to the validated launch wrapper, as for existing Metal
kernels. A fresh Mac check and candidate capture remain pending.

## Forced-history candidate result

The corrected experimental backend at `36b5111` passed the full Metal check,
including 18 operator cases (log SHA-256
`92e2e906a3e305a3ef4778e888bda4960be2efded624c70323ff4fc957091c73`).
The same-batch row-4937 capture has SHA-256
`1323389ca938fd44d1fd23388265a7660be0e5fa45438d8955d55e4857ba8f5f`;
the copied file matches the remote hash and inputs were unchanged. Its first
normalization matches 363,025 of 393,216 CUDA prefix values, exactly the
standalone safe-math probe count, confirming the intended Rust kernel behavior
on those inputs.

The forced-history level-1 position-4 head still selects code 3690 on Metal,
while CUDA selects 3182: CUDA's 3182-minus-3690 margin is `+9.5367e-7`,
default Metal's is `-4.7684e-6`, and experimental Metal's is `-5.7220e-6`.
The first-block maximum cross-device difference increased from `1.5259e-5`
to `4.5776e-5`, although the final hidden maximum decreased slightly from
`8.2850e-6` to `7.8976e-6`. The forced final IDs remained unchanged. These
mixed results do not justify replacing the default backend or claiming a fix;
the observed target-input difference and other operations remain candidates
for investigation. The frozen Metal raw-ID gate remains failed. Raw evidence is
under ignored `reports/phase4-xsum-norm-candidate-mlx-captured-v2/`.

A separate same-batch forced-history run on row 4883 at `09d6c04` also left
the original raw-ID mismatch unchanged: position 40 is 3646 on both default
and experimental Metal, versus 50256 on CUDA. All other forced IDs matched.
Input hashes were unchanged; the candidate safetensors SHA-256 is
`eea48831659d034ccc7cecf1c6513cca0ef594d85f7d2ec7e806c9b2168ade41`,
and its JSON SHA-256 is
`ec28b0ecabfbe06accd76e60939e0a0da65e5c686da5d8256288ca240276a085`;
local copies match the Mac. This does not qualify a full free-generation run.
