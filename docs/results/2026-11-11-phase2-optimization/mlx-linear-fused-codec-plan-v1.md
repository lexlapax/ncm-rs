# Isolated fused interpolation inside the codec

The [finite interpolation proof](mlx-linear-fused-poc-result-v1.md) reduces
peak allocation and elapsed time on ten isolated shapes, but does not establish
model behavior. The adjacent [patch](mlx-linear-fused-codec-v1.patch) is for an
isolated checkout of `05290a7` only; production is unchanged. It constructs a
local MLX kernel handle on each call, preserves the current host index/weight
formula, requests contiguous kernel inputs, and retains the previous gather path
for non-F32, empty or overflowing shapes. The extra operator test compares a
noncontiguous BLC view containing NaN, infinities and signed zero with the
existing same-formula composition, across three output lengths. Bounds and
initialization must be reviewed alongside the narrowly scoped unsafe call.
Creating the handle per call avoids assuming a non-Send/non-Sync handle can be
shared; it may add overhead and is not a proposed final cache design.

Before Metal, run the full `tools/check.sh --cuda` against the patched isolated
Linux checkout, retaining output and failures. The first attempt stopped before
checks because the configured sccache wrapper was unavailable; the second runs
with that wrapper disabled and passed all checks, including 484 report tests
and 18 real-CUDA operator cases. [Logs and probe](raw/mlx-linear-fused-linux-check-v1.tar.xz)
SHA-256 `ceab48877bf9ab07f2635fcfe4a5ef372a6b9996d1b40e85815bcf280fd24e90`. The patch changes only Apple-only operator code and
its Metal-specific tests, and does not alter the model, CUDA adapter, pins or
numerical tolerances.

After Linux passes, build on the Mac in a separate patched checkout and run the
focused Metal operator suite, followed by `tools/check.sh --mlx` if focused
checks pass. Qualify the full trained trace against the pinned reference and
existing unchanged budgets before using full-codec timings. Then compare the
patched and current runtime with three fresh paired processes per binary, rotating
order, using the existing qualified Sudoku bundle and fixed codec benchmark.
Verify all B1/B2/B4 scope input and output hashes; report every median, p95,
allocator peak/active/cache, host HWM, first execution and activity observations.
Record source, executable, native library, dependency and build identities.
Preserve failures and stop if focused tests or qualification fail. The six
processes are an exploratory diagnostic, not an acceptance gate; no repeat or
threshold change is implied. If results are mixed, startup costly or the
resource benefit disappears, retain that result and do not integrate merely
because the isolated operation was faster.
