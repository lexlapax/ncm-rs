# Phase 2 remediation: measurement repair and resampling research

Status: **research/PoC only; neither extended platform gate passes yet**.
The [extension plan](../../PHASE2-PERFORMANCE.md) remains in effect. This milestone
changes report validation and adds an offline experiment, not production inference.
No throughput gain or CUDA qualification is claimed from CPU results.

## Measurement integrity repair

`tools/bench/phase2_mlx_summary.py` previously read baseline evidence from ignored
`reports/phase2-codec/` even when summarizing the published raw bundle. This made
the advertised clean-checkout reproduction fail. It now uses the existing committed
capture/qualification files, with explicit SHA-256 anchors matching the original
benchmark reports. No baseline, oracle, tolerance or raw measurement changed.

Four new report tests cover a checkout with no ignored reports/weights/venvs,
exact reproduction of all historical metrics (only summarizer source provenance
changes), refusal to overwrite outputs, tampered/missing evidence, runtime/source
mutations and baseline substitution. Full report-tool suite: 16 tests pass.
The historical summary is not overwritten or relabeled with new source hashes.

## Research actually performed

Accessed/inspected 2026-11-10. Links are technical provenance; external documentation
is not proof of this project's performance. Local downloaded material remains under
`.reference/phase2-performance-research/`.

1. [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/),
   application profiling and timing sections: realistic workloads, hotspot-first
   optimization, synchronization around CPU timing and separation of CUDA events
   from kernel-only duration. Retrieved HTML SHA-256
   `fcacb0b62ce545e154a6b7dc49778ec663162d7dfb469f85d74b0551e0341f68`.
   Consequence: no acceptance based on dispatch-only timing or contended workstation
   samples; measure the original codec APIs after microbenchmark experiments.
2. [Pinned NCM quantizer](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/quant.py),
   `_down_for_quant` / `_up_to_full`, inspected from the pinned local checkout:
   area/linear interpolation operates on transposed BCL arrays, then materializes
   contiguous BLC output; learned level zero is a different path. A replacement
   must preserve these semantics and cannot remove independent snapshots.
3. Candle 0.11.1 installed source and registry VCS metadata identify commit
   `59f8e903ca80a32ea7f8514b39748fc979e69e82`. Inspected
   [CUDA dispatch](https://github.com/huggingface/candle/blob/59f8e903ca80a32ea7f8514b39748fc979e69e82/candle-core/src/cuda_backend/mod.rs)
   and [bilinear kernel](https://github.com/huggingface/candle/blob/59f8e903ca80a32ea7f8514b39748fc979e69e82/candle-kernels/src/conv.cu).
   Downloaded kernel source matches the installed crate byte-for-byte (SHA-256
   `6d9187017ba53c9511f28962f836c409b326073aab1ecfa036758e0499f9ad4c`).
   The guessed `0.11.1` tag URL returned 404; the resolved commit was used instead.
   **Rejected shortcut:** reshaping 1D to use existing bilinear CUDA dispatch
   would interpolate samples in double precision, not merely calculate coordinates
   in double. It violates the current FP32 arithmetic scope and may also be costly
   on a consumer GPU. It was not adopted or benchmarked.
4. Installed Torch `2.15.0+cu132`, Git
   `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`,
   `torch/include/ATen/native/UpSample.h`: inspected `compute_scales_value`,
   `area_pixel_compute_source_index`, `guard_index_and_lambda` and
   `compute_source_index_and_lambda`. Installed header SHA-256
   `af11f52b3897869432e30ebcc27fc240d40fb5921d8800378ee685ebea5ac5f7`.
   This exposes scalar-typed coordinate/weight arithmetic rather than assuming
   higher-precision host coordinate calculations are interchangeable.
   The CPU experiment below tests this inference against the installed operator;
   CUDA accumulation/FMA behavior must still be qualified separately.
5. Local source inspection: `ops::linear1d` builds two scaled slices and an add
   per output position before concatenation; `area1d` reduces each bin separately.
   Their dispatch count grows with output length. `select_last` also constructs
   indices/sentinels and uses multiple device operations. These are causal
   hypotheses to profile, not measured launch-count or end-to-end speedup claims.

## Small vectorized PoC and retained negative result

`crates/ncm-backend-cuda/examples/resample_poc.rs` explores two gathers plus FP32
weights for linear interpolation, and padded-bin gather/mask/reduction for area.
Shape metadata is constructed separately from application. This avoids writing a
custom kernel before investigating the existing native operations. It is not wired
into `CodecBackend`, and intentionally handles only its fixed finite experiment.

Initial execution rejected strided inputs because Candle `index_select` requires
contiguous storage. The PoC now explicitly materializes it; any eventual benchmark
must include that copy in application time, not silently move it outside timing.
Temporary expanded area storage and metadata uploads must be accounted for too.

Cases: batches 1/2/4, 16 channels, all trained level lengths plus tiny uneven
lengths, both directions to/from length 89, contiguous and strided input, area and
linear operators. Identity cases are deliberately repeated in both directions.
There are 360 rows and 753,984 output values. Synthetic inputs are signed, exactly
representable values; the checker uses pinned Torch CPU and the unchanged primitive
budget `1e-5 + 1e-4*abs(reference)`.

| Coordinate calculation | Versus existing Rust composition | Versus Torch CPU |
| --- | --- | --- |
| f64 host coordinates (existing approach) | 0 mismatches | **70 failing rows / 228 mismatches**, max abs 9.9182e-5, max budget fraction 7.1069 |
| f32 scale/coordinates/weights | 228 mismatches | **360 rows pass**, max abs 4.7684e-7, max budget fraction 0.01095 |

Every failing row is linear; area passes. This shows why testing only against the
old Rust composition would accept the wrong candidate. Existing operator goldens
and 642 trained traces are not rewritten, and this broader synthetic failure does
not invalidate their recorded historical results. Promotion of a coordinate change
requires both those unchanged gates and these broader checks on CUDA and later MLX.
The CPU fix is **not** assumed to establish CUDA behavior or a performance gain.

[FP64-coordinate negative comparison](raw/resample-cpu-f64-comparison.json) and
[FP32-coordinate comparison](raw/resample-cpu-f32-comparison.json) retain every
case, source/input digests and errors. Large reconstructible PoC outputs stay in
`reports/phase2-performance-poc/`, not Git. No historical golden was regenerated.

Reproduce (new output filenames; reference environment from the context map):

```sh
cargo build --release --locked -p ncm-backend-cuda --features cpu --example resample_poc
mkdir -p reports/phase2-performance-repeat
target/release/examples/resample_poc cpu f64 reports/phase2-performance-repeat/f64.json
.reference/venv/bin/python tools/reference/check_resample_poc.py \
  reports/phase2-performance-repeat/f64.json reports/phase2-performance-repeat/f64-check.json
# Expected exit 1 above: retained diagnostic failure, not a passing gate.
target/release/examples/resample_poc cpu f32 reports/phase2-performance-repeat/f32.json
.reference/venv/bin/python tools/reference/check_resample_poc.py \
  reports/phase2-performance-repeat/f32.json reports/phase2-performance-repeat/f32-check.json
python3 -m unittest discover -s tools/bench -p 'test_*.py' -v
```

## Blockers and next experiments

Preflight observed 28% GPU utilization and a resident llama-server (3,232 MiB),
plus desktop applications. Three subsequent `pmon` samples show `-`, not reliable
proof of inactivity. No unrelated process was stopped. A quiet measurement window
was requested; acceptance timing and threshold calibration remain pending.
Nsight Systems/Compute are absent from PATH and the inspected `/opt`/`/usr/local`
locations; CUDA's local CUPTI installation and injection samples are available.
Prefer a project-local tracing setup or user-approved profiler installation;
never silently modify global tooling. No profiler timeline has been captured yet.

Next: establish paired old-Rust/Python baselines and lock noise thresholds; trace
launch/transfer/allocation behavior, validate both PoC variants on CUDA, then time
full codec candidates and investigate decoder/loading independently. An expanded
buffer/gather PoC may lose on bandwidth or allocation even if launch count drops.
If so, use the evidence to decide whether a narrowly scoped kernel is justified.
Apple work waits for extended Linux acceptance. Phase 3 remains blocked.
