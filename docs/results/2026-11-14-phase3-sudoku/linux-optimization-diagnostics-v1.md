# Phase 3 CUDA optimization diagnostics — preliminary

The user requires Rust to be faster than Python and asks for efficient memory use. The initial [three-process timing baseline](linux-baseline-timings-v1.json) did not meet the speed objective: Rust/Python median ratios were 1.057, 1.249, 1.244 and 1.258 for batches 1, 4, 32 and 128 respectively. Background desktop GPU processes were observed. These observations are retained rather than treated as a passing gate.

The following isolated batch-32 Rust experiments each used one fresh process, five warmups and 30 synchronized measured calls. They are directional diagnostics, not a controlled final result. All had the same code/ID hash as the reference workload.

| Candidate | Rust median | Comparison / scope |
| --- | ---: | --- |
| Initial implementation | 569.78 ms | Median of the earlier three-process baseline |
| Reuse accumulated latent; avoid generator snapshots | 543.64 ms | One process; additionally releases checked artifact after loading |
| Existing Candle split-half CUDA RoPE | 465.38 ms | One process; uses the same RoPE formula, with normal FP32 rounding differences checked against goldens |
| Pack Q/K/V in one layout materialization | 446.91 ms | One process |
| Specialized bit-exact CUDA layout copies | 420.78 ms | One process; the nearby Python exploratory median was 453.99 ms |

The first specialized layout test reported CUDA default-pool `used_peak` 462,636,408 bytes and `reserved_peak` 536,870,912 bytes, covering the whole Rust process. The nearby Python process reported PyTorch allocator peaks of 1,309,920,256 bytes allocated and 1,430,257,664 bytes reserved after its model had loaded. These allocator counters have different coverage and reset scopes; they are not directly interchangeable physical-memory measurements. The prior paired process-level GPU samples observed approximately 676 MiB Rust versus 1,470 MiB Python at batch 32, sampled once per second. Whole-process RSS and sampled GPU-process memory, including their limitations, need to be remeasured for the final candidate.

The layout kernels are bounded to the profiled eight heads and 64-wide heads; other shapes use the existing native generic layout operation on the same device. They only copy FP32 bit patterns, preserve the upstream sequence and head layout and do not alter model arithmetic. CUDA regression tests cover shapes including batch 128, short and irregular sequence lengths, signed zero and NaN payloads. The synthetic pinned Python golden also passes. The full 6,000-board comparison and real Apple GPU checks remain required after all changes.

The initial source and executable snapshots are retained under `.reference/phase3-baseline-preserved/`, and later intermediate candidate snapshots under `.reference/phase3-{latent-reuse,fused-rope,packed,geometry,typed}-source/`. Raw exploratory process and profiler output remains local under `.reference/phase3-initial-oracle/` and `reports/phase3-linux-candidate-probe-v1/`. Measurements must not be interpreted as final acceptance, especially the one-process comparisons.
