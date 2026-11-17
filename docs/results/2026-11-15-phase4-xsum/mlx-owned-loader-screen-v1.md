# XSum owned loader: Metal screen

After the [Linux CUDA check, full quality-output comparison and repeated performance
run](cuda-owned-loader-v1.md), pushed `d018aa3` was pulled into the clean Mac
checkout. Full `tools/check.sh --mlx` passed with 18 real-Metal operator cases
(log SHA-256 `10c62f43e88d150f89641cdc4200feacb93b5d0eb1abd0e4da30a04e80a5eaa8`).
The owned-loader CLI matched the pinned Python CUDA reference on all 6,400 codes
and 1,536 IDs in the existing 32-passage top-1 screen, and its seed-42 codes and
IDs matched the retained fixed-point Metal baseline for the same passages. These
are bounded screens, not a full candidate quality comparison; the baseline's
[full-corpus raw-ID gate failure](mlx-full-top1-v1.md) remains open.

The following one-process-per-runtime screen used two warmups and five measured
requests with synchronized generation and output readback. Rust and the adapted
Python MLX reference matched all checked codes and IDs. RSS includes load; timing
excludes it. MLX peak has a different allocation scope from process RSS.

| Batch | Runtime | Median ms | Peak process RSS MiB | MLX peak MiB |
| ---: | --- | ---: | ---: | ---: |
| 1 | Rust | 211.84 | 1296.73 | 1117.88 |
| 1 | Python MLX | 239.86 | 1246.48 | 1206.26 |
| 4 | Rust | 544.66 | 1303.75 | 2318.73 |
| 4 | Python MLX | 598.07 | 1293.22 | 2492.07 |
| 32 | Rust | 2930.55 | 1295.50 | 2583.15 |
| 32 | Python MLX | 4482.08 | 1645.50 | 2828.37 |

Rust is faster in all three cells but exceeds Python RSS by 50.25 MiB at batch 1
and 10.53 MiB at batch 4. The previously observed small-batch memory failure
therefore remains. These are short point measurements, not a paired estimate of
loader improvement or a formal gate; there was no continuous activity monitor,
and before/after process snapshots do not establish exclusivity. The source and
input receipts matched before and after. Python MLX is an adaptation initially
qualified on 32 top-1 inputs, not original PyTorch on Metal.

On the Mac, each Rust run used `/usr/bin/time -l target/release/examples/xsum_perf_screen
artifacts/canonical/xsum-inference-rope-v2-final
reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json BATCH`; each Python run used
`/usr/bin/time -l .reference/mlx-venv/bin/python tools/reference/phase4_xsum_mlx.py
--artifact artifacts/canonical/xsum-inference-rope-v2-final --inputs
reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json --batch BATCH --bench --output OUTPUT`.
Batches ran in order 1, 4, 32, Rust then Python. The
[summary](mlx-owned-loader-screen-v1.json) (SHA-256
`0ecc388497a96c6ccfbf0f38cc185983bf3d0806c313134693b7d29d053fc1ec`) is reproduced
by `tools/bench/summarize_xsum_mlx_screen.py`; its focused parser tests passed.
Raw reports were copied and per-file hash-verified under
`reports/phase4-owned-loader-mlx-screen-captured-v1/`; the CLI comparison is
`reports/phase4-owned-loader-mlx-cli-captured-v1/result.json` (SHA-256
`7d9fe3c181a9a42a654790a9889394e6a43aaaa284f623e71ed750c0c0e3ce0d`).

## Intrusive heap snapshots

A follow-up paused only each owned test process after a sampled RSS threshold of
1,100 MiB, collected `vmmap -summary` and `heap --sortBySize --noContent`, then
resumed it. Both commands and both processes exited successfully, and Rust/Python
outputs matched. This instrumentation is not a performance measurement; threshold
overshoot and different execution stages prevent a direct causal comparison.

At the Rust snapshot, `heap` reported one live allocation of 345,824 KiB, close
to the 345,813.6 KiB generator payload file, and `vmmap` reported 341.4M live
`MALLOC_LARGE` plus 322.2M resident `MALLOC_LARGE (empty)`. At the Python snapshot,
`heap` reported live 345,824 KiB and 236,864 KiB allocations, close to the
respective generator and codec payload sizes; `vmmap` showed 569.0M live
`MALLOC_LARGE`. The allocation sizes are consistent with those payloads but do
not prove allocation identity. The candidate's live codec-sized allocation was
not seen in this snapshot, while substantial empty large regions remained
resident. This narrows the lifetime hypothesis, but the snapshots were at
different stages and do not establish that those regions caused the 50.25 MiB
RSS gap. Graphics/unified-memory counters have a separate scope.

The exact script is `tools/bench/snapshot_xsum_mlx_heap.py` (SHA-256
`c39c6b66ae0bc3f1eb82743603f49ad89c0bd2e067c68677334379f0432aba8a`);
the copied report `reports/phase4-xsum-mlx-rss-stop-captured-v1/summary.json`
hashes to `17a7a6331d612b93853d8c85f9be265648b5d990040e378a6c9ac7a1738a103f`.
Every copied file was hash-verified against the Mac. Further stage-aligned
measurements are needed before attributing the peak or changing allocator behavior.
