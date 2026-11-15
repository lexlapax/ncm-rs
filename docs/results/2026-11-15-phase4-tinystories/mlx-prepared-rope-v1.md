# TinyStories Metal prepared rotary-table candidate

The candidate at `745bb72` retained expanded rotary sine/cosine arrays per generator level so Metal could reuse them across attention blocks. The default backend path kept the existing half-width tables and rotation, and the Metal candidate used the same rotation arithmetic. The candidate passed the CPU prepared-table test, full `tools/check.sh --cuda`, the pinned CUDA Sudoku generator golden and a 128-row TinyStories strict-cuDNN comparison. On Metal, the prepared operation matched the existing rotation exactly for the tested shapes, the pinned Sudoku generator golden and a 128-row retained TinyStories comparison passed, and `tools/check.sh --mlx` passed 18 real-device operator cases. An initial Metal Sudoku command used a nonexistent local artifact path and failed before inference; the retry used `sudoku-easy-qualified` and passed.

A source-bound exploratory old/new comparison then ran one fresh process per binary and batch with five warmups and 30 timed requests. Both used a process-local 4,096 MiB cache setting and included output drop and a GPU barrier. Order was baseline/candidate at batch 32, then candidate/baseline at batch 128.

| Batch | Baseline / candidate median ms | Baseline / candidate p95 ms | Baseline / candidate RSS MiB | Baseline / candidate MLX active MiB |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 619.35 / 614.16 | 627.64 / 622.59 | 1024.88 / 1024.56 | 477.63 / 477.87 |
| 128 | 2075.43 / 2068.98 | 2092.35 / 2086.43 | 1023.97 / 1024.50 | 477.63 / 477.88 |

These sub-1% latency differences come from one process per side and do not establish a reproducible speedup; candidate active memory was about 0.24 MiB higher. They do not explain or resolve the separately observed multi-second batch-128 tail. The exploratory run had 95 approximately two-second CPU activity samples, no direct GPU activity measurement and no exclusive reservation. This was not a comparison against Python and cannot qualify the performance gate. The process-local cache setting is not an instantaneous hard bound.

Because the observed timing benefit is small and unproven while the implementation adds API and retained state, the candidate was removed and the prior generator and backend source restored. The old/new runner and results remain for inspection; no numerical tolerance or model output was changed. The comparison's ignored capture is `reports/phase4-rope-prepared-mlx-ab-v1/`; `manifest.json` SHA-256 is `043c8bd54b0768419a2079ec81210e59ee271392436400ee5170d82bcd3205f1`, and `activity.jsonl` SHA-256 is `19fc64f5237a07f277afe3b2602e30a333a8b859ab425fe62fa3f16299cf72d3`. Both were verified against the Mac originals.
