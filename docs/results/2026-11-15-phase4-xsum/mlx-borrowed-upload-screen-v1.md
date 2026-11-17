# XSum borrowed upload: Metal checks and exploratory screen

The [Linux CUDA candidate validation](borrowed-upload-plan-v1.md) completed its
full output comparison, real-device Sudoku trace and repeated performance matrix.
The first combined build command selected examples but left the CLI unchanged;
this was detected by its identical binary hash. The initial CUDA and Metal CLI
results therefore apply only to the previous binary. After explicit CLI builds,
the CUDA 32-row Python comparison passed, and the Metal CLI comparison also
passed. The first Metal full check and short screen started before the corrected
CUDA CLI check completed; they are retained as exploratory evidence.

After the corrected CUDA check, full `tools/check.sh --mlx` passed on clean
commit `f03650b`, including 18 real-Metal operator cases. The command used
`PATH=/opt/homebrew/bin:$PATH`, `MLX_SYS_PREBUILT=$PWD/.reference/mlx-native`
and `RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"` in that process.
The captured log `reports/phase4-xsum-borrowed-upload-mlx-check-captured-v2.log`
hashes to `ffa432f70881d56d4bf184a087e4c74dc152ef6f3c152e3216f0e137080e21c0`.
The explicitly rebuilt candidate CLI SHA-256 is
`a580ba2ca6c8a2e1a43d4e7a0b3a7b6645b6576f7424ec2ea3adb711a2429414`; its
32-row Python comparison passed after the ordered full check, with result
`reports/phase4-xsum-borrowed-upload-mlx-cli-captured-v3/result.json` SHA-256
`30415fa65a9346989ea8b4924a27d565bfd448ece82c310cb09adc148d7f3d12`.

On the two contiguous batches containing rows 4883 and 4937, the rebuilt
candidate CLI reproduced every code and ID in the saved Metal baseline. Input
and binary hashes passed postflight; the two output hashes were
`13126af2482cd1a8bb7bd83bd382557b01b10d8fa1d718654ccc339629d4cce5` and
`1662e6869c012e017df154da76d72d83f634123dd86b874010e73e6335738050`.
The copied outputs are under
`reports/phase4-xsum-borrowed-upload-mlx-divergence-captured-v1/` and match
the Mac hashes.
This preserves rather than resolves the [full-corpus raw-ID failure](mlx-full-top1-v1.md).
The complete candidate corpus has not been captured.

The earlier exploratory screen used one fresh process per runtime and batch,
two warmups and five trials. Rust and adapted Python MLX produced matching
codes and IDs; the seeded Rust output matched the retained seed-42 Metal output.

| Batch | Runtime | Median ms | Peak RSS MiB |
| ---: | --- | ---: | ---: |
| 1 | Rust | 213.46 | 1205.22 |
| 1 | Python MLX | 238.73 | 1245.97 |
| 4 | Rust | 543.99 | 1205.77 |
| 4 | Python MLX | 597.38 | 1293.16 |
| 32 | Rust | 2939.17 | 1203.77 |
| 32 | Python MLX | 4459.76 | 1645.58 |

These short point measurements favor Rust in latency and process RSS, unlike
the prior small-batch screen, but do not establish repeatability, causation or
an ordered performance pass. The new performance example hash
`393d59b91fc02562d611cbaf2e45d1d9aed29480fbdad096ceb5235e004c806f` differs
from the prior screen's executable. The summary at
`reports/phase4-xsum-borrowed-upload-mlx-screen-captured-v1/summary.json` hashes
to `fe6856561c1e432d0b03062a022a8e4cda5c5dfe672746839b8fa605984ac643`;
all copied screen files matched Mac hashes. The checked alignment probe found
all 53 codec and 193 generator FP32 payloads aligned in that Mac process. Each
actual upload still checks alignment independently. The [repeated plan](mlx-borrowed-upload-repeated-plan-v1.md)
records the pending performance test and its limits.
