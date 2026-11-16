# XSum Metal load memory-map diagnostic

After the user resumed Mac work, the clean Mac checkout was fast-forwarded from
`1039e2c` to pushed `e14ae02`. On 2026-11-16, one batch-1 Rust baseline run and
one Python MLX baseline run were sampled with `/usr/bin/vmmap -summary` about one
second apart while the existing short performance programs ran. The samples are
intrusive and sparsely timed; they do not measure an equivalent point in loading
or provide peak process RSS. The Rust and Python output codes and IDs each matched
their respective retained baseline output. Both processes exited successfully.

The first Rust snapshot showed a 598.9M physical footprint and 572.7M of
`MALLOC_LARGE`. The second showed a 4.9G physical footprint dominated by
4.9G of `IOAccelerator (graphics)` regions; later attempts found the process
had exited. The first Python snapshot showed a 6.2G physical footprint,
including 5.6G of nonvolatile graphics regions and 571.3M of empty large
malloc regions; later attempts found the process had exited. These snapshots
were taken at different execution stages and physical footprint includes graphics
memory, so they cannot be used to compare process RSS, establish a memory winner,
or identify the cause of the previously observed Rust RSS gap. The suspected
retained codec payload remains a hypothesis.

Raw reports remain on the Mac in `reports/phase4-xsum-mlx-load-vmmap-v1/` and
`reports/phase4-xsum-python-mlx-load-vmmap-v1/`; exact copies under local
`reports/phase4-xsum-mlx-load-vmmap-captured-v1/` and
`reports/phase4-xsum-python-mlx-load-vmmap-captured-v1/` passed per-file SHA-256
comparison against the Mac. The Rust input hashes are in the captured report.
The Python run used `.reference/mlx-venv/bin/python tools/reference/phase4_xsum_mlx.py`
with the same artifact and 32-passage input file, `--batch 1 --bench`, and the
existing independent MLX adapter. This diagnostic is not a quality or performance
gate and does not exercise the uncommitted owned-loader candidate.

The Rust command was `target/release/examples/xsum_perf_screen
artifacts/canonical/xsum-inference-rope-v2-final
reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json 1`. The Python command was
`.reference/mlx-venv/bin/python tools/reference/phase4_xsum_mlx.py --artifact
artifacts/canonical/xsum-inference-rope-v2-final --inputs
reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json --batch 1 --bench --output
reports/phase4-xsum-python-mlx-load-vmmap-v1/output.json`. In each case, the
shell launched the command and then attempted `/usr/bin/vmmap -summary "$pid"`
after each one-second delay, eight times; reports include failed attempts after
process exit. This is an observational screen and vmmap itself perturbs timing.

A later non-gating RSS sampling screen used the retained baseline binaries,
`tools/bench/sample_xsum_mlx_rss.py` (SHA-256
`7c60f876dcf9d84fc7465d8f2456d4c4da7e1456034b8717d57770b827f68aaf`), and
one fresh batch-1 process per runtime. It sampled `/bin/ps -o rss=` roughly
every 50 ms while each ran the existing two-warmup/five-trial program; Rust and
Python outputs matched. Rust's largest sampled RSS was 1303.27 MiB at 1.15 s
of a 2.56 s process; Python's was 1241.86 MiB at 2.01 s of a 2.27 s process.
These are sampled values, not guaranteed peaks, and the stages were not marked;
they reproduce the direction of the earlier screen but cannot attribute the
gap to loading or compare generation speed. The sampling itself perturbs timing.
The 39 Rust and 34 Python samples and identities are in
`reports/phase4-xsum-mlx-rss-samples-captured-v1/summary.json` (SHA-256
`426381ceeb2d7ce34bffa61594f8d3250786ec1a0e35a0d6f0067bdc9086a7de`), copied
and verified per-file against the Mac.
