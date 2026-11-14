# Decoder Linux quiet-window diagnostic

After the [failed Linux gate](decode-final-linux-gate-v1.md), the user
reported the Linux desktop quiet and asked work to continue. This fixed
diagnostic ran three fresh processes per runtime across every codec, BERT and
TinyStories scope and checked load, using the exact same native benchmark
binary hashes, CPU placement, fixtures and timing settings as the failed
gate. It is a diagnostic, not an acceptance rerun; it does not replace either
failed audit. The fixed plan SHA-256 is
`85499dbc3bdb23977b8c3cf414f2908ed68d961f34ec0fce86fc6dca5c7779e3`.

For a simple within-process spike indicator, the second-slowest of 30 trials
was divided by that process's median. A ratio above 1.2 occurred in two of
three candidate BERT skip-special processes and two of three candidate
TinyStories decode processes in the initial matrix; it occurred in zero of
three for each in this quiet diagnostic. In the nine-process control it
occurred in three and one candidate processes, respectively. For batch-2
resident block, the quiet diagnostic had one of three previous-Rust processes
above 1.2 and none of three candidate processes; the prior control had six of
nine previous and four of nine candidate processes above 1.2. This indicator
is not the acceptance p95 calculation, and the samples are too small to
establish causation or a passing gate.

The quiet diagnostic recorded 106 GPU samples, with median reported GPU
utilization 4% and median sampled SM clock 1,710 MHz. The earlier initial and
control captures recorded median utilization 10.5% and 7%, respectively;
both had the same 1,710 MHz median sampled SM clock. Samples include benchmark
work. The process monitor recorded nonzero compositor and terminal GPU
activity during the quiet diagnostic, so it does not establish exclusive GPU
access. No clock or power policy was changed.

[Raw reports, CPU/GPU telemetry and plan](raw/decode-quiet-diagnostic-v1.tar.xz)
have SHA-256 `b98290f7bd07c3919336b4fa62f251b7e4dce8022a002c43ae282a9fd5693ca9`.
