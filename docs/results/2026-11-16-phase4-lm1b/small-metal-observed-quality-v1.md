# LM1B-small Metal observed-seed quality retest

The experimental Metal CLI at commit `4fd568b` generated 256 samples for
each already-observed seed 42, 0, 1, 2 and 3, in batch-eight requests.
This is a retest, not fresh held-out evidence. It uses the same fixed
[protocol](heldout-protocol-v1.md), pinned artifact and rotary table,
SplitMix64 request seeds `(seed << 32) | batch`, host FP64 sampler and
penalties. The Mac's `dotsetup.rmlx` service remained running. Each
request ran in a separate CLI process; its output hash is recorded with
the capture. The first seed-42 request exactly matched a separate earlier
batch-eight CLI run. All five transferred capture hashes matched the Mac.

The pinned GPT-2-large scorer ran on Linux CUDA, checking each capture's
Metal binary, source, artifact, rotary, request seeds and hash. Its only
adaptation to the original scorer was Metal identity admission and file
locations; the scoring calculations stayed unchanged. Perplexity and all
individual entropy scores matched the pinned upstream scorer exactly for
each seed. A forged CUDA-penalty identity was rejected. Original Python
capture and score hashes were checked against the previously retained
[CUDA held-out summary](small-heldout-cuda-v1.md).

| Seed | Metal perplexity | Metal mean entropy |
| ---: | ---: | ---: |
| 42 | 90.475563 | 4.31217866 |
| 0 | 95.230335 | 4.31896905 |
| 1 | 92.868411 | 4.30977054 |
| 2 | 96.180485 | 4.31838078 |
| 3 | 93.282320 | 4.31022133 |

The unchanged independent seed-bootstrap comparison **fails** the entropy
interval. Metal minus Python mean entropy is `-0.0007996202`, within the
`±0.0046` point limit, but its 95% interval is
`[-0.0050203402, 0.0036893424]`, extending below `-0.0046`. Perplexity
difference is `-2.264292`, with interval `[-4.675775, 0.202392]`; its point
and interval checks pass. The failed original CUDA quality result and the
Metal controlled-sampling failures remain separate and unchanged. This
observed-seed retest does not qualify Metal or complete Phase 4.

The identity-bound summary repeated byte-for-byte with SHA-256
`1aedd2aed0de83c242fd6f9d346cb15610bf142ce28278a26adc23c5e4591b13`;
the unchanged metric comparator's three unit tests passed. The capture,
Metal scorer and summary script hashes are respectively
`054d3618bd28b3acda13a5761e3cfd6ca579b1bbfd014fe1014339f137893070`,
`a56b9b2310a3b44c03d6c0856ed565f7ec77d28d414fcb9586a99b03a6d90251` and
`fcce66e001a2e30a44e613d59ef98eddda6eebf8fefd4559f51a4462c2fb611e`.
All captures, per-request identities, scores, sources, logs and repeated
summary are retained under `reports/phase4-lm1b-metal-observed-quality-v1/`.
The CLI binary SHA-256 was
`fd09fa0ba4f5ede495fa7ab52b3c1d9ef83ce1990dbdeb82738fa1b1173072c2`.
