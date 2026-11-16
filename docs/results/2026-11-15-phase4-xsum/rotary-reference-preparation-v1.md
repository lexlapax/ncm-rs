# XSum rotary reference preparation

The [CUDA near-tie investigation](cuda-near-tie-investigation-v1.md) found a
batch whose sampled output depends on small differences in rotary tables and
other operations. `tools/reference/xsum_rope_reference.py` prepares a
separately versioned offline reference for subsequent implementation work.
It does not alter the existing XSum artifact, select a new runtime, or change
production inference.

The tool verifies the pinned upstream commit, the source file containing
`precompute_rope_cos_sin`, its package initializer, and the reference Torch
version. It calls that upstream function for positions 0–511, checks that
its two rotary halves agree, and validates the exact FP32 tensor hashes before
writing a safetensors payload and a manifest. The tables contain the first
32 cosine and sine values for each position; the existing target positions
repeat positions from that range. The expected hashes are tied to the
recorded reference environment. Other systems may fail the hash check; they
must not silently substitute a different table. This is a reference artifact
only, not a new accepted model artifact or schema.

On the recorded Linux reference environment, two independent invocations
produced byte-identical manifests and payloads. The final generator SHA-256
was `66327c5c2bcf3349fb84d707786bf9ec19374e7b6c81dec1a74617c22a71e405`;
the payload SHA-256 was
`5d205c1c42b467c00e10d03739e18b6433185bcc4c4f9b066d7c6be69ba8346a`.
The targeted reference tests passed, including rejection of a modified
upstream source and an existing output directory. Locally retained outputs are
`reports/phase4-xsum-rope-reference-v5/` and `-v6/`; they are ignored by Git.

```sh
.reference/venv/bin/python -m unittest discover -s tools/reference -p test_xsum_rope_reference.py
.reference/venv/bin/python tools/reference/xsum_rope_reference.py --upstream .reference/upstream-ncm --output reports/phase4-xsum-rope-reference-v5
```

Choose a new output directory for another run; the tool refuses to replace an
existing one. A future runtime integration needs its own explicit artifact
profile, validation, hardware checks and comparisons. The full XSum CUDA and
MLX gates remain pending.
