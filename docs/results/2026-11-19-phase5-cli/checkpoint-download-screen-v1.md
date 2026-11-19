# Phase 5 pinned checkpoint download screen

2026-11-19. `tools/reference/fetch_checkpoints.py` imports the existing exporters' checkpoint
sizes and SHA-256 pins; it does not introduce a new checkpoint revision or change an artifact.
It requires an explicit profile, fresh destination and byte budget; writes each file with a
pinned-size cap (reading one extra byte to detect overflow); verifies size/hash, HTTPS final URL and
available disk; and rechecks files after download. `--plan` does not create files or use the
network, and `--verify` checks retained sources offline. The destination parent must be trusted and stable. A failed download retains the
partial directory for inspection. Available disk is not reserved and does not cover later exports.

## Checks

- All nine profile pin sets passed `--verify` against the retained local sources: three Sudoku,
  TinyStories, XSum, WMT14 and three LM1B sizes. This reads the actual files and verifies their
  sizes/hashes, not just the listed metadata.
- One real network run downloaded Sudoku easy's config, generator and tokenizer into a fresh
  directory, using an exact 162,378,392-byte budget. All three file bytes matched the retained
  pinned sources. Tool SHA-256: `fe518bf6e349a858a73630eaaa1e40e9cf4e215b41276ace84dee0f06d355238`.
- The current Sudoku exporter consumed those downloaded files with `bit_exact_roundtrip: true`.
  Its new manifest `74cce0a9e87fc802b19854d733dc69b97c2e6e366763c7afc155d13659fa5d51` is
  **not** the qualified manifest and is not proposed as a replacement. The
  [provenance comparator](../../../tools/reference/compare_artifact_provenance.py) verified every
  non-provenance file byte-for-byte against the retained qualified easy artifact; only the recorded
  converter base/hash and Python version differ. Result SHA-256:
  `b49728019a7ef7bc2b9ca1fffd4f2ee0f1beec4adb77c303129a4ad452c20f91`.
- Seven offline unit tests cover no-overwrite, budget/disk checks before network, short/long/wrong-hash
  downloads, HTTP redirect rejection, modified sources, symlinks, plan-only and partial-failure behavior.
  They are included in `tools/check.sh`. The first full check failed the unrelated lifecycle test
  `test_real_writers_and_owned_process_samples_compose` with `capture_error:ObservationMismatch`;
  a focused retry and two subsequent full checks passed. The cause is not established; the failure
  is retained in `/tmp/ncm-phase5-fetch-check.log` for this session.

Reproduction of the actual network screen, from a checkout with the Linux reference environment:

```sh
setpriv --inh-caps=-all .reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile sudoku-easy --output reports/phase5-sudoku-easy-download-v1 --plan
setpriv --inh-caps=-all .reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile sudoku-easy --output reports/phase5-sudoku-easy-download-v1 --budget-bytes 162378392
setpriv --inh-caps=-all .reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile sudoku-easy --output reports/phase5-sudoku-easy-download-v1 --verify
```

Use a fresh destination for a rerun. No other checkpoint profile was freshly downloaded in this
screen, no fresh clone was used, and no Mac setup or GPU quality/performance claim follows.
Full export recipes and the remaining release gates are still pending.
