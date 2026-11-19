# Phase 5 Linux fresh-clone tokenizer setup slice

2026-11-19. This verifies only the pinned-source and tokenizer section of the draft
[setup guide](../../USAGE.md#pinned-sources-and-tokenizer-assets), not the full Phase 5 setup
gate, GPU execution, model quality or macOS. A fresh remote clone of `ncm-rs` at
`e678a409926d47bd56bfc5eace8b36853ec86e0b` was created under ignored
`reports/phase5-guide-tokenizers-fresh-v1/` on the Linux development machine. The guide's
commands were run under `setpriv --inh-caps=-all` from that clone: clone and detach the upstream
at `c70274273f489abc73832b2b98584a8483b6b203`, create a new tokenizer venv, install
`tools/reference/requirements-tokenizers.lock`, fetch the pinned source assets, and convert them.
These operations used the network. Python was 3.14.10; the venv was independent of the retained
reference environments. The fetch accepted the tracked source lock, and conversion completed for
BERT, GPT-2, Qwen3 and TinyStories.

After conversion, the guide's additional lock-check code verified the manifest and tokenizer
payload SHA-256 against `tools/reference/tokenizers-v1.lock.json` for all four profiles (eight
hashes). It does not establish tokenizer behavior, generation equivalence, or quality. The command
was run after the original guide block; it was then added to the guide. The guide commands assume
fresh output paths and a trusted, stable working directory.

The ignored capture `reports/phase5-guide-tokenizers-fresh-v1.log` has SHA-256
`5674812ca1bd830ab911bf5ac41a84e475705b66eb9ced7954d6772d7135c7cf`. The temporary clone's
tracked checkout and nested upstream checkout were clean; its ignored files were the new
`.reference/` and `artifacts/` directories. After inspecting the clone, it was removed. No model
checkpoint was downloaded and no CUDA or Metal test was run. The full Linux and Mac guide replays,
Sudoku medium/hard identity decision, and remaining Phase 5 gates remain pending.
