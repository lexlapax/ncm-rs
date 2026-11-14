# BERT WordPiece vocabulary reservation — provisional

The hash-checked BERT profile fixes a 30,522-entry WordPiece vocabulary. Its
candidate parser reserves that capacity before deserializing entries and then
constructs the same pinned `tokenizers` WordPiece model; normalization,
processing, encoding and decoding remain delegated to that implementation.
The profile capacity is a bound, not an input-derived allocation hint. The
existing manifest, payload and license checks precede parsing and remain intact.
The other three tokenizer profiles retain their existing implementations.

The `asset_bench`, `asset_allocations` and `asset_probe` source fingerprints now
include the new parser; the probe also fingerprints the existing BPE parser.
This expands the reported source inventory, not an artifact or oracle schema.

Three fresh matched BERT benchmark processes per binary ran in rotating order
on CPU 2, using five warmups and 30 trials. The previous process load medians
were 3.9094, 3.8661 and 3.9207 ms; candidate medians were 2.4720, 2.4690 and
2.4671 ms. The median of process medians fell **3.9094 to 2.4690 ms** (about
37%). Process load HWM ranged 15.68–15.73 MB before and 14.22–14.38 MB after;
three processes are diagnostic, not a final memory claim. Encode, decode and
skip-special output hashes matched across all six runs.

One separate allocation capture per binary showed median requested Rust
allocation bytes per checked load falling **7,776,931 to 5,613,868**, and
allocation requests **61,429 to 61,408**; reallocations remained 82. These are
allocator-request counts, not whole-process or physical memory. The pinned
real-asset probe passed all four profiles, their goldens and rejection cases in
a cleared environment. `cargo test -p ncm-tokenizers --locked` passed 18 tests;
the first run failed a new fixture because the pinned WordPiece deserializer
requires borrowed input, corrected by using `from_str`. Tokenizer Clippy and
format checks passed. The retained [raw reports, probe and binary hashes](raw/bert-wordpiece-v1.tar.xz)
have SHA-256 `4e6f12d3866e30350e9863ae123c611b4ab756e0c6da788373a75626b210cab4`.

The saved pre-change binary was run from the clean `bcf2be4` worktree, whose
tokenizer sources and lockfile match `edcd4d3`; the candidate was run from the
current tree. Both used `TOKENIZERS_PARALLELISM=false taskset -c 2 BINARY
ABSOLUTE_CANONICAL_ROOT tests/fixtures/tokenizers-v1.json bert NEW_REPORT`.
The probe used `env -i PATH=/usr/bin HOME=/tmp TOKENIZERS_PARALLELISM=false
target/release/examples/asset_probe artifacts/canonical/tokenizers-qualified
tests/fixtures/tokenizers-v1.json`. No full three-way matrix, fresh CUDA suite,
Apple measurements or acceptance claim follows from this isolated experiment.
The prior 76/78 Linux audit was for the earlier finite-scan candidate.
