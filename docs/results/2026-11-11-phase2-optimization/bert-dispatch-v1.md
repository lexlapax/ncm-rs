# BERT parser dispatch isolation — rejected

This API-preserving experiment moved the BERT parser into a non-inlined helper
to test whether separating its compiled dispatch from Qwen improved loading.
It did not change tokenizer models, hashes, dependency sources or asset checks.
Tokenizer tests passed, and the real four-profile parity/rejection probe passed.

Three fresh matched processes per binary used CPU 2, five warmups and 30 trials.
Median of process medians for Qwen checked load was **66.5763 to 66.9538 ms**;
for BERT it was **2.4541 to 2.4720 ms**. Neither showed a material benefit.
All encode, decode and skip-special output hashes matched for both profiles.
The isolated candidate was reverted without repeating it. The fixed Qwen
acceptance failure is not replaced by this diagnostic.

The [patch, raw reports, probe and binary hashes](raw/bert-dispatch-v1.tar.xz)
have SHA-256 `c2e118c87ec6d4af0f6767234d8b98c9ba75538b5e04468d490b31db0daa1042`.
The previous binary used an isolated source archive of `578a3d9`; the candidate
used the current source. Both used `TOKENIZERS_PARALLELISM=false taskset -c 2
BINARY ABSOLUTE_CANONICAL_ROOT tests/fixtures/tokenizers-v1.json PROFILE
NEW_REPORT`. Production and the pinned dependency remain unchanged; the
combined candidate's Linux gate remains 77/78 and Apple remains pending.
