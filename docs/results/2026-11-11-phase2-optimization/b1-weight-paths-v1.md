# Fixed decoder weight paths — rejected

This candidate constructed the fixed decoder linear and normalization weight
paths at compile time, avoiding runtime string formatting while retaining the
same names, tensor operations and tracing. The macro generated weight and bias
paths from the same literal to avoid separately maintained names. The model unit
and loading tests passed. The [rejected patch](raw/b1-weight-paths-v1.patch.gz)
has SHA-256 `2dced4b714f65cf9d416144220c1cc3d28ca0837dde9bf9fbe59dbc7a493b823`.

Three fresh CUDA benchmark processes per binary used the existing codec workload,
CPU 2, five warmups and 30 trials. Process order was previous/candidate,
candidate/previous, previous/candidate. Previous was the saved finite-scan
binary from `a6290db`, run from an isolated `git archive` of that commit;
the candidate used the current worktree. All 12 batch/scope output-ID hashes
matched. B1 resident-block medians in milliseconds were:

| Process | Previous | Candidate |
| --- | ---: | ---: |
| 1 | 0.177991 | 0.198626 |
| 2 | 0.198483 | 0.179871 |
| 3 | 0.179228 | 0.198572 |

The slow/fast clusters recur in both binaries; these measurements do not
establish whether this patch caused any difference. B2 and B4 block medians
changed by less than 0.25%, and no material benefit was established. The
candidate was reverted without repeating the experiment. [Raw reports and
binary hashes](raw/b1-weight-paths-v1.tar.xz) SHA-256:
`c5cf799256f237052cc06acbdfa0c0db1bcd974a71bd25a307b00c72de2b8047`.

Each command was `taskset -c 2 BINARY ABSOLUTE_CANONICAL_BUNDLE
tests/fixtures/codec-bench-v1.json NEW_REPORT`. This is a diagnostic, not a
quality capture or an acceptance audit. The local release `codec_bench` binary
reflects the rejected patch and must be rebuilt before the next candidate run.
The BERT optimization remains provisional; the previous 76/78 Linux gate and
both platform acceptance requirements remain open.
