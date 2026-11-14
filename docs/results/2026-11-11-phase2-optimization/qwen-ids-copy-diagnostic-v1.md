# Qwen encode ID-copy experiment — deferred

An isolated proposal adds a consuming ID accessor to the pinned vendored
`tokenizers` encoding and uses it after the existing `encode_fast` pipeline,
eliminating the final ID-vector copy. It does not alter tokenization or the
upstream dependency version. It is not integrated: the diagnostic did not
demonstrate a benefit sufficient to justify another dependency change.

The candidate passed `cargo test --locked -p ncm-tokenizers` and all four real
canonical tokenizer probes on Linux. Three fresh processes per binary ran the
pinned 89-string Qwen workload with five warmups and 30 trials, alternating
order. The median of per-process encode medians was 0.93338 ms for current and
0.94840 ms for candidate (ratio **1.01610**); the three paired ratios were
1.01610, 1.06688 and 0.97861. Encode, decode and skip-special output hashes
matched across all runs. These few, unpinned CPU processes do not establish a
regression or general performance result; no Apple test or acceptance gate was
run for this proposal.

The [raw diagnostic](raw/qwen-ids-copy-diagnostic-v1.tar.xz) has SHA-256
`0ceec140d4e68191bfd4ec6c5599563167efd8018dedbd52b55c0f60b3730781`.
It contains the patches, runner, commands, binary hashes, raw reports and probe.
All six standard source inventories matched their corresponding trees. The
benchmark inventory omits `vendor/tokenizers-0.22.2/src/tokenizer/encoding.rs`;
its candidate SHA-256 `58aa0a809add2b1074aad4747bab67bc4e7d62f80a53a612d328a52a410eca48`
is separately recorded in the plan, while current is
`9866cf67379cbffb48e912102da5712fb4f84ff10be1269d60805f901cb781c5`.

The current Apple Qwen encode cell remains pending. Investigate other
profile-backed opportunities without changing the pinned algorithm or treating
this negative diagnostic as an acceptance run.
