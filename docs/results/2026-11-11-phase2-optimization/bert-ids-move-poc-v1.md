# BERT token-ID ownership — isolated experiment rejected

The Apple controlled audit left BERT encode pending on the process-level p95
interval. One possible allocation reduction was tested in a temporary checkout:
add a consuming ID accessor to the pinned vendored tokenizer `Encoding` and
move its IDs instead of copying them after `encode_fast`. Production and the
vendored dependency were not changed. The temporary extra dependency change was
not approved for production and was never promoted.

Three matched fresh processes per binary on Linux CPU 2 used the unchanged BERT
fixture, five warmups and 30 trials. Median-of-process BERT encode time changed
from **1.05689 to 1.05225 ms**, a ratio of **0.99562**, not a material benefit.
Decode medians changed 0.47883 to 0.48695 ms and skip-special decode 0.40711
to 0.41246 ms, so the small sample also showed unrelated variance. This is a
Linux diagnostic, not a Mac acceptance result, and does not explain the Mac p95
uncertainty. The temporary candidate passed 19 tokenizer tests and all four
real-asset profile probes. No Apple acceptance retry was run.

[Raw results, build log, source patches and binary identities](raw/bert-ids-move-poc-v1.tar.xz)
have SHA-256 `7617d6384aa1cf0fc91826e4d758868a16c5647ee84ddbdaccabe00800831f2e`.
