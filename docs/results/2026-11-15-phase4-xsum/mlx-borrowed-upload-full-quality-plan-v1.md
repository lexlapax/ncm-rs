# XSum borrowed upload: full Metal top-1 output comparison plan

The borrowed-upload candidate passed the Linux CUDA full output comparison and
repeated performance measurements, then ordered Metal checks, 32-passage CLI and
seeded replays, and the repeated Metal performance comparison. The candidate also
reproduced the two Metal batches containing the raw-ID differences. Those bounded
checks do not establish full candidate quality.

Capture all 11,334 pinned XSum inputs in the same contiguous 32-row batches as
the retained Metal baseline, using the explicitly qualified candidate CLI and a
clean reviewed checkout. Preserve raw codes and IDs with per-batch receipts and
source/input postflight checks; a partial or failed run is not a pass. Use new
report paths and permit resumption only when all recorded identities and existing
receipts still match. Compare every candidate code and raw ID against the retained
complete Metal capture. An exact match would bind the candidate to that previously
scored output; any discrepancy remains a failure to investigate, not a reason to
regenerate the baseline. Independently keep the known baseline-versus-Python
raw-ID failure visible. A candidate-to-baseline match does not waive that gate or
establish stochastic quality.

Linux focused tests passed for qualification binding, non-overwriting batch
publication, receipt validation, resume ordering and exact difference
attribution (`reports/phase4-xsum-mlx-full-candidate-tools-tests-v3.log`). The
capture writes and syncs each receipt before atomically linking it into place;
it does not claim directory-entry durability across power loss. Run a 32-row
Mac smoke and resume it unchanged before starting a new full-corpus report.
