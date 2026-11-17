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
complete Metal capture. An exact match would bind the candidate to that retained
output; any discrepancy remains a failure to investigate, not a reason to
regenerate the baseline. Independently keep the known baseline-versus-Python
raw-ID failure visible. A candidate-to-baseline match does not waive that gate or
establish stochastic quality.

Linux focused tests passed for qualification binding, non-overwriting batch
publication, receipt validation, resume ordering and exact difference
attribution (`reports/phase4-xsum-mlx-full-candidate-tools-tests-v3.log`). The
capture writes and syncs each receipt before atomically linking it into place;
it does not claim directory-entry durability across power loss. Run a 32-row
Mac smoke and resume it unchanged before starting a new full-corpus report.

On clean Mac commit `5a62960eeefbc6a873c79004746040e3aaacea25`, the 32-row
smoke completed, matched the retained Metal baseline, and a second invocation
validated the same receipts without regenerating them. The smoke manifest hashes
to `7ebe4aa5ac154068651c7b5171ba790bed7d1ad8d05f5e2581ee3e0b4ea44788`.
The full capture is running with `.reference/mlx-venv/bin/python
tools/bench/capture_xsum_mlx_candidate.py --head
5a62960eeefbc6a873c79004746040e3aaacea25 --binary-sha256
a580ba2ca6c8a2e1a43d4e7a0b3a7b6645b6576f7424ec2ea3adb711a2429414
--qualification reports/phase4-xsum-borrowed-upload-mlx-cli-v3/result.json
--qualification-sha256 30415fa65a9346989ea8b4924a27d565bfd448ece82c310cb09adc148d7f3d12
--output reports/phase4-xsum-borrowed-upload-mlx-full-v1`. The first 3,136
completed rows matched the baseline in an interim receipt-checked comparison;
this is not a complete result.

The full capture subsequently completed with input/source postflight. Its
manifest SHA-256 is
`4b27beee855cc27738b62f1c5e6fa5dc2c68f1aeaff0788bf254569f711ece6e`;
all files copied to `reports/phase4-xsum-borrowed-upload-mlx-full-captured-v1/`
matched Mac hashes. The receipt-validated command `setpriv --inh-caps=-all env
OMP_NUM_THREADS=1 .reference/venv/bin/python tools/bench/compare_xsum_mlx_candidate.py
--candidate reports/phase4-xsum-borrowed-upload-mlx-full-captured-v1
--candidate-manifest-sha256 4b27beee855cc27738b62f1c5e6fa5dc2c68f1aeaff0788bf254569f711ece6e
--head 5a62960eeefbc6a873c79004746040e3aaacea25 --binary-sha256
a580ba2ca6c8a2e1a43d4e7a0b3a7b6645b6576f7424ec2ea3adb711a2429414 --output
reports/phase4-xsum-borrowed-upload-mlx-full-comparison-v1.json` found zero code
and zero raw-ID differences across all 11,334 rows. Its result SHA-256 is
`c92c7367d40eb48a0304676b2f0a4f4003061e0c80a6e875d5b9db5294d99224`.

The candidate therefore has the exact retained Metal top-1 outputs; that
baseline's EOS-filtered decoder inputs match pinned Python on all rows, but no
fresh Metal scorer was run. Its six post-EOS raw-ID differences versus Python
still fail the frozen exact gate. This comparison does not qualify stochastic
candidate output.
