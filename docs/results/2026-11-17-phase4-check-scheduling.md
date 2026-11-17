# Phase 4 Linux check scheduling

The Linux report-test portion of `tools/check.sh` now overlaps exactly three
independent existing cases after running every other discovered case serially:

- `test_clock_restoration_capture...test_cancelled_supervisor_finishes_twelve_second_cleanup_retry_and_large_output` uses a private temporary directory and child processes to exercise two six-second fake cleanup attempts.
- `test_qualify_clock_codec...test_all_eighteen_archived_outputs_and_rehashed_negatives_in_fresh_child` checks read-only archived data in its own child process.
- `test_finalize_clock_diagnostic...test_failed_gates_preserve_partial_evidence_without_rehash_or_completion` checks synthetic clock and policy evidence in its own child and temporary directory.

The scheduler fails closed if a selected case is absent or duplicated, records
every started test identity, compares the full discovered/executed multisets,
and fails if any test fails or is omitted. Four focused tests cover partition
coverage, missing/duplicate cases, isolated failure reporting and whole-run
failure propagation. macOS continues using its existing portable selection;
other platforms retain the original serial discovery. No test or timeout was
removed.

A sequential profile ran all 520 pre-existing Linux report tests successfully
in 167.35 seconds. An initial two-case-overlap trial ran 523/523 tests in
155.37 seconds, and two full `tools/check.sh --cuda` runs passed all 524/524
report tests and 18 real-CUDA operator cases. The two isolated cases passed
three additional overlapping repetitions. A third independently isolated case
was then added; three more overlapping repetitions passed, followed by the full
CUDA check with 524/524 report tests and 18 real-CUDA operator cases. The final
report suite took approximately 146.48 seconds (134.09 seconds serial plus the
12.39-second longest overlapping case); the complete CUDA check took 152.86
seconds. Comparing final test identities to the baseline found all 520 prior
cases exactly once plus the four scheduler tests. The single sequential and
final suite timings indicate about 21 seconds saved on this machine; they are
not a controlled performance benchmark and will vary with system load.

Evidence is retained with the following SHA-256 identities:

- `reports/profile-phase4-bench-tests-v1.json`: `99d674be650a0d084eb1eace52850cff87be0bd55b03f7f8c46350d797c08a91`
- `reports/phase4-parallel-report-tests-v1.log`: `11fe48528fb771416f68ae9d7bb31df66696911d2f58eef8ef71894d18f3ecec`
- `reports/phase4-parallel-report-pair-repeat-v1.log`: `496956e52001981398e9bc4345bbd8d564444d6fe1cc838bf918e1b398c292dd`
- `reports/phase4-parallel-check-cuda-v1.log`: `459403c6430811e126822c525905aa845c0faf57e262adfe64b6edd7ed0b3895`
- `reports/phase4-parallel-check-cuda-v2.log`: `e60d4b2e83881d25c9b6b9c61eb44e15963cfe8e36de48a5529cf2490b237f49`
- `reports/phase4-parallel-report-triplet-v1.log`: `a164a262d0c3719986a1ccf6f9c2c81395681c5db86c23462e949df975da3bd2`
- `reports/phase4-parallel-check-cuda-v3.log`: `7f176b537c32edaa30cfb997a314f9895120dd4ece823476686e829882ec4f3c`

This is a validation-workflow optimization, not model correctness, quality,
performance, or a Phase 4 gate pass. The XSum Metal and LM1B held-out failures
remain unchanged.
