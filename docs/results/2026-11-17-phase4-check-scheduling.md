# Phase 4 Linux check scheduling

The Linux report-test portion of `tools/check.sh` now overlaps exactly two
independent existing cases after running every other discovered case serially:

- `test_clock_restoration_capture...test_cancelled_supervisor_finishes_twelve_second_cleanup_retry_and_large_output` uses a private temporary directory and child processes to exercise two six-second fake cleanup attempts.
- `test_qualify_clock_codec...test_all_eighteen_archived_outputs_and_rehashed_negatives_in_fresh_child` checks read-only archived data in its own child process.

The scheduler fails closed if either selected case is absent or duplicated,
records every started test identity, compares the full discovered/executed
multisets, and fails if any test fails or is omitted. Four focused tests cover
partition coverage, missing/duplicate cases, isolated failure reporting and
whole-run failure propagation. macOS continues using its existing portable
selection and does not use this scheduler; other platforms retain the original
serial discovery. No test or timeout was removed.

A sequential profile ran all 520 pre-existing Linux report tests successfully
in 167.35 seconds. A full trial with the first three scheduler tests ran 523/523
in 155.37 seconds; the subsequent required `tools/check.sh --cuda` ran all
524/524 report cases after the fourth scheduler test was added and passed 18
real-CUDA operator cases. Comparing its test identities to the baseline found
all 520 pre-existing cases exactly once plus the four new scheduler tests. The
two isolated cases also passed three additional overlapping repetitions. A
second full CUDA check after preserving the non-Linux shell branch also passed
524/524 report cases and 18 device cases. The
single sequential and overlapping suite timings indicate about 12 seconds
saved on this machine; they are not a controlled performance benchmark and
will vary with system load. The full CUDA check took 161.51 seconds in the
recorded run, including its other checks; the final branch check took 161.53 seconds.

Evidence is retained in `reports/profile-phase4-bench-tests-v1.json`,
`reports/phase4-parallel-report-tests-v1.log`,
`reports/phase4-parallel-report-pair-repeat-v1.log`, and
`reports/phase4-parallel-check-cuda-v1.log`, and
`reports/phase4-parallel-check-cuda-v2.log`, with SHA-256 values respectively:

- `99d674be650a0d084eb1eace52850cff87be0bd55b03f7f8c46350d797c08a91`
- `11fe48528fb771416f68ae9d7bb31df66696911d2f58eef8ef71894d18f3ecec`
- `496956e52001981398e9bc4345bbd8d564444d6fe1cc838bf918e1b398c292dd`
- `459403c6430811e126822c525905aa845c0faf57e262adfe64b6edd7ed0b3895`
- `e60d4b2e83881d25c9b6b9c61eb44e15963cfe8e36de48a5529cf2490b237f49`

This is a validation-workflow optimization, not model correctness, quality,
performance, or a Phase 4 gate pass. The XSum Metal and LM1B held-out failures
remain unchanged.
