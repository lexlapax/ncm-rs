# CUDA original-selection capped-buffer diagnostic

The direct 2.5 MiB heuristic limit changed algorithm selection and had mixed performance. In this follow-up, select with the existing 4 MiB preference, verify that the chosen algorithm reports a workspace requirement at most 2.5 MiB, and launch it with that smaller supplied size. Compare against an independently selected 4 MiB baseline for the same 12 shapes, in three fresh processes with the same 5 warmups and 30 interleaved events. Fail if any selected requirement exceeds the cap; a production implementation would need a general fallback for that case. Validate all output elements against the baseline with the existing diagnostic tolerance, and retain activity and results. This diagnostic still allocates the larger comparison buffer and does not measure native pool use. No startup or gate claim.

Apply the adjacent patch to the zero-workspace source and compile with the same compiler and flags as the capped diagnostic.
