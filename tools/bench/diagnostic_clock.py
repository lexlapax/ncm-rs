"""Explicit process-local lifetime clocks for diagnostic evidence, not provenance.

UTC reads are bracketed by CLOCK_MONOTONIC reads. No calibration is guessed from
elapsed durations; callers retain these envelopes even on failed collections.
"""
from datetime import datetime, timezone
import os
import time


def stamp():
    before = time.monotonic_ns()
    utc = datetime.now(timezone.utc).isoformat()
    after = time.monotonic_ns()
    return {'monotonic_before_ns': before, 'utc': utc, 'monotonic_after_ns': after}


def begin():
    return {'pid': os.getpid(), 'parent_pid': os.getppid(), 'start': stamp()}


def finish(started):
    # Copy rather than mutate the start record held by a caller.
    return {**started, 'end': stamp()}
