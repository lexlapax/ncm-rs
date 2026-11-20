"""Recompute published Phase 5 cold aggregates from sanitized timing/RSS trials."""
import argparse
import json
from pathlib import Path

import audit_phase5_cold_aggregates as audit
import export_phase5_cold_trials as export

TRIALS_SHA = '067d3ed9021045a30d8d87348917fa7e82c28144035841e65a290eba546788d9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--trials', type=Path, required=True)
    args = parser.parse_args()
    aggregate = audit.read_regular(args.aggregate)
    trials = audit.read_regular(args.trials)
    if audit.sha(aggregate) != export.AGGREGATE_SHA or audit.sha(trials) != TRIALS_SHA:
        raise ValueError('published data identity differs')
    export.verify_export(json.loads(trials), json.loads(aggregate))
    if audit.read_regular(args.aggregate) != aggregate or audit.read_regular(args.trials) != trials:
        raise ValueError('input changed during verification')
    print('Seven profiles: 90 measured timing/RSS pairs per condition reproduced; no GPU or quality qualification implied.')


if __name__ == '__main__':
    main()
