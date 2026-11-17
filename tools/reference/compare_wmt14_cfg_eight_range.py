"""Compare a bounded internal-eight diagnostic range with the full pinned Python capture."""
import argparse
import json
from pathlib import Path
from compare_wmt14_cfg_microbatch_capture import compare
from tinystories_artifact import digest_file, json_file

REFERENCE = '78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--candidate-sha256', required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or digest_file(args.reference) != REFERENCE or digest_file(args.candidate) != args.candidate_sha256:
        raise ValueError('output or input identity differs')
    left, right = json.loads(args.reference.read_bytes()), json.loads(args.candidate.read_bytes())
    start, count = right['start'], right['count']
    if (left['start'], left['count'], left['batch_size'], left['cfg_scale'], left['cfg_null_eos']) != (0, 3003, 4, 1.0, 151645):
        raise ValueError('unexpected reference metadata')
    if (right['batch_size'], right['internal_batch_size'], right['cfg_scale'], right['cfg_null_eos']) != (16, 8, 1.0, 151645):
        raise ValueError('unexpected diagnostic metadata')
    result = {'scope': 'bounded WMT14 CFG comparison; Python batch four, diagnostic internal eight',
              'start': start, 'count': count, 'reference_sha256': REFERENCE, 'candidate_sha256': args.candidate_sha256,
              **compare(left['outputs'][start:start + count], right['outputs'], start, count)}
    if digest_file(args.reference) != REFERENCE or digest_file(args.candidate) != args.candidate_sha256:
        raise ValueError('input changed during comparison')
    json_file(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if result['code_differences'] or result['id_differences']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
