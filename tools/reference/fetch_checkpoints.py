"""Fetch one explicitly pinned checkpoint profile for offline export.

Requires the reference environment because it imports the existing exporter
pins. Files are streamed to a new directory with a caller-specified byte
budget. On failure the partial directory is retained for inspection. Parent
and destination directories must be trusted and not concurrently mutated.
"""
import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path, PurePosixPath

REPOSITORY = 'hazyresearch/ncm-checkpoints'
REVISION = 'd342ce8aaea58830eef3b8ad473664673e41f692'
REVISION_15K_EMA = 'e34c3fab451ddbe7cf474cc97c44ab23321f2219'
PROFILES = ('sudoku-easy', 'sudoku-medium', 'sudoku-hard', 'tinystories', 'tinystories-15k-ema',
            'xsum', 'wmt14', 'lm1b-small', 'lm1b-medium', 'lm1b-large')
BLOCK = 1024 * 1024


def profile_revision(profile):
    return REVISION_15K_EMA if profile == 'tinystories-15k-ema' else REVISION


def specs(profile):
    # Imports are deliberately offline and reuse the exporters' existing pins.
    import sudoku_artifact as sudoku
    import tinystories_artifact as tiny
    import xsum_artifact as xsum
    import wmt14_inventory as wmt14
    import lm1b_inventory as lm1b
    import lm1b_export_preparation as lm1b_export

    if (sudoku.REVISION != REVISION or tiny.REVISION != REVISION
            or tiny.REVISION_15K_EMA != REVISION_15K_EMA):
        raise ValueError('checkpoint revision differs from the reviewed pin')
    if profile.startswith('sudoku-'):
        difficulty = profile.removeprefix('sudoku-')
        hashes = sudoku.INPUTS if difficulty == 'easy' else sudoku.PHASE3_INPUTS[difficulty]
        sizes = sudoku.INPUT_SIZES if difficulty == 'easy' else sudoku.PHASE3_SIZES[difficulty]
        files = {name: (sizes[name], sha) for name, sha in hashes.items()}
        prefix = f'sudoku/{difficulty}'
    elif profile == 'tinystories-15k-ema':
        files = tiny.INPUTS_15K_EMA
        prefix = 'tinystories'
    elif profile in ('tinystories', 'xsum', 'wmt14'):
        files = {'tinystories': tiny.INPUTS, 'xsum': xsum.INPUTS, 'wmt14': wmt14.INPUTS}[profile]
        prefix = profile
    elif profile.startswith('lm1b-'):
        size = profile.removeprefix('lm1b-')
        files = {f'{size}/generator.pt': lm1b.CHECKPOINTS[f'{size}/generator.pt'],
                 'tokenizer.pt': lm1b.CHECKPOINTS['tokenizer.pt'],
                 f'{size}/config.yaml': lm1b_export.CONFIGS[size]}
        prefix = 'lm1b'
    else:
        raise ValueError('unsupported profile')
    result = []
    for name, (size, sha) in sorted(files.items()):
        path = PurePosixPath(name)
        if (path.is_absolute() or not path.parts or any(part in ('', '.', '..') for part in path.parts)
                or str(path) != name or type(size) is not int or size <= 0
                or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)):
            raise ValueError('invalid exporter pin')
        result.append({'remote': f'{prefix}/{name}', 'output': name, 'bytes': size, 'sha256': sha})
    if not result:
        raise ValueError('empty checkpoint profile')
    return result


def write_verified(response, destination, expected_size, expected_hash):
    if not response.geturl().startswith('https://'):
        raise ValueError('checkpoint redirect is not HTTPS')
    count = 0
    digest = hashlib.sha256()
    with destination.open('xb') as target:
        while True:
            chunk = response.read(min(BLOCK, expected_size - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > expected_size:
                raise ValueError('checkpoint exceeds pinned byte count')
            digest.update(chunk)
            target.write(chunk)
    if count != expected_size or digest.hexdigest() != expected_hash:
        raise ValueError('checkpoint size or SHA-256 differs')


def fetch(output, records, budget, revision=REVISION):
    if output.exists() or output.is_symlink():
        raise ValueError('output already exists')
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError('output parent must be a real existing directory')
    total = sum(record['bytes'] for record in records)
    if type(budget) is not int or budget < total:
        raise ValueError(f'download budget must be at least {total} bytes')
    if shutil.disk_usage(parent).free < total:
        raise ValueError(f'insufficient free disk space for {total} bytes')
    output.mkdir(exist_ok=False)
    for record in records:
        target = output.joinpath(*PurePosixPath(record['output']).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{record['remote']}"
        with urllib.request.urlopen(url, timeout=90) as response:
            write_verified(response, target, record['bytes'], record['sha256'])
        print(record['remote'], record['bytes'], record['sha256'], flush=True)


def verify(output, records):
    if output.is_symlink() or not output.is_dir():
        raise ValueError('source must be a real directory')
    for record in records:
        path = output
        for part in PurePosixPath(record['output']).parts:
            path = path / part
            if path.is_symlink():
                raise ValueError('source contains a symlink')
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f"source type or size differs: {record['output']}")
        digest = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            while chunk := stream.read(BLOCK):
                count += len(chunk)
                if count > record['bytes']:
                    raise ValueError('source grew beyond pinned size')
                digest.update(chunk)
        if count != record['bytes'] or digest.hexdigest() != record['sha256']:
            raise ValueError(f"source SHA-256 differs: {record['output']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new destination, or existing source for --verify')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='print pins and total; do not create or download')
    mode.add_argument('--verify', action='store_true', help='verify an existing source directory without downloading')
    parser.add_argument('--budget-bytes', type=int, help='required maximum total bytes for download')
    args = parser.parse_args()
    records = specs(args.profile)
    revision = profile_revision(args.profile)
    total = sum(record['bytes'] for record in records)
    if args.plan:
        print(json.dumps({'profile': args.profile, 'repository': REPOSITORY, 'revision': revision,
                          'total_bytes': total, 'files': records}, indent=2, sort_keys=True))
    elif args.verify:
        verify(args.output, records)
        print(json.dumps({'profile': args.profile, 'verified_files': len(records), 'total_bytes': total}, sort_keys=True))
    else:
        fetch(args.output, records, args.budget_bytes, revision=revision)
        verify(args.output, records)
        print(json.dumps({'profile': args.profile, 'verified_files': len(records), 'total_bytes': total}, sort_keys=True))


if __name__ == '__main__':
    main()
