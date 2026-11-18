"""Replay frozen WMT14 quality from a complete batchwise CUDA or Metal capture."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from compare_wmt14_corpus_capture import BATCH, REFERENCE_SHA256, ROWS, compare_batches

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_TEXT_SHA256 = {
    0: '6b8abb2a97c29005489491628a6891f8090be44f7889d0cc6e9118400fcab783',
    1: '1dc421ee6bdd115e050c7d85069b7f7076c177881d35752181b0ee45ca7bf1df',
}
PROTOCOL_SHA256 = '26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1'
SCORERS = ('wmt14-scorer-venv', 'wmt14-scorer-replay-venv')
CUDA_SCOPE = 'WMT14 CUDA frozen external grouping'
METAL_SCOPE = 'WMT14 Metal explicit internal retry, frozen external grouping'


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value, compact=False):
    options = {'separators': (',', ':')} if compact else {'indent': 2}
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, **options)
        stream.write('\n')


def write_receipts(capture_bytes, reference, cfg, runtime, out):
    batches = [json.loads(line) for line in capture_bytes.splitlines()]
    ref_sha = REFERENCE_SHA256[cfg]
    check = compare_batches(reference['outputs'], batches, 0, ROWS)
    batch = {'cfg': cfg, 'start': 0, 'count': ROWS, 'reference_sha256': ref_sha,
             'capture_sha256': digest(capture_bytes), **check}
    write_json(out / 'batch-comparison.json', batch)
    if batch['rejected_batches']:
        return None

    rows = [{'row': row, 'codes': codes, 'raw_ids': ids}
            for record in batches
            for row, (codes, ids) in enumerate(zip(record['codes'], record['raw_ids'], strict=True), record['start'])]
    doc = {'scope': METAL_SCOPE if runtime == 'metal' else CUDA_SCOPE,
           'start': 0, 'count': ROWS, 'batch_size': BATCH, 'outputs': rows}
    comparison = {
        'scope': ('bounded CFG=1 CUDA diagnostic comparison; scoring and performance are separate' if cfg else
                  'bounded CUDA output comparison; scoring and performance are separate'),
        'start': 0, 'count': ROWS, 'reference_sha256': ref_sha,
        **{key: batch[key] for key in ('code_differences', 'id_differences', 'code_rows', 'id_rows')},
    }
    if cfg:
        doc.update(cfg_scale=1.0, cfg_null_eos=151645)
        comparison['cfg_scale'] = 1.0
    candidate = out / 'capture.json'
    write_json(candidate, doc, compact=True)
    comparison['candidate_sha256'] = digest(candidate.read_bytes())
    write_json(out / 'comparison.json', comparison)
    return comparison['candidate_sha256']


def replay(capture, reference_path, reference_text, cfg, runtime, out):
    ref_bytes = reference_path.read_bytes()
    text_bytes = reference_text.read_bytes()
    if digest(ref_bytes) != REFERENCE_SHA256[cfg] or digest(text_bytes) != REFERENCE_TEXT_SHA256[cfg]:
        raise ValueError('frozen WMT14 reference or decoded reference identity differs')
    capture_bytes = capture.read_bytes()
    out.mkdir()
    candidate_sha = write_receipts(capture_bytes, json.loads(ref_bytes), cfg, runtime, out)
    if candidate_sha is None:
        print('WMT14 replay contains rejected batches. See batch-comparison.json.')
        return 1

    options = {'cwd': ROOT, 'env': dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')}
    candidate_text = out / 'decoded.json'
    comparison = out / 'comparison.json'
    subprocess.run([ROOT / '.reference/tokenizers-venv/bin/python', ROOT / 'tools/reference/wmt14_decode_capture.py',
                    out / 'capture.json', candidate_text, '--capture-sha256', candidate_sha], check=True, **options)
    score_args = [reference_text, candidate_text, comparison]
    pinned = ['--reference-text-sha256', REFERENCE_TEXT_SHA256[cfg],
              '--candidate-text-sha256', digest(candidate_text.read_bytes()),
              '--comparison-sha256', digest(comparison.read_bytes()),
              '--protocol-sha256', PROTOCOL_SHA256, '--mode', 'cfg1' if cfg else 'base']
    scores = []
    for name in SCORERS:
        receipt = out / f'{name}.json'
        child = subprocess.run([ROOT / f'.reference/{name}/bin/python',
                                ROOT / 'tools/reference/wmt14_score_heldout.py',
                                *score_args, receipt, *pinned], **options)
        if child.returncode not in (0, 1) or not receipt.is_file():
            child.check_returncode()
            raise ValueError('WMT14 scorer did not produce a receipt')
        payload = receipt.read_bytes()
        report = json.loads(payload)
        passed = report['full']['all_passed'] is True and report['heldout']['all_passed'] is True
        if child.returncode != (0 if passed else 1):
            raise ValueError('WMT14 scorer status disagrees with its receipt')
        scores.append((passed, payload))
    if scores[0] != scores[1]:
        raise ValueError('independent WMT14 scorer receipts differ')
    if any(path.read_bytes() != data for path, data in ((capture, capture_bytes), (reference_path, ref_bytes),
                                                        (reference_text, text_bytes))):
        raise ValueError('WMT14 replay input changed')
    return 0 if scores[0][0] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cfg', type=int, choices=REFERENCE_SHA256, required=True)
    parser.add_argument('--runtime', choices=('cuda', 'metal'), required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-text', type=Path)
    args = parser.parse_args()
    suffix = 'cfg' if args.cfg else 'base'
    reference = args.reference or ROOT / f'reports/phase4-wmt14-{suffix}-cuda-full-v1/python.json'
    reference_text = args.reference_text or ROOT / f'reports/phase4-wmt14-{suffix}-cuda-full-python-text-v1.json'
    raise SystemExit(replay(args.capture.resolve(), reference.resolve(), reference_text.resolve(),
                            args.cfg, args.runtime, args.output.absolute()))


if __name__ == '__main__':
    main()
