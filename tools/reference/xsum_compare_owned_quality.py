"""Compare complete owned-loader CUDA captures with retained XSum quality rows."""
import argparse
import json
from pathlib import Path

from xsum_quality_rust import DATA_SHA, ROPE_ARTIFACT_SHA, read_batch, shard_batches
from xsum_artifact import digest_file

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'reports/phase4-xsum-reference-tie-full-v1-s{}'
MANIFEST_SHA = ('ae580e8ac25b59c2b782f3aeade72daf24ffbe8f4bd637d882f671607bfb2240',
                '393be26641af3531fef4d14d5129a314e090c52fbe92a9cb2202d9edf448bd2f')


def compare_rows(reference, candidate):
    if reference['indices'] != candidate['indices']:
        raise ValueError('batch indices differ')
    result = {'codes': 0, 'ids': 0, 'code_rows': [], 'id_rows': []}
    for index, a, b, c, d in zip(reference['indices'], reference['codes'], candidate['codes'],
                                 reference['raw_ids'], candidate['raw_ids']):
        code_diff = sum(x != y for x, y in zip(a, b))
        id_diff = sum(x != y for x, y in zip(c, d))
        result['codes'] += code_diff
        result['ids'] += id_diff
        if code_diff:
            result['code_rows'].append(index)
        if id_diff:
            result['id_rows'].append(index)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard0', type=Path, required=True)
    parser.add_argument('--shard1', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = {'rows': 0, 'codes': 0, 'ids': 0, 'code_rows': [], 'id_rows': [],
              'candidate_manifest_sha256': [], 'reference_manifest_sha256': list(MANIFEST_SHA),
              'scope': 'exact codes and IDs against previously scored CUDA capture; source/postflight logs remain separate'}
    candidate_sources = None
    for shard, candidate in enumerate((args.shard0, args.shard1)):
        reference = Path(str(REFERENCE).format(shard))
        if digest_file(reference / 'manifest.json') != MANIFEST_SHA[shard]:
            raise ValueError('reference manifest changed')
        manifest_path = candidate / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        expected = {'artifact_profile': 'v2', 'artifact_sha256': ROPE_ARTIFACT_SHA, 'batch_size': 32,
                    'cfg': 0.0, 'data_sha256': DATA_SHA, 'origin': 'rust-strict-cudnn',
                    'schema_version': 1, 'shard': shard, 'shards': 2,
                    'temperature': 1.0, 'top_k': 1, 'top_p': 0.0}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('candidate manifest differs')
        sources = manifest.get('binary_and_source_sha256')
        if not isinstance(sources, dict) or (candidate_sources is not None and sources != candidate_sources):
            raise ValueError('candidate sources differ between shards')
        candidate_sources = sources
        result['candidate_manifest_sha256'].append(digest_file(manifest_path))
        batches, selected = shard_batches(shard, 11334)
        if len(list(candidate.glob('batch-*.json'))) != len(selected):
            raise ValueError('candidate capture is incomplete')
        for ordinal, indices in selected:
            baseline = read_batch(reference / f'batch-{ordinal:05}.json', batches[ordinal], ordinal)
            actual = read_batch(candidate / f'batch-{ordinal:05}.json', batches[ordinal], ordinal)
            differences = compare_rows(baseline, actual)
            result['rows'] += len(indices)
            for key in ('codes', 'ids'):
                result[key] += differences[key]
            for key in ('code_rows', 'id_rows'):
                result[key].extend(differences[key])
    if result['rows'] != 11334:
        raise ValueError('incorrect corpus coverage')
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write('\n')
    print(json.dumps(result, sort_keys=True))
    if result['codes'] or result['ids']:
        raise SystemExit('candidate and retained outputs differ; see the retained comparison')


if __name__ == '__main__':
    main()
