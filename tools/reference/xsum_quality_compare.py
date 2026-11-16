"""Compare two source-bound XSum scoring results without relaxing the frozen rule."""
import argparse
import json
import math
from pathlib import Path

from xsum_artifact import digest_file

ROOT = Path(__file__).resolve().parents[2]
MODEL_REV = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
METRICS = {'rouge1', 'rouge2', 'rougeL', 'mean_words', 'generation_perplexity', 'generation_ppl_token_count'}


def compare(python, rust, scorer_sha):
    for value, origin in ((python, 'pinned-upstream-cuda'), (rust, 'rust-strict-cudnn')):
        if (value.get('schema_version') != 2 or value.get('scope') != origin + ' XSum outputs' or
            value.get('model_revision') != MODEL_REV or value.get('scorer_sha256') != scorer_sha or
            type(value.get('start')) is not int or type(value.get('stop')) is not int or
            not 0 <= value['start'] < value['stop'] <= 11334 or
            not isinstance(value.get('metrics'), dict) or set(value['metrics']) != METRICS or
            not isinstance(value.get('rows'), list) or
            [row.get('index') for row in value['rows']] != list(range(value['start'], value['stop']))):
            raise ValueError('scored output does not match the quality protocol')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in value['metrics'].values()):
            raise ValueError('non-finite quality metric')
        manifests = value.get('shard_manifests')
        if (type(manifests) is not list or len(manifests) != 2 or
            any(m.get('origin') != origin or m.get('shard') != i for i, m in enumerate(manifests))):
            raise ValueError('quality shard provenance differs')
    if (python['start'] != rust['start'] or python['stop'] != rust['stop'] or
        python['versions'] != rust['versions'] or
        python['shard_manifests'][0]['gpu_identity_and_driver'] != rust['shard_manifests'][0]['gpu_identity_and_driver']):
        raise ValueError('quality ranges, evaluator versions or generation devices differ')
    code_differences = id_differences = text_differences = rouge_differences = 0
    different_rows = []
    for p, r in zip(python['rows'], rust['rows']):
        if any(type(row.get(key)) is not list or len(row[key]) != width
               for row in (p, r) for key, width in (('codes', 200), ('raw_ids', 48))):
            raise ValueError('invalid scored output dimensions')
        codes = sum(a != b for a, b in zip(p['codes'], r['codes']))
        ids = sum(a != b for a, b in zip(p['raw_ids'], r['raw_ids']))
        text = p['hypothesis'] != r['hypothesis']
        rouge = p['rouge'] != r['rouge']
        code_differences += codes; id_differences += ids
        text_differences += text; rouge_differences += rouge
        if (codes or ids or text or rouge) and len(different_rows) < 16:
            different_rows.append({'index': p['index'], 'codes': codes, 'ids': ids,
                                   'text_differs': text, 'rouge_differs': rouge})
    metrics_equal = python['metrics'] == rust['metrics']
    return {'start': python['start'], 'stop': python['stop'], 'rows': len(python['rows']),
            'code_differences': code_differences, 'id_differences': id_differences,
            'text_differences': text_differences, 'rouge_row_differences': rouge_differences,
            'metrics_equal': metrics_equal, 'python_metrics': python['metrics'], 'rust_metrics': rust['metrics'],
            'first_different_rows': different_rows,
            'quality_rule_passed': id_differences == text_differences == rouge_differences == 0 and metrics_equal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--rust', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be under reports')
    for path in (args.python, args.rust):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to((ROOT / 'reports').resolve()):
            raise ValueError('scored inputs must be regular files under reports')
    python = json.loads(args.python.read_bytes()); rust = json.loads(args.rust.read_bytes())
    result = compare(python, rust, digest_file(ROOT / 'tools/reference/xsum_quality_score.py'))
    result.update({'schema_version': 1, 'python_score_sha256': digest_file(args.python),
                   'rust_score_sha256': digest_file(args.rust), 'comparator_sha256': digest_file(Path(__file__))})
    with args.output.open('x') as output:
        json.dump(result, output, sort_keys=True, indent=2); output.write('\n')
    print(json.dumps(result, sort_keys=True))
    if not result['quality_rule_passed']: raise SystemExit(1)


if __name__ == '__main__': main()
