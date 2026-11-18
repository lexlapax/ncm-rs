"""Localize entropy differences in archived LM1B-small samples by replaying their codes."""
import argparse
import ast
from bisect import bisect_right
from collections import Counter
import hashlib
from itertools import accumulate
import json
from math import isfinite, log
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import yaml
from omegaconf import OmegaConf

from lm1b_cuda_heldout_capture import CONFIGS, EVALUATOR_SHA
from lm1b_inventory import CHECKPOINTS
from lm1b_quality_compare import SEEDS, compare
from tinystories_artifact import ROOT, UPSTREAM, digest_file, json_file, verify_input

SUMMARY_SHA = '2e11e2b0aac10d152e58b888a23cda46b0df1b45285019d0ccf847e1f2f93002'
RUNTIMES = ('python', 'rust')
BATCH = 8
SAMPLES = 256


def checked_json(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'invalid archived input: {path.name}')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'archived input SHA-256 differs: {path.name}')
    return json.loads(payload)


def checked_rows(rows, count, width, maximum):
    if (type(rows) is not list or len(rows) != count
            or any(type(row) is not list or len(row) != width
                   or any(type(token) is not int or not 0 <= token < maximum for token in row)
                   for row in rows)):
        raise ValueError('invalid replay row shape or range')
    return rows


def entropy(ids):
    probabilities = [count / len(ids) for count in Counter(ids).values()]
    return -sum(p * log(p) for p in probabilities)


def rust_initial_ids(weights, seed):
    cdf = list(accumulate(weights))
    mask = (1 << 64) - 1
    for batch in range(SAMPLES // BATCH):
        state = (seed << 32) | batch
        for _ in range(BATCH):
            state = (state + 0x9e3779b97f4a7c15) & mask
            word = ((state ^ (state >> 30)) * 0xbf58476d1ce4e5b9) & mask
            word = ((word ^ (word >> 27)) * 0x94d049bb133111eb) & mask
            uniform = (((word ^ (word >> 31)) >> 32) + 0.5) / (1 << 32)
            yield bisect_right(cdf, uniform * cdf[-1])


def diff_ids(left, right):
    differences = [sum(a != b for a, b in zip(x, y, strict=True))
                   for x, y in zip(left, right, strict=True)]
    return {'rows': sum(count > 0 for count in differences), 'ids': sum(differences)}


def common_decoder_entropy(records, groups):
    results = {}
    for decoder in RUNTIMES:
        scores = {
            runtime: [dict(row, entropy=groups[f'{runtime}:{row["seed"]}']['decoded_mean_entropy'][decoder])
                      for row in records[runtime]]
            for runtime in RUNTIMES
        }
        computed = compare(scores['python'], scores['rust'])
        results[decoder] = {'difference': computed['difference']['entropy'],
                            'interval_95': computed['interval_95']['entropy'],
                            'interval_passed': computed['checks']['entropy_interval']}
    return results


def install_decoders(evaluator, scope):
    name = 'pick_tokens'
    functions = [node for node in ast.parse(evaluator.read_bytes()).body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError('unexpected upstream decoder')
    node = functions[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(evaluator), 'exec'), scope)
    # The observer receives the exact pinned FP64 scores before history is updated.
    loop = next(statement for statement in node.body if isinstance(statement, ast.For)
                and isinstance(statement.target, ast.Name) and statement.target.id == 't')
    decision = next(i for i, statement in enumerate(loop.body) if isinstance(statement, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'p' for target in statement.targets))
    loop.body.insert(decision + 1, ast.parse('record_margin(t, step)').body[0])
    node.name = 'pick_tokens_observed'
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(evaluator), 'exec'), scope)


def first_difference_margins(scope, logits, python, rust, offset):
    sites = {}
    for row, (left, right) in enumerate(zip(python, rust, strict=True)):
        for t, (expected, other) in enumerate(zip(left, right, strict=True)):
            if expected != other:
                sites.setdefault(t, []).append((row, expected, other))
                break
    if not sites:
        return []
    measured = []

    def record_margin(t, step):
        for row, expected, other in sites.get(t, ()):
            measured.append((offset + row, t, step[row, expected] - step[row, other]))

    scope['record_margin'] = record_margin
    if scope['pick_tokens_observed'](logits).tolist() != python:
        raise ValueError('score observer changes pinned decode')
    margins = [{'row': row, 'token_position': t, 'margin': float(value)}
               for row, t, value in measured]
    if len(margins) != sum(map(len, sites.values())) or any(
            not isfinite(row['margin']) or row['margin'] < 0 for row in margins):
        raise ValueError('invalid first disagreement margin')
    return margins


def upstream_pin(path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != UPSTREAM or git('status', '--porcelain'):
        raise ValueError('upstream checkout must be clean and pinned')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'artifact', 'upstream', 'source', 'rust-binary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--python-cudnn-fp32', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.output.is_symlink() or out.exists() or not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('output must be new and under reports')
    if torch.__version__ != '2.15.0+cu132' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('pinned Torch and exactly one CUDA device required')
    upstream = args.upstream.resolve()
    upstream_pin(upstream)
    summary_path = args.archive / 'summary.json'
    summary = checked_json(summary_path, SUMMARY_SHA)
    source_pins = {'small/generator.pt': CHECKPOINTS['small/generator.pt'],
                   'tokenizer.pt': CHECKPOINTS['tokenizer.pt'], 'small/config.yaml': CONFIGS['small']}
    for name, (size, sha) in source_pins.items():
        verify_input(args.source / name, size, sha)
    evaluator = upstream / 'scripts/eval_shard.py'
    verify_input(evaluator, evaluator.stat().st_size, EVALUATOR_SHA)
    expected_sha = {str(args.source / name): sha for name, (_, sha) in source_pins.items()}
    expected_sha.update({str(evaluator): EVALUATOR_SHA, str(summary_path): SUMMARY_SHA})
    watched = [Path(__file__).resolve(), args.rust_binary, evaluator, summary_path]
    watched += [args.source / name for name in source_pins]
    watched += sorted((upstream / 'models').glob('*.py'))

    torch.set_num_threads(1)
    if args.python_cudnn_fp32:
        torch.backends.cudnn.allow_tf32 = False
    generator = torch.load(args.source / 'small/generator.pt', map_location='cpu', weights_only=True, mmap=True)
    weights = generator['inference']['level0_unigram'].float().cpu()
    allowance = generator['inference']['token_unigram'].double().cuda()
    del generator

    all_codes, archived_ids, seed_reports = [], {}, {}
    for runtime in RUNTIMES:
        for seed in SEEDS:
            group = f'{runtime}:{seed}'
            pin = summary['identities'][group]
            capture_path = args.archive / f'{runtime}-seed-{seed}.json'
            score_path = args.archive / f'{runtime}-seed-{seed}-score.json'
            capture = checked_json(capture_path, pin['capture_sha256'])
            score = checked_json(score_path, pin['score_sha256'])
            expected_sha.update({str(capture_path): pin['capture_sha256'], str(score_path): pin['score_sha256']})
            watched.extend((capture_path, score_path))
            if (capture['size'], capture['seed'], capture['count'], capture['batch_size']) != ('small', seed, SAMPLES, BATCH):
                raise ValueError(f'unexpected capture metadata for {group}')
            codes = checked_rows(capture['codes'], SAMPLES, 641, 8192)
            ids = checked_rows(capture['raw_ids'], SAMPLES, 128, 30528)
            recorded = score['entropy']
            max_error = max(abs(entropy(row) - value) for row, value in zip(ids, recorded, strict=True))
            if max_error > 1e-6:
                raise ValueError(f'entropy replay differs for {group}')
            if runtime == 'python':
                stream = torch.Generator(device='cpu').manual_seed(seed)
                initial = [x for _ in range(SAMPLES // BATCH)
                           for x in torch.multinomial(weights, BATCH, replacement=True, generator=stream).tolist()]
            else:
                initial = list(rust_initial_ids(weights.tolist(), seed))
            if initial != [row[0] for row in codes]:
                raise ValueError(f'initial concept replay differs for {group}')
            all_codes.extend(codes)
            archived_ids[group] = ids
            seed_reports[group] = {
                'initial_codes_replayed': len(initial),
                'duplicate_code_rows': SAMPLES - len({tuple(row) for row in codes}),
                'duplicate_id_rows': SAMPLES - len({tuple(row) for row in ids}),
                'archived_mean_entropy': statistics.mean(recorded),
                'archived_half_entropy': [statistics.mean(recorded[i:i + SAMPLES // 2])
                                          for i in (0, SAMPLES // 2)],
                'max_entropy_recompute_error': max_error,
            }
    before = {str(path): digest_file(path) for path in watched}
    if any(before[path] != sha for path, sha in expected_sha.items()):
        raise ValueError('archived input changed during preflight')
    out.mkdir()
    codes_path = out / 'codes.json'
    json_file(codes_path, all_codes)
    codes_sha = digest_file(codes_path)
    child = subprocess.run([str(args.rust_binary.resolve()), str(args.artifact.resolve()), str(codes_path), codes_sha],
                           check=True, capture_output=True, timeout=300)
    rust_ids = checked_rows(json.loads(child.stdout), len(all_codes), 128, 30528)
    json_file(out / 'rust-ids.json', rust_ids)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from models.vqvae import VQVAE
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'small/config.yaml').read_bytes()))
    state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(state['model'], strict=True)
    codec = codec.cuda().eval()
    del state
    gen = cfg.generation
    scope = {'torch': torch, 'freq': float(gen.frequency_penalty),
             'mu': float(gen.frequency_deficit), 'ADJ': [10.0, 4.0, 0.5], 'ALLOW': allowance}
    install_decoders(evaluator, scope)
    python_ids, margins = [], []
    with torch.no_grad():
        for offset in range(0, len(all_codes), BATCH):
            codes = torch.tensor(all_codes[offset:offset + BATCH], device='cuda', dtype=torch.int64)
            _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            logits = codec.decoder(levels[-1])
            ids = scope['pick_tokens'](logits).tolist()
            python_ids.extend(ids)
            margins.extend(first_difference_margins(scope, logits, ids, rust_ids[offset:offset + BATCH], offset))
    checked_rows(python_ids, len(all_codes), 128, 30528)
    json_file(out / 'python-ids.json', python_ids)
    json_file(out / 'first-difference-margins.json', margins)
    for index, (group, retained) in enumerate(archived_ids.items()):
        start = index * SAMPLES
        python = python_ids[start:start + SAMPLES]
        rust = rust_ids[start:start + SAMPLES]
        python_entropy = [entropy(row) for row in python]
        rust_entropy = [entropy(row) for row in rust]
        seed_reports[group].update(
            decoded_mean_entropy={'python': statistics.mean(python_entropy), 'rust': statistics.mean(rust_entropy)},
            retained_vs_python=diff_ids(retained, python), retained_vs_rust=diff_ids(retained, rust),
            rust_vs_python=diff_ids(rust, python),
            decoder_entropy_difference=statistics.mean(r - p for r, p in zip(rust_entropy, python_entropy, strict=True)),
        )
    if before != {str(path): digest_file(path) for path in watched} or digest_file(codes_path) != codes_sha:
        raise ValueError('fixed-code input or source changed')
    upstream_pin(upstream)
    receipt = {'scope': 'fixed-code replay of previously observed rows', 'summary_sha256': SUMMARY_SHA,
               'codes_sha256': codes_sha, 'watched_sha256': before, 'torch': torch.__version__,
               'python_cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
               'python_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
               'outputs_sha256': {name: digest_file(out / f'{name}-ids.json') for name in RUNTIMES},
               'groups': seed_reports,
               'common_decoder_entropy': common_decoder_entropy(summary['records'], seed_reports),
               'first_difference_margins_sha256': digest_file(out / 'first-difference-margins.json')}
    json_file(out / 'result.json', receipt)
    print(json.dumps(seed_reports, sort_keys=True))


if __name__ == '__main__':
    main()
