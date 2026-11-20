"""Manifest-bound LM1B quality campaign: rehearsal, prospective or nonblind frozen-seed retest.

Build the Rust runner first with
`cargo build --locked --release -p ncm-backend-cuda --features lm1b-experimental,cuda-cudnn
--example lm1b_small_cohort_campaign`. The manifest pins its binary and source hashes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

from compare import RUNTIMES, SEEDS, compare_cohorts, digest
from tinystories_artifact import ROOT, UPSTREAM, verify_input

HERE = Path(__file__).resolve().parent
RUNNER = ROOT / 'target/release/examples/lm1b_small_cohort_campaign'
RUNNER_SOURCE = ROOT / 'crates/ncm-backend-cuda/examples/lm1b_small_cohort_campaign.rs'
METAL_RUNNER_SOURCE = ROOT / 'crates/ncm-backend-mlx/examples/lm1b_small_cohort_campaign.rs'
UPSTREAM_DIR = ROOT / '.reference/upstream-ncm'
PY_SOURCE = ROOT / 'artifacts/upstream/lm1b-phase4'
BERT = ROOT / 'artifacts/upstream/tokenizers-v1/bert'
BERT_MANIFEST = ROOT / 'artifacts/canonical/tokenizers-qualified/bert/manifest.json'
TOKENIZER_LOCK = ROOT / 'tools/reference/tokenizers-v1.lock.json'
MODEL = ROOT / 'artifacts/upstream/gpt2-large-phase4'
ROPE = ROOT / 'reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors'
CUDNN = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib'
SHARD = UPSTREAM_DIR / 'scripts/eval_shard.py'
EVALUATOR = UPSTREAM_DIR / 'scripts/eval_merge.py'
ROPE_SHA = '9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc'
SHARD_SHA = 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a'
EVAL_SHA = '6452cb5cc06a2355ab7cd16b9809b7144a557418e9041e4617f3c50aef5655fc'
TOKENIZER_SHA = 'd88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c'
RETEST_REGISTRY_SHA = 'c9800090de18f18d6719993f1ac36057b6cea479a26cc2e4b3115de6e4beeab1'
# Per size: exported artifact manifest, and pinned upstream generator and config hashes.
SIZES = {
    'small': ('58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76',
              'b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe',
              '4db04299a87f7eb1627de0abb2f0b257e50e5ea0138edda771ad287cdabd7edf'),
    'medium': ('2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c',
               '5aa4621472c2e658ab343e50d98abc808e4c2c6dd26eaf8535897b6a540c1604',
               '93a42e0923b38f06f186072fdc37c1a2b4db5298c7577755390dd36c651d8142'),
    'large': ('86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a',
              'ec34d6c56a81f428303b2fa3a8e1615ceee81b7dec7f59380427f46808ec224c',
              '4410e3710360cae162f8be4186a44a0c7c473be674a1ac3d1d940bfe45d5e70c'),
}


def artifact(size):
    return ROOT / f'artifacts/canonical/lm1b-{size}-inference-v1'


def manifest_sha(size):
    return SIZES[size][0]


def py_weights(size):
    _, generator, config = SIZES[size]
    return {f'{size}/generator.pt': generator, 'tokenizer.pt': TOKENIZER_SHA, f'{size}/config.yaml': config}


def plan_size(plan):
    # Plans written before sizes were supported are small campaigns.
    return plan.get('size', 'small')
PARAMS = {'batch_size': 8, 'samples_per_seed': 256, 'entropy_margin': 0.0046,
          'ppl_margin': 5.46, 'bootstrap_draws': 10_000, 'bootstrap_seed': 42}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def payload(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def atomic_publish(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.receiving-') as temporary:
        temporary.write(payload(value))
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, path)


def json_file(path, sha=None):
    require(path.is_file() and not path.is_symlink(), f'expected a regular record at {path}')
    data = path.read_bytes()
    calculated = hashlib.sha256(data).hexdigest()
    require(sha is None or sha == calculated, f'record changed at {path}')
    return json.loads(data), calculated


def seed_lists(mode, seeds):
    require(mode in ('rehearsal', 'prospective', 'retest'), 'unknown campaign mode')
    require(isinstance(seeds, dict) and set(seeds) == set(RUNTIMES), 'two runtime seed lists required')
    expected = 2 if mode == 'rehearsal' else 80
    for runtime, values in seeds.items():
        require(isinstance(values, list) and len(values) == expected
                and all(type(x) is int and 0 <= x < 2**32 for x in values)
                and len(set(values)) == expected, f'invalid seed list for {runtime}')
    if mode == 'rehearsal':
        require(all(x in SEEDS for values in seeds.values() for x in values), 'rehearsal requires observed seeds')
    else:
        a, b = set(seeds['python']), set(seeds['rust'])
        require(a.isdisjoint(b) and not (a | b) & (set(range(17)) | {42}),
                '80-seed cohort requires disjoint lists outside the original five-seed range')
        if mode == 'retest':
            require(seeds == {'python': list(range(1000, 1080)), 'rust': list(range(2000, 2080))},
                    'CUDA retest requires the original frozen seed lists')
    return seeds


def checkout_identity():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(UPSTREAM_DIR), *args], text=True).strip()
    require(git('rev-parse', 'HEAD') == UPSTREAM and not git('status', '--porcelain'),
            'upstream checkout differs from the pinned clean source')


def pins(size='small'):
    checkout_identity()
    fixed = {artifact(size) / 'manifest.json': manifest_sha(size), ROPE: ROPE_SHA, SHARD: SHARD_SHA, EVALUATOR: EVAL_SHA}
    files = [Path(__file__).resolve(), RUNNER, HERE / 'python_capture.py',
             RUNNER_SOURCE, METAL_RUNNER_SOURCE, HERE / 'compare.py',
             HERE / 'lm1b_shared_forward_score.py', BERT_MANIFEST, TOKENIZER_LOCK,
             ROOT / 'Cargo.lock', *(ROOT / 'tools/reference' / name for name in (
                 'lm1b_heldout_score.py', 'lm1b_quality_compare.py', 'tinystories_artifact.py',
                 'tinystories_quality.py', 'lm1b_cuda_heldout_capture.py', 'lm1b_inventory.py'))]
    fixed.update({path: digest(path) for path in files})
    for path, sha in fixed.items():
        require(digest(path) == sha, f'input pin differs for {path}')
    require(digest(BERT_MANIFEST) == json_file(TOKENIZER_LOCK)[0]['profiles']['bert']['manifest_sha256'],
            'pinned BERT manifest differs')
    return {str(path): sha for path, sha in fixed.items()}


def manifest(path, sha):
    plan, _ = json_file(path, sha)
    require(plan['schema_version'] == 1 and plan['params'] == PARAMS, 'unknown campaign parameters')
    seed_lists(plan['mode'], plan['seeds'])
    if plan['mode'] == 'retest':
        require(plan.get('seed_registry', {}).get('sha256') == RETEST_REGISTRY_SHA,
                'retest seed registry identity differs')
    require(plan['pins'] == pins(plan_size(plan)), 'campaign input identity changed')
    return plan


def validate_capture(item, plan, runtime, seed, seed_sha):
    require((item['size'], item['seed'], item['count'], item['batch_size'], item['seed_manifest_sha256'])
            == (plan_size(plan), seed, 256, 8, seed_sha), f'unexpected capture metadata for {runtime} seed {seed}')
    if runtime == 'python':
        require(item['upstream'] == UPSTREAM and item['input_sha256'] == py_weights(plan_size(plan))
                and item['source_sha256'][str(HERE / 'python_capture.py')] ==
                plan['pins'][str(HERE / 'python_capture.py')],
                'Python capture identity differs')
    elif plan.get('rust_device') == 'metal':
        require(item['binary_sha256'] == plan['metal_runner_sha256'] and item['device'] == 'metal'
                and item['artifact_manifest_sha256'] == manifest_sha(plan_size(plan))
                and item['convolution_math'] == 'MLX Metal'
                and item['sampler'] == 'device FP32 inverse CDF'
                and item['penalty'] == 'host FP64'
                and item['diagnostic']['rotary_tables_sha256'] == ROPE_SHA,
                'Rust Metal capture identity differs')
    else:
        require(item['binary_sha256'] == plan['pins'][str(RUNNER)]
                and item['artifact_manifest_sha256'] == manifest_sha(plan_size(plan))
                and item['convolution_math'] == 'cuDNN graph and efficient attention'
                and item['penalty'] == 'CUDA FP64'
                and item['diagnostic']['rotary_tables_sha256'] == ROPE_SHA,
                'Rust capture identity differs')
    for key, width, limit in (('codes', 641, 8192), ('raw_ids', 128, 30528)):
        values = item[key]
        require(len(values) == 256 and all(len(row) == width and
                    all(type(x) is int and 0 <= x < limit for x in row) for row in values),
                f'invalid {key} in {runtime} seed {seed}')


def capture_paths(directory, runtime, seeds):
    return {seed: directory / f'{runtime}-seed-{seed}.json' for seed in seeds}


def read_captures(plan, plan_sha, run, runtime):
    directory = run / runtime
    require(not directory.is_symlink(), 'capture directory must be real')
    receipt, _ = json_file(directory / 'receipt.json')
    seed_path = directory / 'seeds.json'
    require(seed_path.read_bytes() == payload(plan['seeds'][runtime]), 'runtime seed list differs')
    require(receipt['manifest_sha256'] == plan_sha and receipt['runtime'] == runtime,
            'capture receipt belongs to a different campaign')
    paths = capture_paths(directory, runtime, plan['seeds'][runtime])
    require(set(receipt['captures']) == {str(seed) for seed in paths}
            and {path.name for path in directory.glob(f'{runtime}-seed-*.json')} ==
            {path.name for path in paths.values()}, 'incomplete or unexpected captures')
    for seed, path in paths.items():
        record, sha = json_file(path, receipt['captures'][str(seed)])
        validate_capture(record, plan, runtime, seed, digest(seed_path))
        yield seed, sha, record


def import_python(plan, attempt):
    """Copy the published Python captures, bound by their receipt hash, into a Metal campaign."""
    source = plan['python_source']
    directory = Path(source['directory'])
    receipt, _ = json_file(directory / 'receipt.json', source['receipt_sha256'])
    require(receipt['runtime'] == 'python', 'imported receipt is not a Python capture receipt')
    source_seeds = directory / 'seeds.json'
    require(source_seeds.read_bytes() == payload(plan['seeds']['python']), 'imported seed list differs')
    paths = capture_paths(directory, 'python', plan['seeds']['python'])
    require(set(receipt['captures']) == {str(seed) for seed in paths}, 'imported receipt is incomplete')
    for seed, path in paths.items():
        record, _ = json_file(path, receipt['captures'][str(seed)])
        validate_capture(record, plan, 'python', seed, digest(source_seeds))
        shutil.copyfile(path, attempt / path.name)


def import_metal(plan, attempt, directory):
    """Copy Rust Metal captures produced on Apple hardware; validation follows in capture()."""
    require(directory is not None and directory.is_dir() and not directory.is_symlink(),
            'Metal capture import directory required')
    require((directory / 'seeds.json').read_bytes() == payload(plan['seeds']['rust']),
            'Metal capture seed list differs')
    for path in capture_paths(directory, 'rust', plan['seeds']['rust']).values():
        shutil.copyfile(path, attempt / path.name)
    if (directory / 'run.log').is_file():
        shutil.copyfile(directory / 'run.log', attempt / 'source-run.log')


def capture(plan, plan_sha, run, runtime, imported=None):
    target = run / runtime
    if target.exists():
        for _ in read_captures(plan, plan_sha, run, runtime):
            pass
        print(f'{runtime} capture already verified', flush=True)
        return
    attempt = Path(tempfile.mkdtemp(prefix=f'.{runtime}-attempt-', dir=run))
    seed_path = attempt / 'seeds.json'
    seed_path.write_bytes(payload(plan['seeds'][runtime]))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    env.pop('LD_LIBRARY_PATH', None)
    metal = plan.get('rust_device') == 'metal'
    require(imported is None or (metal and runtime == 'rust'), 'import applies only to Metal Rust captures')
    if metal:
        print(f'importing {runtime} into {attempt.name}', flush=True)
        if runtime == 'python':
            import_python(plan, attempt)
        else:
            import_metal(plan, attempt, imported)
        command = None
    elif runtime == 'python':
        command = [sys.executable, HERE / 'python_capture.py', '--upstream', UPSTREAM_DIR,
                   '--source', PY_SOURCE, '--size', plan_size(plan), '--seeds', seed_path, '--output', attempt]
    else:
        env.update(LD_LIBRARY_PATH=str(CUDNN), NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC=str(ROPE))
        command = [RUNNER, artifact(plan_size(plan)), seed_path, attempt]
    if command is not None:
        print(f'capturing {runtime} in {attempt.name}', flush=True)
        with (attempt / 'run.log').open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    results = {}
    seed_sha = digest(seed_path)
    require(seed_path.read_bytes() == payload(plan['seeds'][runtime]), 'runtime seed list changed')
    paths = capture_paths(attempt, runtime, plan['seeds'][runtime])
    require({path.name for path in attempt.glob(f'{runtime}-seed-*.json')} ==
            {path.name for path in paths.values()}, 'generated capture inventory differs')
    for seed, path in paths.items():
        record, sha = json_file(path)
        validate_capture(record, plan, runtime, seed, seed_sha)
        results[str(seed)] = sha
    require(plan == manifest(run / 'manifest.json', plan_sha), 'manifest changed during capture')
    atomic_publish(attempt / 'receipt.json', {'runtime': runtime, 'manifest_sha256': plan_sha, 'captures': results})
    attempt.rename(target)


def verify_model(scorer):
    for name, (size, sha) in scorer.MODEL_FILES.items():
        verify_input(MODEL / name, size, sha)
    files = json_file(BERT_MANIFEST)[0]['source']['files']
    for name, meta in files.items():
        verify_input(BERT / name, meta['bytes'], meta['sha256'])


def checked_score(path, runtime, seed, capture_sha, plan_sha, score_sha):
    score, sha = json_file(path, score_sha)
    require((score['runtime'], score['seed'], score['capture_sha256'], score['manifest_sha256'])
            == (runtime, seed, capture_sha, plan_sha) and score['upstream_exact'] is True,
            'score identity differs')
    entropy = score['entropy']
    require(len(entropy) == 256 and all(math.isfinite(x) and x >= 0 for x in entropy)
            and math.isfinite(score['ppl']) and score['ppl'] > 0
            and math.isfinite(score['nll_sum']) and score['nll_sum'] >= 0
            and 0 < score['token_count'] <= 256 * 127,
            'invalid score values')
    return score, sha


def score(plan, plan_sha, run):
    import lm1b_heldout_score as scorer
    from lm1b_shared_forward_score import corpus_ppl

    target = run / 'scores'
    if target.exists():
        for _ in read_scores(plan, plan_sha, run):
            pass
        print('scored cohort already verified', flush=True)
        return
    require(all((run / runtime / 'receipt.json').is_file() for runtime in RUNTIMES),
            'capture both runtime cohorts before scoring')
    before = {runtime: digest(run / runtime / 'receipt.json') for runtime in RUNTIMES}
    out = Path(tempfile.mkdtemp(prefix='.score-attempt-', dir=run))
    results = {}
    verify_model(scorer)
    bert = scorer.AutoTokenizer.from_pretrained(BERT, local_files_only=True, trust_remote_code=False)
    ref = scorer.AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if ref.pad_token is None:
        ref.pad_token = ref.eos_token
    model = scorer.AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, use_safetensors=True).cuda().eval()
    _, reference_entropy = scorer.upstream_functions(EVALUATOR, ref, model)
    for runtime in RUNTIMES:
        for seed, capture_sha, item in read_captures(plan, plan_sha, run, runtime):
            path = out / f'{runtime}-seed-{seed}-score.json'
            ids = item['raw_ids']
            ppl, nll, count = corpus_ppl(bert.batch_decode(ids), ref, model, EVALUATOR)
            entropy = [scorer.entropy(row) for row in ids]
            require(entropy == [reference_entropy(row) for row in ids], 'entropy differs from pinned upstream')
            atomic_publish(path, {'runtime': runtime, 'seed': seed, 'capture_sha256': capture_sha,
                                  'manifest_sha256': plan_sha, 'upstream_exact': True,
                                  'ppl': ppl, 'nll_sum': nll, 'token_count': count, 'entropy': entropy})
            results[path.name] = digest(path)
            print(f'{runtime} seed {seed}: scored', flush=True)
    verify_model(scorer)
    for runtime, sha in before.items():
        require(sha == digest(run / runtime / 'receipt.json'), 'capture receipt changed during scoring')
    require(plan == manifest(run / 'manifest.json', plan_sha), 'manifest changed during scoring')
    atomic_publish(out / 'receipt.json', {'manifest_sha256': plan_sha, 'scores': results})
    out.rename(target)


def read_scores(plan, plan_sha, run):
    out = run / 'scores'
    receipt, _ = json_file(out / 'receipt.json')
    expected = {f'{runtime}-seed-{seed}-score.json' for runtime in RUNTIMES for seed in plan['seeds'][runtime]}
    require(not out.is_symlink() and receipt['manifest_sha256'] == plan_sha
            and set(receipt['scores']) == expected and {p.name for p in out.glob('*-score.json')} == expected,
            'missing, extra or unrelated score files')
    for runtime in RUNTIMES:
        for seed, capture_sha, _ in read_captures(plan, plan_sha, run, runtime):
            path = out / f'{runtime}-seed-{seed}-score.json'
            item, score_sha = checked_score(path, runtime, seed, capture_sha, plan_sha, receipt['scores'][path.name])
            yield runtime, seed, capture_sha, score_sha, item


def summary(plan, plan_sha, run):
    records = {runtime: [] for runtime in RUNTIMES}
    receipts = {}
    for runtime, seed, capture_sha, score_sha, item in read_scores(plan, plan_sha, run):
        records[runtime].append({'seed': seed, 'count': 256, 'ppl': item['ppl'],
                                 'entropy': statistics.mean(item['entropy'])})
        receipts[f'{runtime}:{seed}'] = {'capture_sha256': capture_sha, 'score_sha256': score_sha}
    comparison = compare_cohorts(records, plan['seeds'])
    result = {'manifest_sha256': plan_sha, 'mode': plan['mode'], 'records': records,
              'identities': receipts, 'comparison': comparison}
    path = run / 'summary.json'
    if path.exists():
        require(path.read_bytes() == payload(result), 'existing summary differs from replay')
    else:
        atomic_publish(path, result)
    print(digest(path))
    return int(plan['mode'] in ('prospective', 'retest') and not comparison['metrics_passed'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='stage', required=True)
    init = commands.add_parser('init')
    init.add_argument('run', type=Path)
    init.add_argument('--mode', choices=('rehearsal', 'prospective', 'retest'), required=True,
                      help='retest is nonblind and reuses the frozen CUDA 80-seed registry')
    init.add_argument('--seeds', type=Path, required=True)
    init.add_argument('--rust-device', choices=('cuda', 'metal'), default='cuda')
    init.add_argument('--size', choices=tuple(SIZES), default='small')
    init.add_argument('--metal-runner-sha256', help='Metal mode: SHA-256 of the built Apple runner')
    init.add_argument('--python-from', type=Path,
                      help='Metal mode: published Python capture directory of the CUDA campaign')
    init.add_argument('--python-receipt-sha256', help='Metal mode: SHA-256 of that receipt.json')
    for stage in ('capture', 'score', 'summary'):
        action = commands.add_parser(stage)
        action.add_argument('run', type=Path)
        action.add_argument('--manifest-sha256', required=True)
        if stage == 'capture':
            action.add_argument('--runtime', choices=RUNTIMES, required=True)
            action.add_argument('--import', dest='imported', type=Path,
                                help='Metal mode: directory of Rust captures made on Apple hardware')
    args = p.parse_args()
    require(Path.cwd() == ROOT, 'run from the repository root')
    require(not args.run.is_symlink(), 'campaign directory must be real')
    run = args.run.resolve()
    require(run.is_relative_to(ROOT / 'reports') and run != ROOT / 'reports',
            'campaign directory must be under reports')
    path = run / 'manifest.json'
    if args.stage == 'init':
        registry, registry_sha = json_file(args.seeds)
        require(isinstance(registry, dict) and set(RUNTIMES) <= set(registry),
                'seed registry must list every runtime')
        if args.mode == 'retest':
            require(registry_sha == RETEST_REGISTRY_SHA and args.rust_device == 'cuda',
                    'retest requires the frozen registry and CUDA')
        if args.mode in ('prospective', 'retest'):
            # Every pre-registered list, including ones for later runtimes, must be disjoint.
            require(all(isinstance(values, list) for values in registry.values()),
                    'seed registry lists required')
            registered = [x for values in registry.values() for x in values]
            require(len(set(registered)) == len(registered),
                    'prospective seed registry lists must be disjoint')
        metal = args.rust_device == 'metal'
        rust_key = 'rust_metal' if metal else 'rust'
        require(rust_key in registry, f'seed registry lacks {rust_key}')
        seeds = seed_lists(args.mode, {'python': registry['python'], 'rust': registry[rust_key]})
        extra = {}
        if metal:
            require(args.mode == 'prospective' and args.python_from and args.python_receipt_sha256
                    and args.metal_runner_sha256 and len(args.metal_runner_sha256) == 64,
                    'Metal mode needs a prospective run, Python source and receipt hash, and runner hash')
            source = args.python_from.resolve()
            require(source.is_relative_to(ROOT / 'reports'), 'Python source must be under reports')
            json_file(source / 'receipt.json', args.python_receipt_sha256)
            extra = {'rust_device': 'metal', 'metal_runner_sha256': args.metal_runner_sha256,
                     'python_source': {'directory': str(source), 'receipt_sha256': args.python_receipt_sha256}}
        else:
            require(not (args.python_from or args.python_receipt_sha256 or args.metal_runner_sha256),
                    'Metal options require --rust-device metal')
        plan = {'schema_version': 1, 'params': PARAMS, 'mode': args.mode, 'seeds': seeds,
                'seed_registry': {'path': str(args.seeds), 'sha256': registry_sha},
                **extra, **({'size': args.size} if args.size != 'small' else {}),
                'pins': pins(args.size), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
        run.mkdir()
        atomic_publish(path, plan)
        if metal:
            # The exact seed-list bytes the Apple runner must read.
            (run / 'metal-seeds.json').write_bytes(payload(seeds['rust']))
        print(digest(path))
        return
    plan = manifest(path, args.manifest_sha256)
    if args.stage == 'capture':
        capture(plan, args.manifest_sha256, run, args.runtime, args.imported)
    elif args.stage == 'score':
        score(plan, args.manifest_sha256, run)
        manifest(path, args.manifest_sha256)
    else:
        return summary(plan, args.manifest_sha256, run)


if __name__ == '__main__':
    sys.exit(main())
