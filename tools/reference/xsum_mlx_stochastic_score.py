"""Score Metal XSum samples using the pinned CUDA evaluator, with a replay check."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from xsum_artifact import digest_file
from xsum_quality_generate import DATA_SHA, ROOT
from xsum_quality_score import DEPS, EVAL_DEPS, GENPPL_SHA, MODEL, MODEL_HASHES, decode_ids, verified_file, verify_scorer_files
from xsum_stochastic_followup_analyze import metrics

DATA = ROOT / 'artifacts/canonical/text-phase4-v1/xsum.json'
PROTOCOL = ROOT / 'docs/results/2026-11-15-phase4-xsum/stochastic-followup-protocol-v1.json'
PROTOCOL_SHA = '8ccd05295382f6ad33b17183a3f24e5d726da74a73fc1a7027eea81af10d5332'
CAPTURE_SHA = '0b3215af1443f46851f6db1a707350b1da38563be16f595a51a59cb47c5fec11'
ARTIFACT_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
BINARY_SHA = '498b70050edc69f7cc93255b9c681d3bc34ff628b0eb07b35f6e40223654cac2'
REFERENCE_SHA = '32f3b6a3dd93dd2241ff8f9c73c31ed83e77cfaa90fbf45e3e9198b899ddb869'
SCORER_DEPENDENCIES = tuple(ROOT / name for name in (
    'tools/reference/xsum_artifact.py', 'tools/reference/tinystories_artifact.py',
    'tools/reference/xsum_rope_reference.py', 'tools/reference/xsum_quality_generate.py',
    'tools/reference/xsum_quality_score.py', 'tools/reference/xsum_quality_rust.py',
    'tools/reference/xsum_linked_libraries.py', 'tools/reference/xsum_stochastic_followup_capture.py',
    'tools/reference/xsum_stochastic_followup_analyze.py',
))


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()


def read_metal(capture, data):
    manifest = json.loads((capture / 'manifest.json').read_bytes())
    seed = manifest.get('seed')
    expected = {'head': 'e14ae02a517b0b3d1000d942f94d72bc408d851a', 'data_sha256': DATA_SHA,
                'artifact_sha256': ARTIFACT_SHA, 'binary_sha256': BINARY_SHA, 'protocol_sha256': PROTOCOL_SHA,
                'script_sha256': CAPTURE_SHA, 'batches': 64, 'batch_size': 32, 'start': 512,
                'sampling': {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'cfg': 0.0}}
    if seed not in (4, 5, 6, 7, 8) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('unexpected Metal capture manifest')
    if json.loads((capture / 'complete.json').read_bytes()) != {'seed': seed, 'batches': 64, 'binary_sha256': BINARY_SHA}:
        raise ValueError('incomplete capture')
    rows = []
    files = [capture / 'manifest.json', capture / 'complete.json']
    for batch in range(64):
        start = 512 + batch * 32
        prefix = [(row[:512] + [50257] * 512)[:512] for row in data['inputs'][start:start + 32]]
        input_path = capture / f'input-{batch:02}.json'
        receipt_path = capture / f'batch-{batch:02}.json'
        if input_path.read_bytes() != canonical(prefix) + b'\n':
            raise ValueError('captured prefixes differ')
        envelope = json.loads(receipt_path.read_bytes())
        if (set(envelope) != {'record', 'sha256'} or
                envelope['sha256'] != hashlib.sha256(canonical(envelope['record'])).hexdigest()):
            raise ValueError('invalid Metal batch receipt')
        record = envelope['record']
        if (set(record) != {'batch', 'start', 'batch_seed', 'result'}
                or record.get('batch') != batch or record.get('start') != start
                or record.get('batch_seed') != seed * 1000 + batch):
            raise ValueError('Metal batch identity differs')
        result = record['result']
        if type(result) is not dict or set(result) != {'codes', 'ids'}:
            raise ValueError('invalid Metal output')
        for key, width, bound in (('codes', 200, 4096), ('ids', 48, 50304)):
            values = result[key]
            if (type(values) is not list or len(values) != 32 or any(
                type(row) is not list or len(row) != width or any(
                    type(v) is not int or not 0 <= v < bound for v in row) for row in values)):
                raise ValueError('invalid Metal output rows')
        rows.extend(result['ids'])
        files.extend((input_path, receipt_path))
    return manifest, rows, {str(p): digest_file(p) for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--capture', type=Path)
    group.add_argument('--replay-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    if any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH', 'CUDA_INJECTION64_PATH',
                                      'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING',
                                      'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'PYTORCH_ALLOC_CONF',
                                      'PYTORCH_CUDA_ALLOC_CONF')):
        raise ValueError('environment overrides unsupported')
    if digest_file(DATA) != DATA_SHA or digest_file(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('dataset or protocol differs')
    scorer_sha = digest_file(Path(__file__))
    dependency_sha = {str(path.relative_to(ROOT)): digest_file(path) for path in SCORER_DEPENDENCIES}
    data = json.loads(DATA.read_bytes())
    if data.get('profile') != 'xsum' or len(data.get('references', [])) != 11334:
        raise ValueError('invalid dataset')
    if args.replay_reference:
        if digest_file(args.replay_reference) != REFERENCE_SHA:
            raise ValueError('replay reference score differs')
        prior = json.loads(args.replay_reference.read_bytes())
        if [r['index'] for r in prior['rows']] != list(range(512, 2560)):
            raise ValueError('unexpected replay rows')
        raw_ids, capture_manifest, capture_files = None, None, {str(args.replay_reference): REFERENCE_SHA}
    else:
        prior = None
        capture_manifest, raw_ids, capture_files = read_metal(args.capture, data)
    upstream = ROOT / '.reference/upstream-ncm'
    verified_file(upstream / 'evaluation/gen_ppl.py', GENPPL_SHA)
    for name, sha in MODEL_HASHES.items():
        verified_file(MODEL / name, sha)
    verify_scorer_files()
    sys.path.insert(0, str(EVAL_DEPS)); sys.path.insert(0, str(DEPS))
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    versions = {name: importlib.metadata.version(name) for name in ('nltk', 'absl-py', 'six', 'joblib', 'click', 'transformers', 'numpy', 'regex', 'tqdm')}
    expected_versions = {'nltk': '3.9.2', 'absl-py': '2.3.1', 'six': '1.17.0', 'joblib': '1.5.2',
                         'click': '8.2.1', 'transformers': '4.57.3', 'numpy': '2.5.4',
                         'regex': '2026.11.9', 'tqdm': '4.70.1'}
    if versions != expected_versions or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('scorer dependencies or CUDA device differ')
    if prior and (prior.get('torch') != torch.__version__ or prior.get('versions') != versions):
        raise ValueError('replay evaluator versions differ from retained score')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1:
        raise ValueError('one identified CUDA device required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if not tokenizer.is_fast or len(tokenizer) != 50257:
        raise ValueError('unexpected tokenizer')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rows = []
    for offset in range(2048):
        index = 512 + offset
        text = prior['rows'][offset]['text'] if prior else decode_ids(tokenizer, raw_ids[offset])
        rouge = scorer.score(data['references'][index], text)
        rows.append({'index': index, 'text': text, 'words': len(text.split()),
                     'rouge': {key: rouge[key].fmeasure for key in ('rouge1', 'rouge2', 'rougeL')}})
    sys.path.insert(0, str(upstream))
    from evaluation.gen_ppl import generation_perplexity
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True).cuda().eval()
    for start in range(0, len(rows), 32):
        sample = rows[start:start + 32]
        value = generation_perplexity(model, tokenizer, [''] * len(sample), [r['text'] for r in sample],
                                      device='cuda', score_only_generated=True, batch_size=32)
        for row, nll, count in zip(sample, value.nll_sum.tolist(), value.token_count.tolist()):
            if not math.isfinite(nll) or nll < 0 or count < 0:
                raise ValueError('invalid perplexity observation')
            row['nll'], row['tokens'] = nll, count
        del value
        gc.collect()
    if (digest_file(DATA) != DATA_SHA or digest_file(PROTOCOL) != PROTOCOL_SHA
            or digest_file(Path(__file__)) != scorer_sha
            or {str(path.relative_to(ROOT)): digest_file(path) for path in SCORER_DEPENDENCIES} != dependency_sha):
        raise ValueError('inputs changed while scoring')
    if any(digest_file(Path(path)) != sha for path, sha in capture_files.items()):
        raise ValueError('capture changed while scoring')
    for name, sha in MODEL_HASHES.items():
        verified_file(MODEL / name, sha)
    result = {'schema_version': 1, 'runtime': 'replay' if prior else 'metal', 'manifest': capture_manifest,
              'capture_files_sha256': capture_files, 'scorer_sha256': scorer_sha,
              'scorer_dependencies_sha256': dependency_sha,
              'genppl_sha256': GENPPL_SHA, 'model_sha256': MODEL_HASHES, 'versions': versions,
              'torch': torch.__version__, 'gpu_identity_and_driver': gpu[0],
              'metrics': metrics(rows), 'rows': rows}
    if prior:
        result['replay_exact_rows'] = rows == prior['rows']
        result['replay_exact_metrics'] = result['metrics'] == prior['metrics']
    temporary = args.output.with_name(f'{args.output.name}.{os.getpid()}.part')
    with temporary.open('x') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':')); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, args.output); temporary.unlink()
    if prior and not (result['replay_exact_rows'] and result['replay_exact_metrics']):
        raise SystemExit('evaluator replay differs; retained result is not qualified')


if __name__ == '__main__':
    main()
