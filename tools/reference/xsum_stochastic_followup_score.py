"""Score a source-bound stochastic XSum capture using the pinned evaluator."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from safetensors.torch import load_file
from xsum_artifact import INPUTS, digest_file, verify_input
from xsum_stochastic_followup_capture import (ARTIFACT_SHA, BINARY, CUDNN, DATA, DATA_SHA,
    ROOT, SEEDS, SOURCE, UPSTREAM, linked_libraries, sources, validate_batch)
from xsum_quality_score import (DEPS, EVAL_DEPS, GENPPL_SHA, MODEL, MODEL_HASHES,
    decode_ids, verified_file, verify_scorer_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(name) for name in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH',
           'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
           'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
           'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')):
        raise ValueError('environment overrides are unsupported')
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    if args.capture.is_symlink() or not args.capture.is_dir():
        raise ValueError('capture directory required')
    manifest_path = args.capture / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    runtime, subset, seed = manifest['runtime'], manifest['set'], manifest['seed']
    if (runtime not in ('python', 'rust') or subset not in SEEDS or seed not in SEEDS[subset]
            or manifest.get('schema_version') != 1 or manifest.get('start') != (0 if subset == 'development' else 512)
            or manifest.get('rows') != 2048 or manifest.get('batch_size') != 32
            or manifest.get('batch_seeds') != [seed * 1000 + batch for batch in range(64)]
            or manifest.get('sampling') != {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'cfg': 0.0}
            or manifest.get('data_sha256') != DATA_SHA or manifest.get('artifact_sha256') != ARTIFACT_SHA
            or manifest.get('upstream') != UPSTREAM or manifest.get('source_sha256') != sources()
            or manifest.get('native_libraries') != linked_libraries(BINARY, CUDNN)
            or manifest.get('checkpoint_sha256') != {name: spec[1] for name, spec in INPUTS.items()}
            or (subset == 'heldout' and not manifest.get('protocol_sha256'))):
        raise ValueError('capture manifest differs')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if gpu != [manifest.get('gpu_identity_and_driver')]:
        raise ValueError('capture GPU or driver differs')
    if digest_file(DATA) != DATA_SHA:
        raise ValueError('dataset differs')
    upstream = ROOT / '.reference/upstream-ncm'
    verified_file(upstream / 'evaluation/gen_ppl.py', GENPPL_SHA)
    for name, sha in MODEL_HASHES.items():
        verified_file(MODEL / name, sha)
    for name, (size, sha) in INPUTS.items():
        verify_input(SOURCE / name, size, sha)
    verify_scorer_files()
    sys.path.insert(0, str(EVAL_DEPS)); sys.path.insert(0, str(DEPS))
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    versions = {name: importlib.metadata.version(name) for name in ('nltk', 'absl-py', 'six', 'joblib', 'click', 'transformers', 'numpy', 'regex', 'tqdm')}
    expected_versions = {'nltk': '3.9.2', 'absl-py': '2.3.1', 'six': '1.17.0', 'joblib': '1.5.2',
                         'click': '8.2.1', 'transformers': '4.57.3', 'numpy': '2.5.4',
                         'regex': '2026.11.9', 'tqdm': '4.70.1'}
    if versions != expected_versions:
        raise ValueError('scorer dependency versions differ')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('one CUDA device required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if not tokenizer.is_fast or len(tokenizer) != 50257:
        raise ValueError('unexpected tokenizer')
    data = json.loads(DATA.read_bytes())
    if data.get('profile') != 'xsum' or len(data.get('references', [])) != 11334:
        raise ValueError('invalid dataset')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rows = []
    for batch in range(64):
        start = manifest['start'] + batch * 32
        prefixes = [(row[:512] + [50257] * 512)[:512] for row in data['inputs'][start:start + 32]]
        expected = json.dumps(prefixes, separators=(',', ':')).encode() + b'\n'
        if (args.capture / f'input-b{batch}.json').read_bytes() != expected:
            raise ValueError('capture input differs')
        validate_batch(args.capture, runtime, batch, seed * 1000 + batch, prefixes)
        if runtime == 'python':
            ids = load_file(args.capture / f'reference-b{batch}/trace.safetensors')['generator.penalized_ids'].tolist()
        else:
            ids = json.loads((args.capture / f'rust-b{batch}.json').read_bytes())['ids']
        for offset, raw in enumerate(ids):
            index = start + offset
            text = decode_ids(tokenizer, raw)
            rouge = scorer.score(data['references'][index], text)
            rows.append({'index': index, 'text': text, 'words': len(text.split()),
                         'rouge': {key: rouge[key].fmeasure for key in ('rouge1', 'rouge2', 'rougeL')}})
    sys.path.insert(0, str(upstream))
    from evaluation.gen_ppl import generation_perplexity
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True).cuda().eval()
    for start in range(0, len(rows), 32):
        sample = rows[start:start + 32]
        result = generation_perplexity(model, tokenizer, [''] * len(sample), [r['text'] for r in sample],
                                       device='cuda', score_only_generated=True, batch_size=32)
        for row, nll, count in zip(sample, result.nll_sum.tolist(), result.token_count.tolist()):
            if not math.isfinite(nll) or nll < 0 or count < 0:
                raise ValueError('invalid perplexity observation')
            row['nll'], row['tokens'] = nll, count
        del result
        gc.collect()
    total_tokens = sum(r['tokens'] for r in rows)
    if total_tokens <= 0:
        raise ValueError('no evaluable tokens')
    metrics = {key: 100 * sum(r['rouge'][key] for r in rows) / len(rows) for key in ('rouge1', 'rouge2', 'rougeL')}
    metrics['mean_words'] = sum(r['words'] for r in rows) / len(rows)
    metrics['generation_perplexity'] = math.exp(sum(r['nll'] for r in rows) / total_tokens)
    if digest_file(DATA) != DATA_SHA or manifest['source_sha256'] != sources():
        raise ValueError('inputs changed while scoring')
    for name, sha in MODEL_HASHES.items():
        verified_file(MODEL / name, sha)
    result = {'schema_version': 1, 'manifest': manifest, 'manifest_sha256': digest_file(manifest_path),
              'scorer_sha256': digest_file(Path(__file__)), 'genppl_sha256': GENPPL_SHA,
              'model_sha256': MODEL_HASHES, 'versions': versions, 'torch': torch.__version__,
              'metrics': metrics, 'rows': rows}
    temporary = args.output.with_name(f'{args.output.name}.{os.getpid()}.part')
    with temporary.open('x') as f:
        json.dump(result, f, sort_keys=True, separators=(',', ':')); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    os.link(temporary, args.output); temporary.unlink()


if __name__ == '__main__':
    main()
