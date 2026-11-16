"""Score verified XSum quality batches with pinned, local evaluator assets."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from xsum_artifact import INPUTS, digest_file
from xsum_quality_generate import DATA_SHA, EVALUATOR_SHA, UPSTREAM, read_record, shard_batches
from xsum_quality_rust import ARTIFACT_SHA, input_hashes as rust_input_hashes, read_batch as read_rust_batch

ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT / '.reference/phase4-xsum-rouge-v1'
EVAL_DEPS = ROOT / '.reference/phase4-eval-deps'
PACKAGES = ROOT / 'artifacts/upstream/xsum-quality-evaluator'
MODEL = ROOT / 'artifacts/upstream/gpt2-evaluator-phase4-v1'
MODEL_HASHES = {
    'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
    'generation_config.json': 'ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    'model.safetensors': '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707',
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
}
WHEELS = {
    'nltk-3.9.2-py3-none-any.whl': '1e209d2b3009110635ed9709a67a1a3e33a10f799490fa71cf4bec218c11c88a',
    'absl_py-2.3.1-py3-none-any.whl': 'eeecf07f0c2a93ace0772c92e596ace6d3d3996c042b2128459aaae2a76de11d',
    'six-1.17.0-py2.py3-none-any.whl': '4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274',
    'joblib-1.5.2-py3-none-any.whl': '4e1f0bdbb987e6d843c70cf43714cb276623def372df3c22fe5266b2670bc241',
    'click-8.2.1-py3-none-any.whl': '61a3265b914e850b85317d0b3109c7f8cd35a670f963866005d6ef1d5175a12b',
}
ROUGE_ARCHIVE = 'rouge_score-0.1.2.tar.gz'
ROUGE_SHA = 'c7d4da2683e68c9abf0135ef915d63a46643666f848e558a1b9f7ead17ff0f04'
GENPPL_SHA = '31cde7d31aadc8cd873f7a853e818270e27e4789af4fb573d1821c8a0072ef5f'


def verified_file(path, sha):
    if path.is_symlink() or not path.is_file() or digest_file(path) != sha:
        raise ValueError(f'verification failed: {path.name}')


def verify_scorer_files():
    verified_file(PACKAGES / ROUGE_ARCHIVE, ROUGE_SHA)
    with tarfile.open(PACKAGES / ROUGE_ARCHIVE, 'r:gz') as archive:
        for item in archive:
            prefix = 'rouge_score-0.1.2/rouge_score/'
            if not item.name.startswith(prefix) or not item.isfile():
                continue
            name = item.name.removeprefix(prefix)
            if '/' in name or not name.endswith('.py') or item.size > 100000:
                raise ValueError('unexpected ROUGE scorer source')
            path = DEPS / 'rouge_score' / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != archive.extractfile(item).read():
                raise ValueError('installed ROUGE scorer differs')
    for name, sha in WHEELS.items():
        wheel = PACKAGES / name
        verified_file(wheel, sha)
        with zipfile.ZipFile(wheel) as archive:
            for entry in archive.infolist():
                name_in_wheel = entry.filename
                if not name_in_wheel.endswith('.py') or '.data/' in name_in_wheel or '.dist-info/' in name_in_wheel:
                    continue
                parts = Path(name_in_wheel).parts
                if not parts or any(part in ('', '.', '..') for part in parts):
                    raise ValueError('unexpected wheel member')
                installed = DEPS.joinpath(*parts)
                if installed.is_symlink() or not installed.is_file() or installed.read_bytes() != archive.read(entry):
                    raise ValueError('installed scorer dependency differs')


def checked_rows(shard_dirs, stop):
    all_rows = {}
    manifests = []
    rust_hashes = None
    for shard, directory in enumerate(shard_dirs):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('quality shard must be a regular directory')
        metadata = directory / 'manifest.json'
        if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size > 65536:
            raise ValueError('quality manifest missing')
        manifest = json.loads(metadata.read_bytes())
        if (manifest.get('schema_version') != 1 or manifest.get('shard') != shard or
            manifest.get('shards') != 2 or manifest.get('data_sha256') != DATA_SHA or
            manifest.get('batch_size') != 32 or manifest.get('temperature') != 1.0 or
            manifest.get('top_k') != 1 or manifest.get('top_p') != 0.0 or manifest.get('cfg') != 0.0 or
            not manifest.get('gpu_identity_and_driver')):
            raise ValueError('quality manifest does not match pinned protocol')
        origin = manifest.get('origin')
        if origin == 'pinned-upstream-cuda':
            if (manifest.get('upstream') != UPSTREAM or manifest.get('evaluator_sha256') != EVALUATOR_SHA or
                manifest.get('sources') != {k: v[1] for k, v in INPUTS.items()} or
                manifest.get('tf32') is not False or manifest.get('sdpa') != 'math' or
                manifest.get('between_batch_gc') is not True or type(manifest.get('seed')) is not int or
                manifest.get('generator_sha256') != digest_file(ROOT / 'tools/reference/xsum_quality_generate.py')):
                raise ValueError('upstream quality manifest differs')
            reader = read_record
        elif origin == 'rust-strict-cudnn':
            if rust_hashes is None:
                rust_hashes = rust_input_hashes()
            if manifest.get('artifact_sha256') != ARTIFACT_SHA or manifest.get('binary_and_source_sha256') != rust_hashes:
                raise ValueError('Rust quality manifest differs')
            reader = read_rust_batch
        else:
            raise ValueError('unknown quality origin')
        manifests.append(manifest)
        _, batches = shard_batches(shard, stop)
        for ordinal, indices in batches:
            record = reader(directory / f'batch-{ordinal:05}.json', indices, ordinal)
            for idx, codes, ids in zip(indices, record['codes'], record['raw_ids']):
                if idx in all_rows:
                    raise ValueError('duplicate quality row')
                all_rows[idx] = {'index': idx, 'codes': codes, 'raw_ids': ids}
    if sorted(all_rows) != list(range(stop)):
        raise ValueError('quality rows do not cover the requested boundary')
    comparable = [{k: v for k, v in m.items() if k != 'shard'} for m in manifests]
    if comparable[0] != comparable[1]:
        raise ValueError('quality shard environments differ')
    return [all_rows[i] for i in range(stop)], manifests


def decode_ids(tokenizer, raw):
    eos, pad = 50256, 50257
    if eos in raw:
        raw = raw[:raw.index(eos)]
    return tokenizer.decode([x for x in raw if x not in (eos, pad)], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard0', type=Path, required=True)
    parser.add_argument('--shard1', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=11334)
    parser.add_argument('--with-ppl', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 11334 or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('invalid row bounds or output')
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output already exists')
    for key in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES',
                'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE'):
        if os.environ.get(key): raise ValueError('injection or device override is unsupported')
    if args.data.is_symlink() or digest_file(args.data) != DATA_SHA:
        raise ValueError('dataset hash differs')
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if revision != UPSTREAM or dirty: raise ValueError('upstream checkout must be clean and pinned')
    verified_file(args.upstream / 'evaluation/gen_ppl.py', GENPPL_SHA)
    for name, sha in MODEL_HASHES.items():
        if args.with_ppl or name != 'model.safetensors': verified_file(MODEL / name, sha)
    verify_scorer_files()
    sys.path.insert(0, str(EVAL_DEPS)); sys.path.insert(0, str(DEPS))
    from rouge_score import rouge_scorer
    from transformers import AutoTokenizer
    versions = {name: importlib.metadata.version(name) for name in ('nltk', 'absl-py', 'six', 'joblib', 'click', 'transformers', 'numpy', 'regex', 'tqdm')}
    expected_versions = {'nltk': '3.9.2', 'absl-py': '2.3.1', 'six': '1.17.0', 'joblib': '1.5.2',
                         'click': '8.2.1', 'transformers': '4.57.3', 'numpy': '2.5.4',
                         'regex': '2026.11.9', 'tqdm': '4.70.1'}
    if versions != expected_versions: raise ValueError('scorer dependency versions differ')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if not tokenizer.is_fast or len(tokenizer) != 50257: raise ValueError('unexpected evaluator tokenizer')
    data = json.loads(args.data.read_bytes())
    if data.get('profile') != 'xsum' or len(data.get('references', [])) != 11334: raise ValueError('invalid dataset')
    rows, manifests = checked_rows([args.shard0, args.shard1], args.stop)
    rows = rows[args.start:]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for row in rows:
        row['hypothesis'] = decode_ids(tokenizer, row['raw_ids'])
        scores = scorer.score(data['references'][row['index']], row['hypothesis'])
        row['rouge'] = {key: scores[key].fmeasure for key in ('rouge1', 'rouge2', 'rougeL')}
    metrics = {key: 100 * sum(row['rouge'][key] for row in rows) / len(rows) for key in ('rouge1', 'rouge2', 'rougeL')}
    metrics['mean_words'] = sum(len(row['hypothesis'].split()) for row in rows) / len(rows)
    if args.with_ppl:
        import gc
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise ValueError('one CUDA device required')
        sys.path.insert(0, str(args.upstream.resolve()))
        from evaluation.gen_ppl import generation_perplexity
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True).cuda().eval()
        nll, counts = [], []
        for start in range(0, len(rows), 32):
            hyps = [row['hypothesis'] for row in rows[start:start + 32]]
            result = generation_perplexity(model, tokenizer, [''] * len(hyps), hyps, device='cuda', score_only_generated=True, batch_size=32)
            nll.append(result.nll_sum); counts.append(result.token_count)
            del result
            gc.collect()
        nll = torch.cat(nll); counts = torch.cat(counts)
        metrics['generation_perplexity'] = math.exp(float(nll.sum().item()) / max(int(counts.sum().item()), 1))
        metrics['generation_ppl_token_count'] = int(counts.sum().item())
    if digest_file(args.data) != DATA_SHA: raise ValueError('dataset changed while scoring')
    for name, sha in MODEL_HASHES.items():
        if args.with_ppl or name != 'model.safetensors': verified_file(MODEL / name, sha)
    result = {'schema_version': 2, 'scope': manifests[0]['origin'] + ' XSum outputs',
              'start': args.start, 'stop': args.stop,
              'model_revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e', 'versions': versions,
              'scorer_sha256': digest_file(Path(__file__)), 'shard_manifests': manifests,
              'metrics': metrics, 'rows': rows}
    temporary = args.output.with_name(f'{args.output.name}.{os.getpid()}.part')
    with temporary.open('x') as f:
        json.dump(result, f, sort_keys=True, separators=(',', ':'))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.link(temporary, args.output)
    temporary.unlink()


if __name__ == '__main__':
    main()
