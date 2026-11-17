"""Score one identity-checked LM1B-small held-out capture against pinned upstream."""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinystories_artifact import ROOT, UPSTREAM, digest_file, json_file, verify_input
from tinystories_quality import MODEL_FILES
from lm1b_inventory import CHECKPOINTS
from lm1b_cuda_heldout_capture import CONFIGS

EVALUATOR_SHA = '6452cb5cc06a2355ab7cd16b9809b7144a557418e9041e4617f3c50aef5655fc'
SAMPLES = {
    'small': '0b1299d3f9b874a25ceb769f4d94733ccce0a62d8e2bc46b33a72eb28b6f850a',
    'medium': '0b9cb97801932f8bd81b30f4648691d4ee9a353a69941b6378e2fce20150fc13',
    'large': '8a7a3b83bed32f3df2156a88a28721a4c74fcc5d603ae581adc76bd98496d3f6',
}


def valid_targets(ids, eos):
    first_eos = (ids == eos).cumsum(-1) == 1
    return (first_eos[..., 1:] + (ids != eos)[..., 1:])


def entropy(ids):
    t = torch.tensor(ids)
    return float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum())


def corpus_ppl(texts, tokenizer, model):
    enc = tokenizer(texts, return_tensors='pt', return_token_type_ids=False,
                    truncation=True, padding=True, max_length=128)
    ids, attention = enc['input_ids'].cuda(), enc['attention_mask'].cuda()
    nll_sum = tok_sum = 0.0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 16):
            s, m = ids[i:i + 16], attention[i:i + 16]
            logits = model(s, attention_mask=m)[0].transpose(-1, -2)
            nll = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction='none')
            valid = valid_targets(s, tokenizer.eos_token_id).to(nll.dtype)
            nll_sum += float((nll * valid).sum())
            tok_sum += float(valid.sum())
    if tok_sum == 0:
        raise ValueError('no scoreable target tokens')
    return float(np.exp(nll_sum / tok_sum)), nll_sum, tok_sum


def upstream_functions(path, tokenizer, model):
    tree = ast.parse(path.read_bytes())
    parent = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'genppl_flm']
    if len(parent) != 1:
        raise ValueError('unexpected upstream evaluator')
    functions = [n for n in parent[0].body if isinstance(n, ast.FunctionDef) and n.name in ('corpus_ppl', 'entropy')]
    if {n.name for n in functions} != {'corpus_ppl', 'entropy'}:
        raise ValueError('unexpected upstream functions')
    scope = {'torch': torch, 'F': F, 'np': np, 'rtok': tokenizer, 'rlm': model}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), 'exec'), scope)
    return scope['corpus_ppl'], scope['entropy']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--bert', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--capture-sha256', required=True)
    p.add_argument('--runtime', choices=('python', 'rust'), required=True)
    p.add_argument('--seed', type=int, choices=(42, 0, 1, 2, 3), required=True)
    p.add_argument('--rust-binary-sha256')
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    upstream = args.upstream.resolve()
    def check_upstream():
        if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM or
                subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
            raise ValueError('upstream must be clean and pinned')
    check_upstream()
    watched = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_cuda_heldout_capture.py').resolve(),
               Path(__file__).with_name('lm1b_inventory.py').resolve(), Path(__file__).with_name('tinystories_quality.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(),
               ROOT / 'tools/reference/tokenizers-v1.lock.json',
               ROOT / 'artifacts/canonical/tokenizers-qualified/bert/manifest.json']
    before = {str(path): digest_file(path) for path in watched}
    evaluator = upstream / 'scripts/eval_merge.py'
    verify_input(evaluator, evaluator.stat().st_size, EVALUATOR_SHA)
    for name, (length, sha) in MODEL_FILES.items():
        verify_input(args.model / name, length, sha)
    tokenizer_manifest = ROOT / 'artifacts/canonical/tokenizers-qualified/bert/manifest.json'
    lock = json.loads((ROOT / 'tools/reference/tokenizers-v1.lock.json').read_bytes())['profiles']['bert']
    verify_input(tokenizer_manifest, tokenizer_manifest.stat().st_size, lock['manifest_sha256'])
    bert_manifest = json.loads(tokenizer_manifest.read_bytes())
    for name, meta in bert_manifest['source']['files'].items():
        verify_input(args.bert / name, meta['bytes'], meta['sha256'])
    path = args.capture
    verify_input(path, path.stat().st_size, args.capture_sha256)
    sample = json.loads(path.read_bytes())
    if (sample['size'], sample['seed'], sample['count'], sample['batch_size']) != ('small', args.seed, 256, 8):
        raise ValueError('unexpected held-out capture metadata')
    if args.runtime == 'python':
        identities = {'small/generator.pt': CHECKPOINTS['small/generator.pt'][1],
                      'tokenizer.pt': CHECKPOINTS['tokenizer.pt'][1],
                      'small/config.yaml': CONFIGS['small'][1]}
        source = str((ROOT / 'tools/reference/lm1b_cuda_heldout_capture.py').resolve())
        if (args.rust_binary_sha256 is not None or sample.get('upstream') != UPSTREAM
                or sample.get('input_sha256') != identities
                or sample.get('source_sha256', {}).get(source) != digest_file(Path(source))):
            raise ValueError('unexpected Python generation identity')
    else:
        diagnostic = {'rotary_tables_sha256': '9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc',
                      'backend': 'reference operators with 1 MiB QKV preference'}
        if (sample.get('artifact_manifest_sha256') != '58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76'
                or sample.get('diagnostic') != diagnostic or sample.get('penalty') != 'CUDA FP64'
                or args.rust_binary_sha256 is None or sample.get('binary_sha256') != args.rust_binary_sha256):
            raise ValueError('unexpected Rust generation identity')
    codes = sample['codes']
    if len(codes) != 256 or any(len(row) != 641 or any(type(x) is not int or not 0 <= x < 8192 for x in row) for row in codes):
        raise ValueError('invalid generated codes')
    rows = sample['raw_ids']
    if len(rows) != 256:
        raise ValueError('invalid held-out sample count')
    paths = {'heldout': path}
    sample_hashes = {'heldout': args.capture_sha256}
    if (not rows or any(len(row) != 128 or any(type(x) is not int or not 0 <= x < 30528 for x in row) for row in rows)):
        raise ValueError('invalid sample rows')
    bert = AutoTokenizer.from_pretrained(args.bert, local_files_only=True, trust_remote_code=False)
    ref = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if ref.pad_token is None:
        ref.pad_token = ref.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, use_safetensors=True).cuda().eval()
    texts = bert.batch_decode(rows)
    reference_ppl, reference_entropy = upstream_functions(evaluator, ref, model)
    expected = reference_ppl(texts)
    actual, nll, count = corpus_ppl(texts, ref, model)
    actual_entropy = [entropy(row) for row in rows]
    expected_entropy = [reference_entropy(row) for row in rows]
    if actual != expected or actual_entropy != expected_entropy:
        raise ValueError('diagnostic differs from pinned upstream')
    for name, (length, sha) in MODEL_FILES.items():
        verify_input(args.model / name, length, sha)
    for name, meta in bert_manifest['source']['files'].items():
        verify_input(args.bert / name, meta['bytes'], meta['sha256'])
    for size, path in paths.items():
        verify_input(path, path.stat().st_size, sample_hashes[size])
    verify_input(evaluator, evaluator.stat().st_size, EVALUATOR_SHA)
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('scorer inputs changed during probe')
    check_upstream()
    json_file(args.output, {'scope': 'one held-out seed scored with pinned upstream; overall gate requires comparison',
                             'runtime': args.runtime, 'seed': args.seed, 'scorer_sha256': before[str(Path(__file__).resolve())],
                             'upstream': UPSTREAM, 'evaluator_sha256': EVALUATOR_SHA, 'sample_sha256': sample_hashes,
                             'count': len(rows), 'ppl': actual, 'nll_sum': nll, 'token_count': count,
                             'entropy': actual_entropy, 'upstream_exact': True,
                             'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    print(json.dumps({'ppl': actual, 'token_count': count, 'upstream_exact': True,
                      'report_sha256': digest_file(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
