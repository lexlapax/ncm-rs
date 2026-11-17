"""Compare a bounded LM1B quality diagnostic with the pinned upstream scorer."""
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
    p.add_argument('--repeat', type=int, choices=(1, 6), default=1)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    upstream = args.upstream.resolve()
    def check_upstream():
        if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM or
                subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
            raise ValueError('upstream must be clean and pinned')
    check_upstream()
    watched = [Path(__file__).resolve(), Path(__file__).with_name('tinystories_quality.py').resolve(),
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
    paths = {size: ROOT / f'reports/phase4-lm1b-{size}-cpu-probe-v1.json' for size in SAMPLES}
    rows = []
    for size, path in paths.items():
        verify_input(path, path.stat().st_size, SAMPLES[size])
        sample = json.loads(path.read_bytes())
        if sample['size'] != size or len(sample['raw_ids']) != 128 or any(type(x) is not int or not 0 <= x < 30528 for x in sample['raw_ids']):
            raise ValueError('invalid sample')
        rows.append(sample['raw_ids'])
    rows = rows * args.repeat
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
        verify_input(path, path.stat().st_size, SAMPLES[size])
    verify_input(evaluator, evaluator.stat().st_size, EVALUATOR_SHA)
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('scorer inputs changed during probe')
    check_upstream()
    json_file(args.output, {'scope': 'repeated inspected CPU samples; scorer development only, not quality or held-out validation',
                             'upstream': UPSTREAM, 'evaluator_sha256': EVALUATOR_SHA, 'sample_sha256': SAMPLES,
                             'count': len(rows), 'repeat': args.repeat, 'ppl': actual, 'nll_sum': nll, 'token_count': count,
                             'entropy': actual_entropy, 'upstream_exact': True,
                             'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    print(json.dumps({'ppl': actual, 'token_count': count, 'upstream_exact': True,
                      'report_sha256': digest_file(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
