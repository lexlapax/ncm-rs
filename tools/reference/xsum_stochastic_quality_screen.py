"""Describe XSum stochastic sample quality on development rows."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from xsum_artifact import INPUTS, digest_file
from xsum_quality_generate import DATA_SHA, UPSTREAM
from xsum_quality_score import (DEPS, EVAL_DEPS, GENPPL_SHA, MODEL, MODEL_HASHES,
                                decode_ids, verified_file, verify_scorer_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--python-reference', type=Path, nargs='+', required=True)
    parser.add_argument('--rust', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('new output required')
    if digest_file(args.data) != DATA_SHA or not torch.cuda.is_available():
        raise ValueError('pinned data and CUDA required')
    data = json.loads(args.data.read_bytes())
    if len(args.python_reference) != len(args.rust) or not 1 <= len(args.rust) <= 8:
        raise ValueError('need 1..8 paired batches')
    samples = {'python': [], 'rust': []}
    trace_hashes, rust_hashes = [], []
    for batch, (directory, rust_path) in enumerate(zip(args.python_reference, args.rust)):
        manifest = json.loads((directory / 'report.json').read_bytes())
        trace_path = directory / 'trace.safetensors'
        if (manifest.get('upstream') != UPSTREAM or manifest.get('batch') != 32
                or manifest.get('seed') != 42 + batch or manifest.get('cfg_scale') != 0.0
                or manifest.get('sources') != {name: value[1] for name, value in INPUTS.items()}
                or manifest.get('mode') != 'upstream temperature 1, top-k 0, top-p 1'
                or manifest.get('trace_sha256') != digest_file(trace_path)):
            raise ValueError('unexpected Python sample trace')
        trace = load_file(trace_path)
        rows = trace['generator.prefix_ids'].tolist()
        expected = [(row[:512] + [50257] * 512)[:512]
                    for row in data['inputs'][batch * 32:(batch + 1) * 32]]
        if rows != expected:
            raise ValueError('input rows differ from pinned dataset')
        samples['python'].extend(trace['generator.penalized_ids'].tolist())
        samples['rust'].extend(json.loads(rust_path.read_bytes())['ids'])
        trace_hashes.append(digest_file(trace_path)); rust_hashes.append(digest_file(rust_path))
    count_rows = 32 * len(args.rust)
    if any(len(rows) != count_rows or any(len(row) != 48 or any(type(x) is not int or not 0 <= x < 50304
                                                        for x in row) for row in rows)
           for rows in samples.values()):
        raise ValueError('invalid sampled output')
    verified_file(args.upstream / 'evaluation/gen_ppl.py', GENPPL_SHA)
    for name, sha in MODEL_HASHES.items():
        verified_file(MODEL / name, sha)
    verify_scorer_files()
    sys.path.insert(0, str(EVAL_DEPS)); sys.path.insert(0, str(DEPS))
    sys.path.insert(0, str(args.upstream.resolve()))
    from evaluation.gen_ppl import generation_perplexity
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = {}
    for runtime, raw in samples.items():
        texts = [decode_ids(tokenizer, row) for row in raw]
        scores = [scorer.score(reference, text) for reference, text in zip(data['references'][:count_rows], texts)]
        ppl = generation_perplexity(model, tokenizer, [''] * count_rows, texts, device='cuda',
                                    score_only_generated=True, batch_size=32)
        nll, count = float(ppl.nll_sum.sum().item()), int(ppl.token_count.sum().item())
        results[runtime] = {key: 100 * sum(row[key].fmeasure for row in scores) / count_rows
                            for key in ('rouge1', 'rouge2', 'rougeL')}
        results[runtime].update({'generation_perplexity': math.exp(nll / max(count, 1)),
                                 'scored_tokens': count,
                                 'mean_words': sum(len(text.split()) for text in texts) / count_rows})
    report = {'scope': f'exploratory first {count_rows} rows, different random streams',
              'data_sha256': DATA_SHA, 'python_trace_sha256': trace_hashes,
              'rust_output_sha256': rust_hashes,
              'scorer_sha256': digest_file(Path(__file__)),
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'metrics': results}
    with args.output.open('x') as out:
        json.dump(report, out, sort_keys=True, indent=2)
        out.write('\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
