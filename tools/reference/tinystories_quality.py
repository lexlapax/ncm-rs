"""Score TinyStories samples with the pinned upstream GPT-2-large metric."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinystories_artifact import UPSTREAM, verify_input

MODEL_FILES = {
    'config.json': (666, '7fccdcfd6622055342a734c663ee0b61ff4fd697f42467595df0bf4448c8c170'),
    'merges.txt': (456318, '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),
    'model.safetensors': (3247159078, '5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce'),
    'tokenizer.json': (1355256, '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'),
    'tokenizer_config.json': (26, '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f'),
    'vocab.json': (1042301, '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'),
}
REVISION = '32b71b12589c2f8d625668d2335a01cac3249519'


def streaming_generation_nll(model, tokenizer, texts, batch_size, projection_chunk):
    """Same per-document generated-token NLL with a bounded LM-head workspace."""
    import torch.nn.functional as F
    if not tokenizer.is_fast:
        raise ValueError('streamed scorer requires the pinned fast tokenizer')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    all_nll, all_count = [], []
    for offset in range(0, len(texts), batch_size):
        part = texts[offset:offset + batch_size]
        enc = tokenizer(part, return_tensors='pt', padding=True, truncation=True,
                        max_length=1024, return_offsets_mapping=True).to(device)
        ids, mask = enc['input_ids'], enc['attention_mask']
        offsets = enc['offset_mapping'].cpu()
        mask_cpu = mask.cpu()
        real = (mask_cpu == 1) & ~((offsets[:, :, 0] == 0) & (offsets[:, :, 1] == 0))
        # All prompts are empty in the TinyStories official protocol.
        is_generated = real & (offsets[:, :, 0] >= 0)
        no_gen = ~is_generated.any(dim=1)
        first = is_generated.long().argmax(dim=1)
        first[no_gen] = ids.size(1)
        start = (first - 1).clamp(min=0).to(device)
        positions = torch.arange(ids.size(1) - 1, device=device).unsqueeze(0)
        include = mask[:, 1:].bool() & (positions >= start.unsqueeze(1))
        hidden = model.transformer(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        for row in range(0, len(part), projection_chunk):
            last = min(row + projection_chunk, len(part))
            logits = model.lm_head(hidden[row:last, :-1, :])
            labels = ids[row:last, 1:]
            nll = -F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            keep = include[row:last]
            all_nll.append((nll * keep.float()).sum(dim=1).detach().cpu())
            all_count.append(keep.sum(dim=1).to(torch.long).detach().cpu())
    return torch.cat(all_nll), torch.cat(all_count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--batch', type=int, default=32)
    checks = p.add_mutually_exclusive_group()
    checks.add_argument('--check-only', action='store_true')
    checks.add_argument('--entropy-only', action='store_true')
    checks.add_argument('--qualify-stream', action='store_true')
    p.add_argument('--stream-output', action='store_true')
    p.add_argument('--projection-chunk', type=int, default=4)
    p.add_argument('samples', type=Path, nargs='+')
    args = p.parse_args()
    if not 1 <= args.batch <= 32 or not 1 <= args.projection_chunk <= 8 or (not (args.check_only or args.entropy_only) and not torch.cuda.is_available()):
        raise ValueError('invalid evaluator batch or CUDA unavailable')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if rev != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, digest) in MODEL_FILES.items():
        verify_input(args.model / name, size, digest)
    payload = args.tokenizer.read_bytes()
    if hashlib.sha256(payload).hexdigest() != 'fdf961a269fa4506a72d12056277450aaed13d31392b39fd640ff6db546516b4':
        raise ValueError('tokenizer payload differs')
    data = json.loads(payload)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from tokenization.bpe_tokenizer import BPETokenizer
    from evaluation.gen_ppl import generation_perplexity
    bpe = BPETokenizer({i: bytes.fromhex(v) for i, v in enumerate(data['vocab_hex'])},
                       [tuple(bytes.fromhex(x) for x in pair) for pair in data['merges_hex']],
                       {'eos_token': '<|endoftext|>', 'pad_token': '<|pad|>'})
    samples = []
    for path in args.samples:
        raw = path.read_bytes()
        item = json.loads(raw)
        texts, ids, toks, codes = item['texts'], item['raw_ids'], item['toks'], item['codes']
        if not 1 <= len(texts) <= 256 or not len(texts) == len(ids) == len(toks) == len(codes):
            raise ValueError(f'invalid sample count: {path}')
        for text, row, tokens, code in zip(texts, ids, toks, codes):
            if not isinstance(text, str) or len(row) != 128 or len(code) != 641 or any(type(x) != int or not 0 <= x < 8192 for x in row) or any(type(x) != int or not 0 <= x < 4096 for x in code):
                raise ValueError(f'invalid sample shape: {path}')
            end = row.index(0) if 0 in row else len(row)
            expected = bpe.decode([x for x in row[:end] if x not in (0, 1)], skip_special_tokens=True)
            if expected != text or bpe.encode(text) != tokens:
                raise ValueError(f'tokenizer parity mismatch: {path}')
        samples.append((path, hashlib.sha256(raw).hexdigest(), item))
    if args.check_only:
        print(json.dumps({'validated_samples': len(samples), 'tokenizer_parity': True, 'model_files_verified': True}))
        return
    if args.entropy_only:
        scores = []
        for _, digest, item in samples:
            values = []
            for tokens in item['toks']:
                if tokens:
                    counts = Counter(tokens)
                    values.append(-sum((v / len(tokens)) * math.log(v / len(tokens)) for v in counts.values()))
            if not values:
                raise ValueError('entropy undefined for empty samples')
            scores.append({'sample_sha256': digest, 'seed': item['seed'], 'count': len(item['texts']),
                           'nonempty_count': len(values), 'entropy': float(np.mean(values))})
        print(json.dumps({'scores': scores, 'metric': 'upstream eval_merge entropy', 'gen_ppl': None}, sort_keys=True))
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, use_safetensors=True).cuda().eval()
    if args.qualify_stream:
        differences = []
        with torch.inference_mode():
            for _, digest, item in samples:
                texts = item['texts'][:8]
                ref = generation_perplexity(model, tokenizer, [''] * len(texts), texts, batch_size=8)
                nll, count = streaming_generation_nll(model, tokenizer, texts, 8, args.projection_chunk)
                if not torch.equal(count, ref.token_count):
                    raise ValueError('streamed scorer token counts differ')
                delta = (nll - ref.nll_sum).abs() / count.clamp(min=1)
                maximum = float(delta.max())
                if not math.isfinite(maximum) or maximum > 1e-4:
                    raise ValueError(f'streamed scorer exceeds declared NLL tolerance: {maximum}')
                differences.append({'sample_sha256': digest, 'count': len(texts), 'max_abs_mean_nll_difference': maximum})
        print(json.dumps({'qualification': differences, 'token_counts_exact': True, 'max_allowed_abs_mean_nll_difference': 1e-4,
                          'batch': 8, 'projection_chunk': args.projection_chunk, 'gpu': torch.cuda.get_device_name(0)}, sort_keys=True))
        return
    scores = []
    for path, digest, item in samples:
        texts = item['texts']
        with torch.inference_mode():
            if args.stream_output:
                nll, count = streaming_generation_nll(model, tokenizer, texts, args.batch, args.projection_chunk)
                ppl = math.exp(float(nll.sum()) / max(int(count.sum()), 1))
            else:
                ppl = float(generation_perplexity(model, tokenizer, [''] * len(texts), texts, batch_size=args.batch).mean_ppl)
        entropies = []
        for tokens in item['toks']:
            if tokens:
                counts = Counter(tokens)
                entropies.append(-sum((v / len(tokens)) * math.log(v / len(tokens)) for v in counts.values()))
        entropy = float(np.mean(entropies)) if entropies else float('nan')
        if not math.isfinite(ppl) or not math.isfinite(entropy):
            raise ValueError('nonfinite quality metric')
        scores.append({'sample_sha256': digest, 'seed': item['seed'], 'count': len(texts), 'gen_ppl': ppl, 'entropy': entropy})
    print(json.dumps({'scores': scores, 'upstream': UPSTREAM, 'evaluator_model_revision': REVISION,
                      'transformers': __import__('transformers').__version__, 'torch': torch.__version__,
                      'cuda_device': torch.cuda.get_device_name(0), 'evaluator_batch': args.batch,
                      'metric': 'upstream evaluation.gen_ppl.generation_perplexity and eval_merge entropy',
                      'official_evaluator_batch': args.batch == 32, 'streamed_output': args.stream_output,
                      'projection_chunk': args.projection_chunk if args.stream_output else None}, sort_keys=True))


if __name__ == '__main__':
    main()
