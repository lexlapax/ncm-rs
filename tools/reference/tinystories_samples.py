"""Generate TinyStories samples from pinned Python, using non-executable BPE assets."""
import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from tinystories_artifact import INPUTS, UPSTREAM, verify_input


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    args = p.parse_args()
    if not 1 <= args.count <= 256 or not 1 <= args.batch <= 128 or not torch.cuda.is_available():
        raise ValueError('invalid sample count, batch or CUDA unavailable')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if rev != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, digest) in INPUTS.items():
        verify_input(args.source / name, size, digest)
    payload = args.tokenizer.read_bytes()
    if hashlib.sha256(payload).hexdigest() != 'fdf961a269fa4506a72d12056277450aaed13d31392b39fd640ff6db546516b4':
        raise ValueError('tokenizer payload differs')
    token_data = json.loads(payload)
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    from tokenization.bpe_tokenizer import BPETokenizer
    tokenizer = BPETokenizer({i: bytes.fromhex(v) for i, v in enumerate(token_data['vocab_hex'])},
                             [tuple(bytes.fromhex(x) for x in pair) for pair in token_data['merges_hex']],
                             cfg.tokenizer.pre_tokenizer.special_tokens)
    codec_data = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    model_data = torch.load(args.source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with contextlib.redirect_stdout(sys.stderr):
        codec = VQVAE(**cfg.tokenizer.vqvae)
        codec.load_state_dict(codec_data['model'], strict=True)
        codec = codec.cuda().eval()
        model = NextConceptTransformer(codec, cfg.model).cuda()
        model.load_state_dict(model_data['model'], strict=True)
        model.eval()
    unigram = model_data['inference']['level0_unigram']
    all_codes, all_ids, texts = [], [], []
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        for offset in range(0, args.count, args.batch):
            batch = min(args.batch, args.count - offset)
            initial = torch.multinomial(unigram, batch, replacement=True).reshape(batch, 1).cuda()
            prefix = torch.zeros((batch, 1), dtype=torch.long, device='cuda')
            codes = model.generate(initial, prefix, temperature=1.0, top_k=0, top_p=1.0,
                                   use_kv_cache=False, cfg_scale=0.0)
            _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            ids = codec.decoder(levels[-1]).argmax(-1).cpu().tolist()
            all_codes.extend(codes.cpu().tolist())
            all_ids.extend(ids)
            for row in ids:
                end = row.index(0) if 0 in row else len(row)
                texts.append(tokenizer.decode([i for i in row[:end] if i not in (0, 1)], skip_special_tokens=True))
    report = {'seed': args.seed, 'batch': args.batch, 'rng': 'PyTorch multinomial',
              'codes': all_codes, 'raw_ids': all_ids, 'texts': texts,
              'toks': [tokenizer.encode(text) for text in texts]}
    print(json.dumps(report, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
