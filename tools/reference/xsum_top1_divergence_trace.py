"""Capture pinned CUDA teacher traces for the two full-corpus Metal ID divergences."""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from xsum_artifact import INPUTS, UPSTREAM, verify_input, digest_file
from xsum_quality_generate import DATA_SHA

ROOT = Path(__file__).resolve().parents[2]
SCORED_SHA = "9e8c211611890833aab2cd4582ffbce13decdac26de92b53e5eab9f86fc3f76d"
ROWS = (4883, 4937)


def select_rows(data, scored):
    if data.get("profile") != "xsum" or len(data.get("inputs", [])) != 11334 or len(scored.get("rows", [])) != 11334:
        raise ValueError("unexpected XSum input or reference rows")
    prefixes, codes, ids = [], [], []
    for index in ROWS:
        row, ref = data["inputs"][index], scored["rows"][index]
        if (type(row) is not list or len(row) != 560 or ref.get("index") != index
                or len(ref.get("codes", [])) != 200 or len(ref.get("raw_ids", [])) != 48):
            raise ValueError("invalid selected row")
        if any(type(v) is not int or not 0 <= v < 50304 for v in row + ref["raw_ids"]) or any(
                type(v) is not int or not 0 <= v < 4096 for v in ref["codes"]):
            raise ValueError("invalid selected token")
        prefixes.append((row[:512] + [50257] * 512)[:512])
        codes.append(ref["codes"]); ids.append(ref["raw_ids"])
    return prefixes, codes, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not torch.cuda.is_available():
        raise ValueError('new output and CUDA required')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if rev != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    data_path = ROOT / 'artifacts/canonical/text-phase4-v1/xsum.json'
    score_path = ROOT / 'reports/phase4-xsum-quality-python-full-rouge-v1.json'
    if digest_file(data_path) != DATA_SHA or digest_file(score_path) != SCORED_SHA:
        raise ValueError('pinned dataset or scored reference differs')
    prefixes, saved_codes, saved_ids = select_rows(json.loads(data_path.read_bytes()), json.loads(score_path.read_bytes()))
    source_sha = digest_file(Path(__file__))
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    if cfg.task != 'xsum':
        raise ValueError('incorrect task')
    evaluator = args.upstream / 'scripts/eval_shard.py'
    evaluator_hash = 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a'
    if digest_file(evaluator) != evaluator_hash:
        raise ValueError('untrusted evaluator source')
    tree = ast.parse(evaluator.read_bytes())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'pick_tokens']
    if len(funcs) != 1:
        raise ValueError('expected one decoder function')
    decoder_code = compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(evaluator), 'exec')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    codec_state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    generator_state = torch.load(args.source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(codec_state['model'], strict=True)
    codec = codec.cuda().eval()
    model = NextConceptTransformer(codec, cfg.model).cuda()
    model.load_state_dict(generator_state['model'], strict=True)
    model.eval()
    traces = {}
    def save(name, value):
        traces[name] = value.detach().cpu().contiguous().clone()
    levels = list(codec.vq.levels)
    batch = 2
    codes = torch.tensor(saved_codes, dtype=torch.long, device='cuda')
    forced = list(torch.split(codes, levels, dim=1))
    prefix = torch.tensor(prefixes, dtype=torch.long, device='cuda')
    calls = [0]
    def forced_sample(logits, temperature, top_k, top_p):
        level = calls[0]
        save(f'generator.level_{level:02d}.logits', logits)
        if tuple(logits.shape) != (batch, levels[level], 4096):
            raise ValueError('unexpected generator head shape')
        calls[0] += 1
        return forced[level]
    model._sample_level_raw = forced_sample
    scope = {'torch': torch, 'freq': 2.0, 'mu': 1.0, 'ADJ': [5.0, 4.0, 0.5],
             'ALLOW': generator_state['inference']['token_unigram'].cuda().double()}
    exec(decoder_code, scope)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        save('generator.prefix_ids', prefix)
        generated = model.generate(None, prefix, temperature=1.0, top_k=1, top_p=0.0,
                                   use_kv_cache=False, cfg_scale=0.0)
        if calls[0] != len(levels) or not torch.equal(generated, torch.cat(forced, dim=1)):
            raise ValueError('forced generator codes changed')
        save('generator.forced_codes', generated)
        z_hat = torch.zeros(batch, levels[-1], codec.embed_dim, device='cuda')
        for level, codes in enumerate(forced):
            z_hat, _ = codec.vq.get_next_autoregressive_input(level, z_hat, codec.vq.embeddings[level](codes))
            save(f'reconstruction.level_{level:02d}', z_hat)
        final = codec.decoder(z_hat)
        save('generator.final_logits', final)
        save('generator.greedy_ids', final.argmax(-1))
        penalized = scope['pick_tokens'](final)
        save('generator.penalized_ids', penalized)
        reference_id_differences = int((penalized.cpu() != torch.tensor(saved_ids)).sum())
        torch.cuda.synchronize()
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if (digest_file(evaluator) != evaluator_hash or digest_file(data_path) != DATA_SHA
            or digest_file(score_path) != SCORED_SHA or digest_file(Path(__file__)) != source_sha):
        raise ValueError('source, data or reference changed')
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(traces, str(args.output / 'trace.safetensors'))
    report = {'upstream': rev, 'sources': {k: v[1] for k, v in INPUTS.items()},
              'torch': torch.__version__, 'cuda_device': torch.cuda.get_device_name(0),
              'tf32': False, 'sdpa': 'math', 'batch': batch, 'rows': list(ROWS),
              'data_sha256': DATA_SHA, 'scored_reference_sha256': SCORED_SHA, 'script_sha256': source_sha,
              'penalized_id_differences_vs_saved_python': reference_id_differences,
              'mode': 'teacher-forced fixed code indices; generator sampling not evaluated',
              'trace_sha256': digest_file(args.output / 'trace.safetensors'), 'tensors': sorted(traces)}
    with (args.output / 'report.json').open('x') as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write('\n')


if __name__ == '__main__':
    main()
