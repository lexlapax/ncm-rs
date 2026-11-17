"""Time pinned LM1B-small development outputs; not a release gate."""
import argparse
import ast
import json
import time
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from lm1b_inventory import CHECKPOINTS
from tinystories_artifact import ROOT, UPSTREAM, digest_file, json_file, verify_input

CONFIGS = {
    'small': (2749, '4db04299a87f7eb1627de0abb2f0b257e50e5ea0138edda771ad287cdabd7edf'),
    'medium': (2754, '93a42e0923b38f06f186072fdc37c1a2b4db5298c7577755390dd36c651d8142'),
    'large': (2753, '4410e3710360cae162f8be4186a44a0c7c473be674a1ac3d1d940bfe45d5e70c'),
}
EVALUATOR_SHA = 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--size', choices=tuple(CONFIGS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(11, 12, 13, 14, 15), default=11)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.output.is_symlink() or output.exists() or not output.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be a new file under reports')
    upstream = args.upstream.resolve()
    rev = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != UPSTREAM or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('upstream checkout must be clean and pinned')
    paths = {f'{args.size}/generator.pt': CHECKPOINTS[f'{args.size}/generator.pt'],
             'tokenizer.pt': CHECKPOINTS['tokenizer.pt'], f'{args.size}/config.yaml': CONFIGS[args.size]}
    for name, (size, sha) in paths.items():
        verify_input(args.source / name, size, sha)
    evaluator = upstream / 'scripts/eval_shard.py'
    if digest_file(evaluator) != EVALUATOR_SHA:
        raise ValueError('unexpected evaluator')
    source_files = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_inventory.py').resolve(),
                    Path(__file__).with_name('tinystories_artifact.py').resolve(), evaluator]
    source_files += sorted((upstream / 'models').glob('*.py'))
    before = {str(path): digest_file(path) for path in source_files}
    cfg = OmegaConf.create(yaml.safe_load((args.source / args.size / 'config.yaml').read_bytes()))
    gen = cfg.generation
    if (cfg.task != f'lm1b-{args.size}' or gen.temperature != 1.0 or gen.top_k != 0 or
            gen.top_p != 1.0 or gen.cfg_scale != 0.0 or gen.frequency_penalty != 11.5 or
            gen.frequency_deficit != 2.0 or str(gen.adjacency_penalty) != '10,4,0.5'):
        raise ValueError('unsupported generation configuration')
    tree = ast.parse(evaluator.read_bytes())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'pick_tokens']
    if len(funcs) != 1:
        raise ValueError('unexpected evaluator function')
    decoder = compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(evaluator), 'exec')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    torch.set_num_threads(1)
    codec_state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    generator_state = torch.load(args.source / args.size / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(codec_state['model'], strict=True)
    codec.cuda().eval()
    model = NextConceptTransformer(codec, cfg.model)
    model.load_state_dict(generator_state['model'], strict=True)
    model.cuda().eval()
    inference = generator_state['inference']
    scope = {'torch': torch, 'freq': 11.5, 'mu': 2.0, 'ADJ': [10.0, 4.0, 0.5],
             'ALLOW': inference['token_unigram'].double().cuda()}
    exec(decoder, scope)
    if args.seed in set(cfg.eval.seeds) or cfg.eval.batch_size != 8 or cfg.eval.n_samples != 256:
        raise ValueError('development seed or evaluation settings differ')
    reference_path = ROOT / 'reports/phase4-lm1b-small-dev-seed-11-v1.json'
    reference_sha = '123ce052e32ebac1981321a72d728c7b16c876be5c8a118dd4668b0dc917760a'
    if args.seed != 11 or digest_file(reference_path) != reference_sha:
        raise ValueError('only checked development seed 11 is supported')
    reference = json.loads(reference_path.read_bytes())
    if (reference['size'], reference['seed'], reference['count'], reference['batch_size']) != ('small', 11, 256, 8):
        raise ValueError('unexpected development reference')
    def generate():
        level0 = torch.multinomial(inference['level0_unigram'].float().cpu(), 8, replacement=True).view(8, 1).cuda()
        with torch.no_grad():
            codes = model.generate(level0, torch.full((8, 1), 101, device='cuda', dtype=torch.long),
                                   temperature=1.0, top_k=0, top_p=1.0, use_kv_cache=False,
                                   cfg_scale=0.0, cfg_null_eos=101)
            _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            ids = scope['pick_tokens'](codec.decoder(levels[-1]))
        return codes.tolist(), ids.tolist()
    torch.manual_seed(11)
    for _ in range(2):
        generate()
    torch.cuda.synchronize()
    torch.manual_seed(11)
    times = []
    for batch in range(8):
        torch.cuda.synchronize()
        started = time.perf_counter()
        codes, ids = generate()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        if codes != reference['codes'][batch * 8:(batch + 1) * 8] or ids != reference['raw_ids'][batch * 8:(batch + 1) * 8]:
            raise ValueError(f'output differs from development capture at batch {batch}')
    if digest_file(reference_path) != reference_sha:
        raise ValueError('development capture changed')
    for name, (size, sha) in paths.items():
        verify_input(args.source / name, size, sha)
    if before != {str(p): digest_file(p) for p in source_files}:
        raise ValueError('probe source changed')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('upstream checkout changed')
    status = Path('/proc/self/status').read_text()
    hwm = next(line for line in status.splitlines() if line.startswith('VmHWM:'))
    json_file(output, {'scope': 'LM1B development timing; not release gate', 'runtime': 'python',
                       'batch': 8, 'warmups': 2, 'trials': 8, 'seconds': times, 'output_matches_capture': True,
                       'vmhwm': hwm, 'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'upstream': rev, 'reference_sha256': reference_sha, 'source_sha256': before,
                       'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
    print(json.dumps({'output_sha256': digest_file(output), 'times': times}, sort_keys=True))


if __name__ == '__main__':
    main()
