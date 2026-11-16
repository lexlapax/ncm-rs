"""Generate pinned-upstream XSum quality batches with resumable RNG state."""
import argparse
import ast
import base64
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from xsum_artifact import INPUTS, UPSTREAM, digest_file, verify_input

DATA_SHA = 'cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a'
EVALUATOR_SHA = 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a'
ROOT = Path(__file__).resolve().parents[2]


def encoded_state(value):
    return base64.b64encode(bytes(value.tolist())).decode('ascii')


def restore_state(value):
    return torch.tensor(list(base64.b64decode(value, validate=True)), dtype=torch.uint8)


def validate_record(value, expected, ordinal):
    if type(value) is not dict or value.get('ordinal') != ordinal or value.get('indices') != expected:
        raise ValueError('quality batch identity differs')
    for key, width, bound in [('codes', 200, 4096), ('raw_ids', 48, 50304)]:
        rows = value.get(key)
        if type(rows) is not list or len(rows) != len(expected):
            raise ValueError('quality batch row count differs')
        if any(type(row) is not list or len(row) != width or
               any(type(x) is not int or not 0 <= x < bound for x in row) for row in rows):
            raise ValueError('quality batch token shape differs')
    for key in ('cpu_rng', 'cuda_rng'):
        raw = base64.b64decode(value[key], validate=True)
        if not 1 <= len(raw) <= 65536:
            raise ValueError('invalid RNG state')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def read_record(path, expected, ordinal):
    if path.is_symlink() or not path.is_file():
        raise ValueError('quality batch must be a regular file')
    envelope = json.loads(path.read_bytes())
    if type(envelope) is not dict or set(envelope) != {'record', 'sha256'}:
        raise ValueError('invalid quality batch envelope')
    record = envelope['record']
    if hashlib.sha256(canonical(record)).hexdigest() != envelope['sha256']:
        raise ValueError('quality batch receipt differs')
    validate_record(record, expected, ordinal)
    return record


def shard_batches(shard, stop):
    if shard not in (0, 1) or not 1 <= stop <= 11334:
        raise ValueError('invalid shard or stop')
    full = list(range(shard, 11334, 2))
    batches = [full[i:i + 32] for i in range(0, len(full), 32)]
    return batches, [(i, batch) for i, batch in enumerate(batches) if batch[-1] < stop]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--shard', type=int, choices=(0, 1), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--stop', type=int, default=11334)
    args = ap.parse_args()
    forbidden = ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES',
                 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG',
                 'NVIDIA_TF32_OVERRIDE', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
    if any(os.environ.get(name) for name in forbidden):
        raise ValueError('injection or device-override environment is unsupported')
    if args.output.is_symlink():
        raise ValueError('quality output cannot be a symlink')
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / 'reports').resolve()) or not 1 <= args.stop <= 11334:
        raise ValueError('output must be under reports and stop must be 1..11334')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one CUDA device required')
    gpu_lines = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'], text=True
    ).strip().splitlines()
    if len(gpu_lines) != 1:
        raise ValueError('GPU identity is ambiguous')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if rev != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if args.data.is_symlink() or digest_file(args.data) != DATA_SHA:
        raise ValueError('unexpected converted XSum dataset')
    data = json.loads(args.data.read_bytes())
    if data.get('profile') != 'xsum' or len(data.get('inputs', [])) != 11334:
        raise ValueError('unexpected dataset shape')
    evaluator = args.upstream / 'scripts/eval_shard.py'
    if digest_file(evaluator) != EVALUATOR_SHA:
        raise ValueError('untrusted evaluator source')
    tree = ast.parse(evaluator.read_bytes())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'pick_tokens']
    if len(funcs) != 1:
        raise ValueError('expected one decoder function')
    decoder = compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(evaluator), 'exec')
    manifest = {'schema_version': 1, 'upstream': rev, 'data_sha256': DATA_SHA,
                'sources': {k: v[1] for k, v in INPUTS.items()}, 'evaluator_sha256': EVALUATOR_SHA,
                'generator_sha256': digest_file(Path(__file__)), 'seed': args.seed, 'shard': args.shard,
                'shards': 2, 'batch_size': 32, 'temperature': 1.0, 'top_k': 1, 'top_p': 0.0,
                'cfg': 0.0, 'tf32': False, 'sdpa': 'math', 'between_batch_gc': True, 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
                'gpu_identity_and_driver': gpu_lines[0].strip()}
    if out.exists():
        metadata = out / 'manifest.json'
        if out.is_symlink() or not out.is_dir() or metadata.is_symlink() or not metadata.is_file() or json.loads(metadata.read_bytes()) != manifest:
            raise ValueError('quality output manifest differs')
    else:
        out.mkdir(parents=True)
        with (out / 'manifest.json').open('x') as f:
            json.dump(manifest, f, sort_keys=True, indent=2); f.write('\n')
    # Stop is an input-index boundary. Only complete original batches are run.
    batches, selected = shard_batches(args.shard, args.stop)
    existing = sorted(out.glob('batch-*.json'))
    if [p.name for p in existing] != [f'batch-{i:05}.json' for i in range(len(existing))]:
        raise ValueError('quality batch sequence has gaps')
    for i, path in enumerate(existing):
        if i >= len(batches): raise ValueError('too many output batches')
        read_record(path, batches[i], i)
    if len(existing) > len(selected):
        raise ValueError('stop precedes completed batches')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    codec_state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    generator_state = torch.load(args.source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(codec_state['model'], strict=True)
    codec = codec.cuda().eval()
    model = NextConceptTransformer(codec, cfg.model).cuda()
    model.load_state_dict(generator_state['model'], strict=True)
    model.eval()
    scope = {'torch': torch, 'freq': 2.0, 'mu': 1.0, 'ADJ': [5.0, 4.0, 0.5],
             'ALLOW': generator_state['inference']['token_unigram'].cuda().double()}
    exec(decoder, scope)
    del codec_state, generator_state
    torch.manual_seed(args.seed)
    if existing:
        previous = read_record(existing[-1], batches[len(existing)-1], len(existing)-1)
        torch.set_rng_state(restore_state(previous['cpu_rng']))
        torch.cuda.set_rng_state(restore_state(previous['cuda_rng']))
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for ordinal, indices in selected[len(existing):]:
            rows = [data['inputs'][i] for i in indices]
            prefix = [(row[:512] + [50257] * 512)[:512] for row in rows]
            x = torch.tensor(prefix, device='cuda', dtype=torch.long)
            codes = model.generate(None, x, temperature=1.0, top_k=1, top_p=0.0,
                                   use_kv_cache=False, cfg_scale=0.0)
            _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            ids = scope['pick_tokens'](codec.decoder(levels[-1]))
            record = {'ordinal': ordinal, 'indices': indices, 'codes': codes.tolist(), 'raw_ids': ids.tolist(),
                      'cpu_rng': encoded_state(torch.get_rng_state()), 'cuda_rng': encoded_state(torch.cuda.get_rng_state())}
            validate_record(record, indices, ordinal)
            envelope = {'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}
            raw = canonical(envelope) + b'\n'
            path = out / f'batch-{ordinal:05}.json'
            temp = out / f'batch-{ordinal:05}.{os.getpid()}.part'
            with temp.open('xb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            # A hard-link publication cannot overwrite a concurrent result;
            # an interrupted .part remains separate from completed batches.
            os.link(temp, path)
            temp.unlink()
            directory = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            print(json.dumps({'shard': args.shard, 'ordinal': ordinal, 'rows': len(indices)}), flush=True)
            del x, codes, levels, ids, _
            gc.collect()
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if digest_file(args.data) != DATA_SHA or digest_file(evaluator) != EVALUATOR_SHA:
        raise ValueError('quality sources changed')


if __name__ == '__main__':
    main()
