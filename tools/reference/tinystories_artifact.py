"""Export the pinned TinyStories inference and training tensors as checked data.

This is an explicit offline profile, not a general checkpoint importer. Source
and destination parents must be trusted and not concurrently mutated.
"""
import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

UPSTREAM = 'c70274273f489abc73832b2b98584a8483b6b203'
REVISION = 'd342ce8aaea58830eef3b8ad473664673e41f692'
INPUTS = {
    'generator.pt': (356160651, '0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff'),
    'tokenizer.pt': (221861835, '76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81'),
    'config.yaml': (2483, '4009baccad5a159bfc12cd48f3da17c22c4a281b86f71f2fc70b7c1bbb62004b'),
    'bpe_vocab_8192.pickle': (95122, 'fb176b4c68108e4b5ef61b9ff2af3731600124e3e160970a6bd78cd0a5d3ff6c'),
    'bpe_merges_8192.pickle': (108277, '0d4cab13f2e7b684b75eaa290a52e596ba87744ef13d943cf3a50a5a8e080ec1'),
}
UPSTREAM_15K_EMA = '094f25ca5fd8fe2bc7f0ff77bf3f860c3a34e7e8'
REVISION_15K_EMA = 'e34c3fab451ddbe7cf474cc97c44ab23321f2219'
INPUTS_15K_EMA = {
    'generator.pt': (356150923, '9d66ab61de9223a5229175fea730747d03acdb1794a37038e7a0d771a45ca875'),
    'tokenizer.pt': (221863115, '85ce688e0161f6e1a785d549e57d8da65ea67a40543a8fc7a67934b68550eac5'),
    'config.yaml': (2242, 'efdd98b675a24f956e9b832d03a2a2e88db05931d434090ef65f0f34bcb2cd9c'),
    'bpe_vocab_8192.pickle': (95122, 'fb176b4c68108e4b5ef61b9ff2af3731600124e3e160970a6bd78cd0a5d3ff6c'),
    'bpe_merges_8192.pickle': (108277, '0d4cab13f2e7b684b75eaa290a52e596ba87744ef13d943cf3a50a5a8e080ec1'),
}
TRAINING = re.compile(r'vq\.(ema_cluster_size|ema_w|codebook_hit)_\d+$')
ROOT = Path(__file__).resolve().parents[2]


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_input(path, size, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size or digest_file(path) != expected:
        raise ValueError(f'input type, size or SHA-256 differs: {path.name}')


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_map(value, expected_count):
    if not isinstance(value, dict) or len(value) != expected_count:
        raise ValueError('checkpoint tensor inventory count differs')
    addresses = []
    for name, tensor in value.items():
        if (not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                or tensor.dtype not in (torch.float32, torch.int64)
                or tensor.layout != torch.strided or not tensor.is_contiguous()):
            raise ValueError(f'unsupported checkpoint tensor: {name}')
        addresses.append(tensor.untyped_storage().data_ptr())
    if len(set(addresses)) != len(addresses):
        raise ValueError('unexpected shared tensor storage')
    return value


def checkpoint_parts(generator, tokenizer):
    if not isinstance(generator, dict) or set(generator) != {'model', 'inference'}:
        raise ValueError('unexpected generator checkpoint metadata')
    if not isinstance(tokenizer, dict) or set(tokenizer) not in ({'model'}, {'model', 'inference'}):
        raise ValueError('unexpected tokenizer checkpoint metadata')
    model = tensor_map(generator['model'], 197)
    codec = tensor_map(tokenizer['model'], 106)
    gen_inf = generator['inference']
    if not isinstance(gen_inf, dict) or set(gen_inf) not in ({'level0_unigram'}, {'level0_unigram', 'config'}):
        raise ValueError('unexpected inference metadata')
    inference = tensor_map({'level0_unigram': gen_inf['level0_unigram']}, 1)
    unigram = inference['level0_unigram']
    if (unigram.dtype != torch.float32 or tuple(unigram.shape) != (4096,)
            or not bool(torch.isfinite(unigram).all()) or bool((unigram < 0).any())
            or not math.isclose(float(unigram.double().sum()), 1.0, rel_tol=0, abs_tol=1e-6)):
        raise ValueError('invalid level-zero unigram distribution')
    runtime = {k: v for k, v in codec.items() if not TRAINING.fullmatch(k)}
    training = {k: v for k, v in codec.items() if TRAINING.fullmatch(k)}
    if len(runtime) != 70 or len(training) != 36:
        raise ValueError('unexpected codec training/runtime classification')
    return {'generator.safetensors': model, 'codec.safetensors': runtime,
            'training.safetensors': training, 'inference.safetensors': inference}


def json_file(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')


def export(source, tokenizer_asset, output, release='v1'):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if release == 'v1':
        upstream_commit, revision, inputs = UPSTREAM, REVISION, INPUTS
    elif release == '15k-ema':
        upstream_commit, revision, inputs = UPSTREAM_15K_EMA, REVISION_15K_EMA, INPUTS_15K_EMA
    else:
        raise ValueError(f'unsupported TinyStories release selector: {release}')
    for name, (size, expected) in inputs.items():
        verify_input(source / name, size, expected)
    lock = json.loads((ROOT / 'tools/reference/tokenizers-v1.lock.json').read_bytes())['profiles']['tinystories']
    tokenizer_manifest = tokenizer_asset / 'manifest.json'
    payload, license_file = tokenizer_asset / 'tokenizer.json', tokenizer_asset / 'LICENSE'
    metadata = json.loads(tokenizer_manifest.read_bytes())
    verify_input(tokenizer_manifest, tokenizer_manifest.stat().st_size, lock['manifest_sha256'])
    verify_input(payload, metadata['payload_bytes'], lock['payload_sha256'])
    verify_input(license_file, license_file.stat().st_size, metadata['license_sha256'])
    if metadata['profile'] != 'tinystories' or metadata['kind'] != 'byte_bpe':
        raise ValueError('wrong canonical tokenizer profile')
    config = yaml.safe_load((source / 'config.yaml').read_bytes())
    if config['task'] != 'tinystories':
        raise ValueError('wrong checkpoint task')
    generator = torch.load(source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    tokenizer = torch.load(source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    bundles = checkpoint_parts(generator, tokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.ncm-tinystories-', dir=output.parent))
    try:
        inventory = {}
        for filename, state in bundles.items():
            save_file(dict(state), temporary / filename)
            restored = load_file(temporary / filename)
            if set(restored) != set(state):
                raise ValueError(f'roundtrip tensor names changed: {filename}')
            entries = []
            for name in sorted(state):
                original, loaded = state[name], restored[name]
                raw = tensor_bytes(original)
                if original.dtype != loaded.dtype or original.shape != loaded.shape or raw != tensor_bytes(loaded):
                    raise ValueError(f'non-bit-exact tensor roundtrip: {filename}/{name}')
                entries.append({'name': name, 'shape': list(original.shape),
                                'dtype': 'F32' if original.dtype == torch.float32 else 'I64',
                                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            inventory[filename] = entries
            del restored
        json_file(temporary / 'config.json', config)
        shutil.copyfile(payload, temporary / 'tokenizer.json')
        shutil.copyfile(tokenizer_manifest, temporary / 'tokenizer-manifest.json')
        shutil.copyfile(license_file, temporary / 'tokenizer.LICENSE')
        json_file(temporary / 'provenance.json', {
            'converter_sha256': digest_file(Path(__file__)),
            'converter_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'source_sha256': {name: sha for name, (_, sha) in inputs.items()},
            'source_loading': 'torch.load(weights_only=True, mmap=True); no unrestricted retry',
            'code_license': 'MIT', 'weights_license': 'MIT', 'aliases': [],
            'inference_metadata': ['level0_unigram'],
            'training_state': 'EMA/count/hit buffers separated, not dropped',
            'serialization': 'safetensors; unchanged upstream layouts/dtypes',
        })
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in sorted(temporary.iterdir())}
        json_file(temporary / 'manifest.json', {
            'schema_version': 1, 'profile': 'tinystories-inference-v1', 'upstream_commit': upstream_commit,
            'checkpoint_repository': 'hazyresearch/ncm-checkpoints', 'checkpoint_revision': revision,
            'files': files, 'tensors': inventory,
        })
        for name, (size, expected) in inputs.items():
            verify_input(source / name, size, expected)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'profile': 'tinystories-inference-v1',
            'bundle_tensor_counts': {k: len(v) for k, v in bundles.items()},
            'manifest_sha256': digest_file(output / 'manifest.json'), 'bit_exact_roundtrip': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--release', choices=('v1', '15k-ema'), default='v1')
    args = parser.parse_args()
    if args.report.exists() or args.report.is_symlink():
        parser.error('report already exists')
    report = export(args.source, args.tokenizer_asset, args.output, release=args.release)
    json_file(args.report, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
