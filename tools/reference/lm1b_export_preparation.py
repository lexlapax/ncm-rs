"""Lossless CPU LM1B checkpoint preparation; not a runtime artifact."""
import argparse
import json
import math
import platform
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from lm1b_inventory import CHECKPOINTS
from tinystories_artifact import ROOT, REVISION, UPSTREAM, digest_file, json_file, tensor_map, verify_input
from wmt14_export_preparation import save_checked

CONFIGS = {
    'small': (2749, '4db04299a87f7eb1627de0abb2f0b257e50e5ea0138edda771ad287cdabd7edf'),
    'medium': (2754, '93a42e0923b38f06f186072fdc37c1a2b4db5298c7577755390dd36c651d8142'),
    'large': (2753, '4410e3710360cae162f8be4186a44a0c7c473be674a1ac3d1d940bfe45d5e70c'),
}
MODEL_COUNTS = {'small': 213, 'medium': 381, 'large': 549}


def checkpoint_parts(generator, tokenizer, size):
    if size not in MODEL_COUNTS or not isinstance(generator, dict) or set(generator) != {'model', 'inference'}:
        raise ValueError('unexpected generator checkpoint')
    if not isinstance(tokenizer, dict) or set(tokenizer) != {'model'}:
        raise ValueError('unexpected tokenizer checkpoint')
    model = tensor_map(generator['model'], MODEL_COUNTS[size])
    codec = tensor_map(tokenizer['model'], 136)
    inference = tensor_map(generator['inference'], 2)
    if set(inference) != {'level0_unigram', 'token_unigram'}:
        raise ValueError('unexpected inference tensors')
    for name, width in [('level0_unigram', 8192), ('token_unigram', 30528)]:
        value = inference[name]
        if (value.dtype != torch.float32 or tuple(value.shape) != (width,) or
                not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or
                not math.isclose(float(value.double().sum()), 1.0, rel_tol=0, abs_tol=1e-6)):
            raise ValueError(f'invalid inference distribution: {name}')
    return {'generator.safetensors': model, 'codec.safetensors': codec,
            'inference.safetensors': inference}


def export(source, tokenizer_asset, size, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if size not in MODEL_COUNTS:
        raise ValueError('unknown LM1B size')
    paths = {f'{size}/generator.pt': CHECKPOINTS[f'{size}/generator.pt'],
             'tokenizer.pt': CHECKPOINTS['tokenizer.pt'], f'{size}/config.yaml': CONFIGS[size]}
    for name, (length, sha) in paths.items():
        verify_input(source / name, length, sha)
    lock_path = ROOT / 'tools/reference/tokenizers-v1.lock.json'
    lock = json.loads(lock_path.read_bytes())['profiles']['bert']
    manifest, payload, license_file = (tokenizer_asset / n for n in ('manifest.json', 'tokenizer.json', 'LICENSE'))
    verify_input(manifest, manifest.stat().st_size, lock['manifest_sha256'])
    meta = json.loads(manifest.read_bytes())
    verify_input(payload, meta['payload_bytes'], lock['payload_sha256'])
    verify_input(license_file, license_file.stat().st_size, meta['license_sha256'])
    if (meta['profile'], meta['eos_token_id'], meta['pad_token_id'], meta['total_vocab_size']) != ('bert', 101, 0, 30522):
        raise ValueError('unexpected tokenizer profile')
    config = yaml.safe_load((source / size / 'config.yaml').read_bytes())
    if config['task'] != f'lm1b-{size}':
        raise ValueError('unexpected LM1B config')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_inventory.py').resolve(),
               Path(__file__).with_name('wmt14_export_preparation.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(), lock_path,
               manifest.resolve(), payload.resolve(), license_file.resolve()]
    before = {p.name: digest_file(p) for p in watched}
    if len(before) != len(watched):
        raise ValueError('duplicate watched names')
    generator = torch.load(source / size / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    tokenizer = torch.load(source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    parts = checkpoint_parts(generator, tokenizer, size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.ncm-lm1b-prep-', dir=output.parent))
    try:
        inventory = {name: save_checked(temporary / name, values) for name, values in parts.items()}
        json_file(temporary / 'config.json', config)
        for origin, name in ((payload, 'tokenizer.json'), (manifest, 'tokenizer-manifest.json'),
                             (license_file, 'tokenizer.LICENSE')):
            shutil.copyfile(origin, temporary / name)
        json_file(temporary / 'provenance.json', {
            'scope': 'experimental checkpoint inventory; no runtime or inference qualification',
            'python': platform.python_version(), 'torch': torch.__version__,
            'upstream_commit': UPSTREAM, 'checkpoint_revision': REVISION,
            'source_sha256': {name: sha for name, (_, sha) in paths.items()},
            'source_loading': 'torch.load(weights_only=True, mmap=True); no unrestricted retry',
            'watched_sha256': before, 'tensor_policy': 'all checkpoint tensors preserved; no inference classification',
        })
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in sorted(temporary.iterdir())}
        json_file(temporary / 'manifest.json', {'schema_version': 1, 'profile': f'lm1b-{size}-preparation-v1',
                                                  'files': files, 'tensors': inventory})
        for name, (length, sha) in paths.items():
            verify_input(source / name, length, sha)
        if before != {p.name: digest_file(p) for p in watched}:
            raise ValueError('export inputs changed')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'profile': f'lm1b-{size}-preparation-v1',
            'tensor_counts': {name: len(values) for name, values in inventory.items()},
            'manifest_sha256': digest_file(output / 'manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer-asset', type=Path, required=True)
    parser.add_argument('--size', choices=tuple(MODEL_COUNTS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.tokenizer_asset, args.size, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
