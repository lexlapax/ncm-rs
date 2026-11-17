"""Create an experimental, lossless WMT14 checkpoint inventory; not a runtime artifact."""
import argparse
import hashlib
import json
import platform
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from tinystories_artifact import ROOT, UPSTREAM, REVISION, digest_file, json_file, tensor_bytes, verify_input
from wmt14_artifact import checkpoint_parts
from wmt14_inventory import INPUTS


def save_checked(path, tensors):
    """Round-trip names, shapes, dtypes and bytes; return the complete inventory."""
    save_file(dict(tensors), path)
    restored = load_file(path)
    if set(restored) != set(tensors):
        raise ValueError('tensor names changed during serialization')
    inventory = []
    for name in sorted(tensors):
        original, actual = tensors[name], restored[name]
        raw = tensor_bytes(original)
        if original.dtype != actual.dtype or original.shape != actual.shape or raw != tensor_bytes(actual):
            raise ValueError(f'tensor changed during serialization: {name}')
        inventory.append({'name': name, 'shape': list(original.shape),
                          'dtype': 'F32' if original.dtype == torch.float32 else 'I64',
                          'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return inventory


def export(source, tokenizer_asset, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for name, (size, sha) in INPUTS.items():
        verify_input(source / name, size, sha)
    lock_path = ROOT / 'tools/reference/tokenizers-v1.lock.json'
    lock = json.loads(lock_path.read_bytes())['profiles']['qwen3']
    manifest, payload, license_file = (tokenizer_asset / name for name in ('manifest.json', 'tokenizer.json', 'LICENSE'))
    verify_input(manifest, manifest.stat().st_size, lock['manifest_sha256'])
    meta = json.loads(manifest.read_bytes())
    verify_input(payload, meta['payload_bytes'], lock['payload_sha256'])
    verify_input(license_file, license_file.stat().st_size, meta['license_sha256'])
    if (meta['profile'], meta['eos_token_id'], meta['pad_token_id'], meta['total_vocab_size']) != ('qwen3', 151645, 151643, 151669):
        raise ValueError('unexpected tokenizer profile')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('wmt14_artifact.py').resolve(),
               Path(__file__).with_name('wmt14_inventory.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(), lock_path,
               manifest.resolve(), payload.resolve(), license_file.resolve()]
    before = {str(path): digest_file(path) for path in watched}
    config = yaml.safe_load((source / 'config.yaml').read_bytes())
    if config['task'] != 'wmt14':
        raise ValueError('unexpected task')
    generator = torch.load(source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    tokenizer = torch.load(source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    parts = checkpoint_parts(generator, tokenizer)
    codec = {**parts['codec_other'], **parts['ema_and_hits']}
    if len(codec) != 96:
        raise ValueError('codec tensor preservation failed')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.ncm-wmt14-prep-', dir=output.parent))
    try:
        inventory = {'generator.safetensors': save_checked(temporary / 'generator.safetensors', parts['generator']),
                     'codec.safetensors': save_checked(temporary / 'codec.safetensors', codec)}
        json_file(temporary / 'config.json', config)
        for origin, name in ((payload, 'tokenizer.json'), (manifest, 'tokenizer-manifest.json'),
                             (license_file, 'tokenizer.LICENSE')):
            shutil.copyfile(origin, temporary / name)
        json_file(temporary / 'provenance.json', {
            'scope': 'experimental checkpoint inventory; no runtime or inference qualification',
            'python': platform.python_version(), 'torch': torch.__version__,
            'upstream_commit': UPSTREAM, 'checkpoint_revision': REVISION,
            'source_sha256': {name: sha for name, (_, sha) in INPUTS.items()},
            'config_identity': 'locally observed SHA-256, not independently verified as an upstream pin',
            'source_loading': 'torch.load(weights_only=True, mmap=True); no unrestricted retry',
            'watched_sha256': before, 'tensor_policy': 'all checkpoint tensors preserved; no inference classification',
        })
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in sorted(temporary.iterdir())}
        json_file(temporary / 'manifest.json', {'schema_version': 1, 'profile': 'wmt14-preparation-v1',
                                                  'files': files, 'tensors': inventory})
        for name, (size, sha) in INPUTS.items():
            verify_input(source / name, size, sha)
        if before != {str(path): digest_file(path) for path in watched}:
            raise ValueError('export inputs changed')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'profile': 'wmt14-preparation-v1', 'generator_tensors': len(inventory['generator.safetensors']),
            'codec_tensors': len(inventory['codec.safetensors']), 'manifest_sha256': digest_file(output / 'manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.tokenizer_asset, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
