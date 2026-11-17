"""Export the explicitly pinned WMT14 inference artifact without changing tensor bytes."""
import argparse
import json
import platform
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from tinystories_artifact import REVISION, ROOT, UPSTREAM, digest_file, json_file, verify_input
from wmt14_artifact import checkpoint_parts
from wmt14_export_preparation import save_checked
from wmt14_inventory import INPUTS


def bundle_states(generator, tokenizer):
    parts = checkpoint_parts(generator, tokenizer)
    return {'generator.safetensors': parts['generator'], 'codec.safetensors': parts['codec_other'],
            'training.safetensors': parts['ema_and_hits']}


def export(source, tokenizer_asset, output):
    out = output.resolve()
    if output.exists() or output.is_symlink() or not out.is_relative_to((ROOT / 'artifacts/canonical').resolve()):
        raise ValueError('new output under artifacts/canonical required')
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
               Path(__file__).with_name('wmt14_export_preparation.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(), lock_path,
               manifest.resolve(), payload.resolve(), license_file.resolve()]
    before = {str(path): digest_file(path) for path in watched}
    config = yaml.safe_load((source / 'config.yaml').read_bytes())
    if config['task'] != 'wmt14':
        raise ValueError('unexpected checkpoint task')
    generator = torch.load(source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    tokenizer = torch.load(source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    bundles = bundle_states(generator, tokenizer)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.ncm-wmt14-', dir=out.parent))
    try:
        inventory = {name: save_checked(temporary / name, state) for name, state in bundles.items()}
        json_file(temporary / 'config.json', config)
        for origin, name in ((payload, 'tokenizer.json'), (manifest, 'tokenizer-manifest.json'),
                             (license_file, 'tokenizer.LICENSE')):
            shutil.copyfile(origin, temporary / name)
        json_file(temporary / 'provenance.json', {
            'converter_sha256': before[str(Path(__file__).resolve())],
            'helper_sha256': {name: before[str(Path(__file__).with_name(name).resolve())]
                              for name in ('wmt14_artifact.py', 'wmt14_inventory.py',
                                           'wmt14_export_preparation.py', 'tinystories_artifact.py')},
            'tokenizer_lock_sha256': before[str(lock_path)],
            'tokenizer_asset_sha256': {'manifest.json': before[str(manifest.resolve())],
                                       'tokenizer.json': before[str(payload.resolve())],
                                       'LICENSE': before[str(license_file.resolve())]},
            'python': platform.python_version(), 'torch': torch.__version__,
            'source_sha256': {name: sha for name, (_, sha) in INPUTS.items()},
            'source_loading': 'torch.load(weights_only=True, mmap=True); no unrestricted retry',
            'training_state': 'EMA/count/hit buffers separated and retained for generation-only artifact',
            'inference_metadata': [], 'serialization': 'safetensors; unchanged checkpoint layouts and dtypes',
            'code_license': 'MIT', 'weights_license': 'MIT', 'aliases': [],
        })
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in sorted(temporary.iterdir())}
        json_file(temporary / 'manifest.json', {
            'schema_version': 1, 'profile': 'wmt14-inference-v1', 'upstream_commit': UPSTREAM,
            'checkpoint_repository': 'hazyresearch/ncm-checkpoints', 'checkpoint_revision': REVISION,
            'files': files, 'tensors': inventory,
        })
        for name, (size, sha) in INPUTS.items():
            verify_input(source / name, size, sha)
        if before != {str(path): digest_file(path) for path in watched}:
            raise ValueError('export inputs changed')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'profile': 'wmt14-inference-v1', 'tensor_counts': {name: len(state) for name, state in bundles.items()},
            'manifest_sha256': digest_file(out / 'manifest.json'), 'bit_exact_roundtrip': True,
            'scope': 'artifact export only; runtime and GPU qualification pending'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.tokenizer_asset, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
