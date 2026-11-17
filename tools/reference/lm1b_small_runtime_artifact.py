"""Export the pinned LM1B-small inference artifact; no model support implied."""
import argparse
import json
import platform
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from lm1b_export_preparation import CONFIGS, checkpoint_parts
from lm1b_inventory import CHECKPOINTS
from tinystories_artifact import ROOT, REVISION, TRAINING, UPSTREAM, digest_file, json_file, verify_input
from wmt14_export_preparation import save_checked


def bundle_states(generator, tokenizer):
    parts = checkpoint_parts(generator, tokenizer, 'small')
    codec = parts['codec.safetensors']
    expected = {f'vq.{kind}_{i}' for kind in ('ema_cluster_size', 'ema_w', 'codebook_hit') for i in range(12)}
    training = {name: value for name, value in codec.items() if TRAINING.fullmatch(name)}
    if set(training) != expected:
        raise ValueError('unexpected LM1B training-state inventory')
    other = {name: value for name, value in codec.items() if name not in training}
    if len(other) != 100:
        raise ValueError('unexpected LM1B codec inventory')
    return {'generator.safetensors': parts['generator.safetensors'], 'codec.safetensors': other,
            'training.safetensors': training, 'inference.safetensors': parts['inference.safetensors']}


def export(source, tokenizer_asset, output):
    out = output.resolve()
    if output.exists() or output.is_symlink() or not out.is_relative_to((ROOT / 'artifacts/canonical').resolve()):
        raise ValueError('new output under artifacts/canonical required')
    paths = {'small/generator.pt': CHECKPOINTS['small/generator.pt'],
             'tokenizer.pt': CHECKPOINTS['tokenizer.pt'], 'small/config.yaml': CONFIGS['small']}
    for name, (size, sha) in paths.items():
        verify_input(source / name, size, sha)
    lock_path = ROOT / 'tools/reference/tokenizers-v1.lock.json'
    lock = json.loads(lock_path.read_bytes())['profiles']['bert']
    manifest, payload, license_file = (tokenizer_asset / name for name in ('manifest.json', 'tokenizer.json', 'LICENSE'))
    verify_input(manifest, manifest.stat().st_size, lock['manifest_sha256'])
    meta = json.loads(manifest.read_bytes())
    verify_input(payload, meta['payload_bytes'], lock['payload_sha256'])
    verify_input(license_file, license_file.stat().st_size, meta['license_sha256'])
    if (meta['profile'], meta['eos_token_id'], meta['pad_token_id'], meta['total_vocab_size']) != ('bert', 101, 0, 30522):
        raise ValueError('unexpected tokenizer profile')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_export_preparation.py').resolve(),
               Path(__file__).with_name('lm1b_inventory.py').resolve(),
               Path(__file__).with_name('wmt14_export_preparation.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(), lock_path,
               manifest.resolve(), payload.resolve(), license_file.resolve()]
    before = {p.name: digest_file(p) for p in watched}
    if len(before) != len(watched):
        raise ValueError('duplicate watched names')
    config = yaml.safe_load((source / 'small/config.yaml').read_bytes())
    if config['task'] != 'lm1b-small':
        raise ValueError('unexpected task')
    generator = torch.load(source / 'small/generator.pt', map_location='cpu', weights_only=True, mmap=True)
    tokenizer = torch.load(source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    bundles = bundle_states(generator, tokenizer)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.ncm-lm1b-small-inference-prep-', dir=out.parent))
    try:
        inventory = {name: save_checked(temporary / name, state) for name, state in bundles.items()}
        json_file(temporary / 'config.json', config)
        for origin, name in ((payload, 'tokenizer.json'), (manifest, 'tokenizer-manifest.json'),
                             (license_file, 'tokenizer.LICENSE')):
            shutil.copyfile(origin, temporary / name)
        json_file(temporary / 'provenance.json', {
            'scope': 'bounded inference artifact export; no model or GPU qualification',
            'python': platform.python_version(), 'torch': torch.__version__,
            'source_sha256': {name: sha for name, (_, sha) in paths.items()}, 'watched_sha256': before,
            'source_loading': 'torch.load(weights_only=True, mmap=True); no unrestricted retry',
            'training_state': '36 EMA/hit buffers retained separately; classification limited to generation preparation',
            'inference_metadata': ['level0_unigram', 'token_unigram'],
            'serialization': 'safetensors; unchanged tensor layouts and dtypes', 'aliases': [],
        })
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in sorted(temporary.iterdir())}
        json_file(temporary / 'manifest.json', {'schema_version': 1, 'profile': 'lm1b-small-inference-v1',
                  'upstream_commit': UPSTREAM, 'checkpoint_repository': 'hazyresearch/ncm-checkpoints',
                  'checkpoint_revision': REVISION, 'files': files, 'tensors': inventory})
        for name, (size, sha) in paths.items():
            verify_input(source / name, size, sha)
        if before != {p.name: digest_file(p) for p in watched}:
            raise ValueError('inputs changed during export')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'profile': 'lm1b-small-inference-v1',
            'tensor_counts': {name: len(state) for name, state in bundles.items()},
            'manifest_sha256': digest_file(out / 'manifest.json'),
            'scope': 'artifact export only; model and GPU qualification pending'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.tokenizer_asset, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
