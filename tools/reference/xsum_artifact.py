"""Export only the pinned XSum checkpoint pair as checked inference data."""
import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from tinystories_artifact import (REVISION, ROOT, TRAINING, UPSTREAM, digest_file,
                                  json_file, tensor_bytes, tensor_map, verify_input)
from xsum_rope_reference import EXPECTED as ROPE_HASHES

INPUTS = {
    'generator.pt': (354355211, 'f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4'),
    'tokenizer.pt': (299626079, '1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7'),
    'config.yaml': (2345, '5dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493'),
}


def checkpoint_parts(generator, tokenizer):
    if not isinstance(generator, dict) or set(generator) != {'model', 'inference'}:
        raise ValueError('unexpected generator checkpoint metadata')
    if not isinstance(tokenizer, dict) or set(tokenizer) != {'model'}:
        raise ValueError('unexpected tokenizer checkpoint metadata')
    model = tensor_map(generator['model'], 194)
    codec = tensor_map(tokenizer['model'], 80)
    inference = tensor_map(generator['inference'], 1)
    if set(inference) != {'token_unigram'}:
        raise ValueError('unexpected inference metadata')
    unigram = inference['token_unigram']
    if (unigram.dtype != torch.float32 or tuple(unigram.shape) != (50304,)
            or not bool(torch.isfinite(unigram).all()) or bool((unigram < 0).any())
            or not math.isclose(float(unigram.double().sum()), 1.0, rel_tol=0, abs_tol=1e-6)):
        raise ValueError('invalid token allowance distribution')
    runtime = {name: tensor for name, tensor in codec.items() if not TRAINING.fullmatch(name)}
    training = {name: tensor for name, tensor in codec.items() if TRAINING.fullmatch(name)}
    if len(runtime) != 53 or len(training) != 27:
        raise ValueError('unexpected codec training/runtime classification')
    return {'generator.safetensors': model, 'codec.safetensors': runtime,
            'training.safetensors': training, 'inference.safetensors': inference}


def reference_rope(directory):
    """Read only the explicitly pinned rotary tensors, with no inferred fallback."""
    payload = directory / 'rope.safetensors'
    verify_input(payload, 131216, '5d205c1c42b467c00e10d03739e18b6433185bcc4c4f9b066d7c6be69ba8346a')
    tensors = load_file(payload)
    if set(tensors) != set(ROPE_HASHES):
        raise ValueError('invalid rotary reference inventory')
    for name, expected in ROPE_HASHES.items():
        value = tensors[name]
        if (value.dtype != torch.float32 or tuple(value.shape) != (512, 32)
                or hashlib.sha256(tensor_bytes(value)).hexdigest() != expected):
            raise ValueError(f'invalid pinned rotary tensor: {name}')
    return {f'rope_{name}': value for name, value in tensors.items()}


def export(source, tokenizer_asset, output, rope_reference=None):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for name, (size, expected) in INPUTS.items():
        verify_input(source / name, size, expected)
    lock_path = ROOT / 'tools/reference/tokenizers-v1.lock.json'
    lock = json.loads(lock_path.read_bytes())['profiles']['gpt2']
    manifest = tokenizer_asset / 'manifest.json'
    payload = tokenizer_asset / 'tokenizer.json'
    license_file = tokenizer_asset / 'LICENSE'
    verify_input(manifest, manifest.stat().st_size, lock['manifest_sha256'])
    metadata = json.loads(manifest.read_bytes())
    verify_input(payload, metadata['payload_bytes'], lock['payload_sha256'])
    verify_input(license_file, license_file.stat().st_size, metadata['license_sha256'])
    if (metadata['profile'] != 'gpt2' or metadata['kind'] != 'huggingface'
            or metadata['eos_token_id'] != 50256 or metadata['pad_token_id'] != 50257
            or metadata['total_vocab_size'] != 50258):
        raise ValueError('wrong canonical GPT-2 tokenizer profile')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('tinystories_artifact.py').resolve(),
               lock_path, manifest.resolve(), payload.resolve(), license_file.resolve()]
    if rope_reference is not None:
        watched.extend([(rope_reference / 'rope.safetensors').resolve(),
                        Path(__file__).with_name('xsum_rope_reference.py').resolve()])
    before = {str(path): digest_file(path) for path in watched}
    config = yaml.safe_load((source / 'config.yaml').read_bytes())
    if config['task'] != 'xsum':
        raise ValueError('wrong checkpoint task')
    generator = torch.load(source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    tokenizer = torch.load(source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    bundles = checkpoint_parts(generator, tokenizer)
    profile = 'xsum-inference-v1'
    if rope_reference is not None:
        bundles['inference.safetensors'].update(reference_rope(rope_reference))
        profile = 'xsum-inference-v2'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.ncm-xsum-', dir=output.parent))
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
        shutil.copyfile(manifest, temporary / 'tokenizer-manifest.json')
        shutil.copyfile(license_file, temporary / 'tokenizer.LICENSE')
        json_file(temporary / 'provenance.json', {
            'converter_sha256': before[str(Path(__file__).resolve())],
            'helper_sha256': before[str(Path(__file__).with_name('tinystories_artifact.py').resolve())],
            'tokenizer_lock_sha256': before[str(lock_path)],
            'converter_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'source_sha256': {name: sha for name, (_, sha) in INPUTS.items()},
            'source_loading': 'torch.load(weights_only=True, mmap=True); no unrestricted retry',
            'code_license': 'MIT', 'weights_license': 'MIT', 'aliases': [],
            'inference_metadata': (['token_unigram'] if rope_reference is None
                                   else ['token_unigram', 'rope_cos', 'rope_sin']),
            **({} if rope_reference is None else {
                'rope_reference_sha256': before[str((rope_reference / 'rope.safetensors').resolve())],
                'rope_reference_generator_sha256': before[str(Path(__file__).with_name('xsum_rope_reference.py').resolve())],
            }),
            'training_state': 'EMA/count/hit buffers separated, not dropped',
            'serialization': ('safetensors; unchanged upstream layouts/dtypes' if rope_reference is None
                              else 'safetensors; unchanged checkpoint tensors plus pinned rotary reference'),
        })
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in sorted(temporary.iterdir())}
        json_file(temporary / 'manifest.json', {
            'schema_version': 1, 'profile': profile, 'upstream_commit': UPSTREAM,
            'checkpoint_repository': 'hazyresearch/ncm-checkpoints', 'checkpoint_revision': REVISION,
            'files': files, 'tensors': inventory,
        })
        for name, (size, expected) in INPUTS.items():
            verify_input(source / name, size, expected)
        if before != {str(path): digest_file(path) for path in watched}:
            raise ValueError('converter or tokenizer inputs changed during export')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'profile': profile, 'bundle_tensor_counts': {k: len(v) for k, v in bundles.items()},
            'manifest_sha256': digest_file(output / 'manifest.json'), 'bit_exact_roundtrip': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--rope-reference', type=Path)
    args = parser.parse_args()
    if args.report.exists() or args.report.is_symlink():
        parser.error('report already exists')
    report = export(args.source, args.tokenizer_asset, args.output, args.rope_reference)
    json_file(args.report, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
