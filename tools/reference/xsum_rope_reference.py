"""Generate a separately versioned, pinned XSum rotary reference table.

This is an offline reference-preparation tool. Production inference does not
invoke Python, and this output is not yet an accepted model artifact.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = 'c70274273f489abc73832b2b98584a8483b6b203'
COMMON_SHA256 = '41b7c2c653d7ba312d583caca878ea3708d83da1a770b0b39651dbfd19754b73'
TORCH_VERSION = '2.15.0+cu132'
EXPECTED = {
    'cos': 'f1376ecd82c6f574ade39abbdc95db403338b466306c02062a260079116c7ed6',
    'sin': '81716ee36cb0d78e78c093171a1751124c57467459608f5d810b3a163866e523',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def generate(upstream, output):
    upstream = upstream.resolve(strict=True)
    source = upstream / 'models/common.py'
    if digest(source.read_bytes()) != COMMON_SHA256:
        raise ValueError('upstream rotary implementation differs')
    if (upstream / 'models/__init__.py').read_bytes() != b'':
        raise ValueError('upstream models package initializer differs')
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM or torch.__version__ != TORCH_VERSION:
        raise ValueError('upstream commit or reference torch version differs')
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    sys.path.insert(0, str(upstream))
    from models.common import precompute_rope_cos_sin

    with torch.inference_mode():
        cosine, sine = precompute_rope_cos_sin(torch.arange(512, dtype=torch.long), 64, 100000)
    tables = {}
    for name, full in [('cos', cosine), ('sin', sine)]:
        if full.dtype != torch.float32 or tuple(full.shape) != (512, 64):
            raise ValueError('unexpected upstream rotary table')
        if not torch.equal(full[:, :32], full[:, 32:]):
            raise ValueError('upstream rotary halves differ')
        table = full[:, :32].contiguous()
        raw = table.numpy().tobytes()
        if digest(raw) != EXPECTED[name]:
            raise ValueError(f'{name} table differs from pinned reference')
        tables[name] = table

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.xsum-rope-', dir=output.parent))
    try:
        payload = temporary / 'rope.safetensors'
        save_file(tables, payload)
        loaded = load_file(payload)
        if set(loaded) != set(tables) or any(loaded[k].shape != tables[k].shape
                                                or loaded[k].dtype != torch.float32
                                                or digest(loaded[k].numpy().tobytes()) != EXPECTED[k]
                                                for k in tables):
            raise ValueError('rotary serialization roundtrip differs')
        manifest = {
            'schema_version': 1,
            'profile': 'xsum-rope-reference-v1',
            'upstream_commit': UPSTREAM,
            'upstream_common_sha256': COMMON_SHA256,
            'torch_version': TORCH_VERSION,
            'generator_sha256': digest(Path(__file__).read_bytes()),
            'shape': [512, 32],
            'dtype': 'F32',
            'tensor_sha256': EXPECTED,
            'payload_sha256': digest(payload.read_bytes()),
        }
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    resolved = output.resolve()
    if not any(resolved.is_relative_to(ROOT / base) for base in ('reports', 'artifacts')):
        raise ValueError('output must be under reports or artifacts')
    print(json.dumps(generate(args.upstream, output), sort_keys=True))


if __name__ == '__main__':
    main()
