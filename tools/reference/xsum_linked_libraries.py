"""Identify the native CUDA libraries linked by an XSum CUDA executable."""
import hashlib
import os
import subprocess
from pathlib import Path


REQUIRED = {'libcuda.so.1', 'libcurand.so.10', 'libcublas.so.13',
            'libcublasLt.so.13', 'libcudnn.so.9'}


def linked_libraries(binary, cudnn_dir):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(cudnn_dir)
    lines = subprocess.check_output(['ldd', str(binary)], env=env, text=True).splitlines()
    resolved = {}
    for line in lines:
        fields = line.strip().split()
        if len(fields) < 3 or fields[0] not in REQUIRED:
            continue
        if fields[1] != '=>' or fields[2] == 'not':
            raise ValueError(f'cannot resolve required native library {fields[0]}')
        path = Path(fields[2]).resolve(strict=True)
        if not path.is_file() or fields[0] in resolved:
            raise ValueError(f'invalid native library {fields[0]}')
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
                digest.update(block)
        resolved[fields[0]] = {'path': str(path), 'sha256': digest.hexdigest()}
    if set(resolved) != REQUIRED:
        raise ValueError('required linked CUDA libraries were not resolved')
    return resolved
