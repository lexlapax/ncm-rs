"""Probe LM1B codec training-state dependence on one fixed-code CPU sample."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from lm1b_export_preparation import CONFIGS
from lm1b_inventory import CHECKPOINTS
from tinystories_artifact import ROOT, TRAINING, UPSTREAM, digest_file, json_file, tensor_bytes, verify_input

SAMPLE_SHA = '0b1299d3f9b874a25ceb769f4d94733ccce0a62d8e2bc46b33a72eb28b6f850a'


def training_names(names):
    expected = {f'vq.{kind}_{index}' for kind in ('ema_cluster_size', 'ema_w', 'codebook_hit') for index in range(12)}
    found = {name for name in names if TRAINING.fullmatch(name)}
    if found != expected:
        raise ValueError('unexpected LM1B codec training-state inventory')
    return sorted(found)


def summarize(tensors):
    return [{'shape': list(t.shape), 'dtype': str(t.dtype),
             'sha256': hashlib.sha256(tensor_bytes(t)).hexdigest()} for t in tensors]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    upstream = args.upstream.resolve()
    def upstream_state():
        rev = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
        if rev != UPSTREAM or status:
            raise ValueError('upstream must be clean and pinned')
    upstream_state()
    inputs = {'tokenizer.pt': CHECKPOINTS['tokenizer.pt'], 'small/config.yaml': CONFIGS['small']}
    for name, (size, sha) in inputs.items():
        verify_input(args.source / name, size, sha)
    verify_input(args.sample, args.sample.stat().st_size, SAMPLE_SHA)
    sample = json.loads(args.sample.read_bytes())
    codes = sample['codes']
    if sample['size'] != 'small' or len(codes) != 641 or any(type(x) is not int or not 0 <= x < 8192 for x in codes):
        raise ValueError('unexpected fixed-code input')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_export_preparation.py').resolve(),
               Path(__file__).with_name('lm1b_inventory.py').resolve(), Path(__file__).with_name('tinystories_artifact.py').resolve(),
               *sorted((upstream / 'models').glob('*.py'))]
    before = {str(p): digest_file(p) for p in watched}
    config = OmegaConf.create(yaml.safe_load((args.source / 'small/config.yaml').read_bytes()))
    state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    names = training_names(state['model'])
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from models.vqvae import VQVAE
    torch.set_num_threads(1)
    codec = VQVAE(**config.tokenizer.vqvae)
    codec.load_state_dict(state['model'], strict=True)
    codec.eval()
    indices = torch.tensor([codes], dtype=torch.long)
    def capture():
        with torch.inference_mode():
            ncm_input = codec.vq.idx_to_ncm_input(indices)
            _, levels = codec.vq.idx_to_zhats(indices, return_zlevels=True)
            logits = codec.decoder(levels[-1])
        return [ncm_input, *levels, logits]
    original = capture()
    model_state = codec.state_dict()
    with torch.no_grad():
        for name in names:
            model_state[name].fill_(7)
    changed = capture()
    if any(not torch.equal(a, b) for a, b in zip(original, changed, strict=True)):
        raise ValueError('fixed-code inference changed after training-state mutation')
    for name, (size, sha) in inputs.items():
        verify_input(args.source / name, size, sha)
    verify_input(args.sample, args.sample.stat().st_size, SAMPLE_SHA)
    if before != {str(p): digest_file(p) for p in watched}:
        raise ValueError('source changed during probe')
    upstream_state()
    json_file(args.output, {'scope': 'one fixed-code CPU codec probe; not exhaustive or GPU qualification',
                             'sample_sha256': SAMPLE_SHA, 'training_names': names,
                             'outputs': summarize(original), 'exact_after_mutation': True,
                             'upstream_commit': UPSTREAM})
    print(json.dumps({'candidate_tensors': len(names), 'outputs': len(original),
                      'report_sha256': digest_file(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
