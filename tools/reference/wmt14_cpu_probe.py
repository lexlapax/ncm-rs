"""Capture one pinned WMT14 upstream forward pass on CPU, not a GPU quality gate."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from tinystories_artifact import ROOT, UPSTREAM, digest_file, json_file, verify_input
from wmt14_inventory import INPUTS

TOKENIZER_SHA = '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prefixes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (args.output.is_symlink() or output.exists() or
            not output.is_relative_to((ROOT / 'reports').resolve())):
        raise ValueError('output must be a new file under reports')
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
    if revision != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if args.prefixes.is_symlink() or digest_file(args.prefixes) != TOKENIZER_SHA:
        raise ValueError('tokenizer reference differs')
    source_files = [Path(__file__).resolve(), Path(__file__).with_name('wmt14_inventory.py').resolve(),
                    Path(__file__).with_name('tinystories_artifact.py').resolve(),
                    upstream / 'scripts/eval_shard.py'] + sorted((upstream / 'models').glob('*.py'))
    before = {str(path): digest_file(path) for path in source_files}
    data = json.loads(args.prefixes.read_bytes())
    prefix = data['prefixes'][0]
    if len(data['prefixes']) != 3003 or len(prefix) != 96 or any(type(x) is not int or x < 0 or x >= 151680 for x in prefix):
        raise ValueError('unexpected prefix shape')
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    generation = cfg.generation
    if (cfg.task != 'wmt14' or generation.temperature != 1.0 or generation.top_k != 1 or
            generation.top_p != 0.0 or generation.cfg_scale != 0.0 or
            generation.get('frequency_penalty', 0) or generation.get('adjacency_penalty')):
        raise ValueError('unsupported generation configuration')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    codec_state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    generator_state = torch.load(args.source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    torch.set_num_threads(1)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(codec_state['model'], strict=True)
    codec.eval()
    model = NextConceptTransformer(codec, cfg.model)
    model.load_state_dict(generator_state['model'], strict=True)
    model.eval()
    with torch.inference_mode():
        codes = model.generate(None, torch.tensor([prefix], dtype=torch.long), temperature=1.0,
                               top_k=1, top_p=0.0, use_kv_cache=False, cfg_scale=0.0)
        _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
        ids = codec.decoder(levels[-1]).argmax(-1)
    code_values, id_values = codes.tolist()[0], ids.tolist()[0]
    if (len(code_values) != 313 or len(id_values) != 64 or
            any(not 0 <= x < 4096 for x in code_values) or any(not 0 <= x < 151680 for x in id_values)):
        raise ValueError('unexpected output shape or range')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if digest_file(args.prefixes) != TOKENIZER_SHA or before != {str(p): digest_file(p) for p in source_files}:
        raise ValueError('probe inputs changed')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('upstream checkout changed')
    json_file(output, {'scope': 'one upstream CPU row, not a CUDA, Metal, or quality gate',
                       'upstream': revision, 'torch': torch.__version__, 'prefix_reference_sha256': TOKENIZER_SHA,
                       'source_sha256': before, 'checkpoint_sha256': {name: sha for name, (_, sha) in INPUTS.items()},
                       'row': 0, 'prefix': prefix, 'codes': code_values, 'raw_ids': id_values})
    print(json.dumps({'row': 0, 'codes': len(code_values), 'raw_ids': len(id_values),
                      'output_sha256': digest_file(output)}, sort_keys=True))


if __name__ == '__main__':
    main()
