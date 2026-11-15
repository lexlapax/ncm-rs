"""Trace pinned TinyStories first-level learned upsampling on CUDA."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file
from tinystories_artifact import INPUTS, UPSTREAM, verify_input


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--sample', type=Path, help='JSON sample file; use its first-level codes')
    args = p.parse_args()
    if args.output.exists() or not torch.cuda.is_available():
        raise ValueError('output must be new and CUDA must be available')
    if subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'], text=True).strip() != UPSTREAM or subprocess.check_output(['git','-C',str(args.upstream),'status','--porcelain'],text=True).strip():
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items(): verify_input(args.source / name, size, sha)
    cfg = OmegaConf.create(yaml.safe_load((args.source/'config.yaml').read_bytes()))
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state=torch.load(args.source/'tokenizer.pt',map_location='cpu',weights_only=True,mmap=True)
    codec=VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(state['model'],strict=True)
    codec=codec.cuda().eval()
    tensors={}
    def save(name,x): tensors[name]=x.detach().cpu().contiguous().clone()
    with torch.inference_mode():
        initial = [0, 37] if args.sample is None else [row[0] for row in json.loads(args.sample.read_bytes())['codes'][:128]]
        if not initial or any(type(i) is not int or not 0 <= i < 4096 for i in initial):
            raise ValueError('invalid first-level codes')
        ids=torch.tensor([[i] for i in initial],device='cuda')
        x=codec.vq.embeddings[0](ids)
        save('embedding',x)
        x=x.transpose(-1,-2)
        for i,layer in enumerate(codec.vq.learned_up_l1):
            x=layer(x)
            save(f'cascade.{i}',x.transpose(-1,-2))
        x=x.transpose(-1,-2).contiguous()
        layer=codec.vq.quant_resi[0.0]
        conv=torch.nn.functional.conv1d(x.transpose(1,2).contiguous(), layer.weight, layer.bias, padding=1).transpose(1,2).contiguous()
        save('residual.conv',conv)
        save('residual.input_half',x * 0.5)
        save('residual.conv_half',conv * 0.5)
        save('refined',codec.vq._apply_quant_resi(x,0))
        torch.cuda.synchronize()
    args.output.mkdir(parents=True,exist_ok=False)
    save_file(tensors,str(args.output/'trace.safetensors'))
    (args.output/'report.json').write_text(json.dumps({'upstream':UPSTREAM,'tf32':False,'cuda':torch.cuda.get_device_name(0),'trace_sha256':hashlib.sha256((args.output/'trace.safetensors').read_bytes()).hexdigest(),'tensors':sorted(tensors),'sample_sha256':hashlib.sha256(args.sample.read_bytes()).hexdigest() if args.sample else None},indent=2)+'\n')


if __name__ == '__main__': main()
