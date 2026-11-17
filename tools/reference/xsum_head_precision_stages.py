"""Compare arithmetic stages on captured XSum row; diagnostic only."""
import hashlib
import torch
from safetensors.torch import load_file

paths = {
    'cuda': ('reports/phase4-xsum-mlp-cuda-v1/row-4937.safetensors', '3723d9fda929e4d503614961f2f2126a18b6cefa563accb51c9cb3b8eb54ce0f'),
    'metal': ('reports/phase4-xsum-refined-norm-row-captured-v1/row-4937.safetensors', '9147eaf4bf36060151ae91e53cfe17fdafbccb82c41578fd69916d76a37b863c'),
    'head': ('reports/phase4-xsum-head-chain-input-v1.safetensors', 'eb7ec6d50460e4521ce06e5f5e827ba9fc35e8001a72e8098bd68499a53dc184'),
}
def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
assert all(digest(p) == h for p, h in paths.values())
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
d = load_file(paths['head'][0], device='cuda')
with torch.inference_mode():
    for name in ('cuda', 'metal'):
        trace = load_file(paths[name][0])
        hidden = trace['diagnostic.level_01.hidden'].cuda()
        full = torch.zeros((32, 6, 768), device='cuda'); full[9] = hidden[0]
        flat = full.view(192, 768)
        captured = trace['generator.level_01.logits'][0, 4]
        print(name, 'captured', float(captured[3182] - captured[3690]), int(captured.argmax()))
        for projection in ('f32', 'f64'):
            dtype = torch.float32 if projection == 'f32' else torch.float64
            projected = flat.to(dtype) @ d['weight'].to(dtype).T
            for dot in ('f32', 'f64'):
                dtype2 = torch.float32 if dot == 'f32' else torch.float64
                raw = projected.to(dtype2) @ d['book'].to(dtype2).T
                score = (((raw * 2 - d['norm'].to(dtype2)) * d['scales'][0].to(dtype2)) * d['scales'][1].to(dtype2)).view(32, 6, 4096)[9, 4]
                print(name, projection, dot, float(score[3182] - score[3690]), int(score.argmax()), flush=True)
assert all(digest(p) == h for p, h in paths.values())
