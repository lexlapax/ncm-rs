"""Phase 2 codec-fp32-v1 oracle. No changes to pinned upstream model math."""
import argparse
import copy
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from sudoku_artifact import UPSTREAM, TRAINING, digest, tensor_bytes, json_file

ROOT = Path(__file__).resolve().parents[2]
POLICY = {
    "primitive": {"atol": 1e-5, "rtol": 1e-4},
    "component": {"atol": 1e-4, "rtol": 2e-4},
    "distance": {"atol": 1e-3, "rtol": 2e-4},
    "accumulated": {"atol": 5e-4, "rtol": 2e-4},
}


def reference_inputs(length):
    if length == 89:
        boards = []
        for shift in (0, 1):
            board = []
            for row in range(9):
                if row:
                    board.append(10)
                board.extend((row * 3 + row // 3 + col + shift) % 9 + 1 for col in range(9))
            boards.append(board)
        blank = [10 if (i+1) % 10 == 0 else 0 for i in range(89)]
        mixed = [v if v == 10 or i % 3 else 0 for i, v in enumerate(boards[0])]
        return {"solved_batch2": boards, "blank_batch1": [blank], "mixed_batch2": [mixed, boards[1]]}
    return {"pattern_batch2": [[(i*5+3) % 12 for i in range(length)],
                                [(i*7+1) % 12 for i in range(length)]],
            "constant_batch1": [[1]*length]}


def category(name):
    if name.endswith("distances"):
        return "distance"
    if name.endswith(("cumulative", "logits", "decoder.output", "decoder.input", "decoder.middle_block.output")):
        return "accumulated"
    return "component"


def margins(scores, kind, largest):
    values = scores.detach().cpu().double()
    if values.ndim == 0 or values.shape[-1] < 2 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("margins require finite rows with at least two scores")
    pair = values.topk(2, dim=-1, largest=largest).values
    gap = (pair[..., 0]-pair[..., 1]).abs()
    tolerance = POLICY[kind]
    bound = 2*(tolerance["atol"] + tolerance["rtol"]*values.abs().amax(-1))
    return {"shape": list(gap.shape), "minimum_gap": gap.min().item(),
            "gaps": gap.flatten().tolist(), "stable": (gap > bound).flatten().tolist()}


def capture(codec, ids):
    traces, decisions, handles = {}, {}, []
    phase = ["encode"]
    distance_level = [0]
    cumulative = {}

    def save(name, value):
        if name in traces:
            raise RuntimeError(f"duplicate trace: {name}")
        value = value.detach().cpu().contiguous().clone()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"nonfinite trace: {name}")
        traces[name] = value

    originals = {name: getattr(codec.vq, name) for name in
                 ("_distances", "_down_for_quant", "_apply_quant_resi")}

    def distances(x, weight):
        level = distance_level[0]
        output = originals["_distances"](x, weight)
        name = f"encode.level_{level:02d}"
        save(name+".quant_input", x)
        save(name+".distances", output)
        decisions[name] = margins(output, "distance", largest=False)
        if level == len(codec.vq.levels)-1:
            save(name+".residual_before", x.reshape(ids.shape[0], -1, codec.embed_dim))
        distance_level[0] += 1
        return output

    def down(x, level, length):
        save(f"encode.level_{level:02d}.residual_before", x)
        return originals["_down_for_quant"](x, level, length)

    def phi(x, level):
        output = originals["_apply_quant_resi"](x, level)
        name = f"{phase[0]}.level_{level:02d}"
        save(name+".upsampled", x)
        save(name+".contribution", output)
        if phase[0] != "encode":
            old = cumulative.get(phase[0], torch.zeros_like(output))
            cumulative[phase[0]] = old + output
            save(name+".cumulative", cumulative[phase[0]])
        return output

    codec.vq._distances, codec.vq._down_for_quant, codec.vq._apply_quant_resi = distances, down, phi
    for name, module in codec.named_modules():
        if (name == "encoder" or name == "decoder" or name.startswith("decoder.")) and not isinstance(module, (torch.nn.Dropout, torch.nn.ModuleList)):
            # Record semantic boundaries, not eval-mode identity dropout modules.
            def hook(_module, args, output, name=name):
                if isinstance(output, torch.Tensor):
                    save(f"{phase[0]}.{name}.input", args[0])
                    save(f"{phase[0]}.{name}.output", output)
            handles.append(module.register_forward_hook(hook))
    try:
        save("input.ids", ids)
        z = codec.encode_z(ids)
        save("encode.pre_quant", z)
        codes, snapshots = codec.vq.z_to_idx(z, return_zhats=True)
        save("encode.codes", codes)
        for level, snapshot in enumerate(snapshots):
            save(f"encode.level_{level:02d}.cumulative", snapshot)
        assert distance_level[0] == len(codec.vq.levels)
        supplied = torch.cat([
            (torch.arange(ids.shape[0]*length, device=ids.device).reshape(ids.shape[0], length)+3*level) % vocab
            for level, (length, vocab) in enumerate(zip(codec.vq.levels, codec.vq.vqvae_vocab_sizes))
        ], dim=1)
        save("supplied.codes", supplied)
        for label, selection in (("decode", codes), ("forced", supplied)):
            phase[0] = label
            logits = codec.decode(selection)
            save(label+".logits", logits)
            save(label+".ids", logits.argmax(-1))
            decisions[label] = margins(logits, "accumulated", largest=True)
            # Shadow sums capture independent snapshots while original idx_to_z_q runs.
            assert torch.equal(cumulative[label].cpu(), traces[label+".decoder.input"])
    finally:
        for name, method in originals.items():
            setattr(codec.vq, name, method)
        for handle in handles:
            handle.remove()
    assert torch.equal(codec.encode(ids), codes)
    for level in range(len(codec.vq.levels)):
        assert torch.equal(traces[f"encode.level_{level:02d}.cumulative"],
                           traces[f"decode.level_{level:02d}.cumulative"])
    snapshots = [v for k, v in traces.items() if k.endswith(".cumulative")]
    assert len({v.data_ptr() for v in snapshots}) == len(snapshots)
    return traces, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--profile", choices=["tiny", "sudoku"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if torch.__version__.split("+")[0] != "2.15.0":
        raise RuntimeError("Torch version differs from pinned oracle")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    if subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        raise RuntimeError("unpinned upstream")
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("dirty upstream")
    anchor = json.loads((ROOT / "docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json").read_bytes())
    for name in ("config.json", "codec.safetensors", "training.safetensors"):
        data = (args.bundle / name).read_bytes()
        if len(data) != anchor["files"][name]["bytes"] or digest(data) != anchor["files"][name]["sha256"]:
            raise RuntimeError(f"untrusted canonical source: {name}")
    config = json.loads((args.bundle / "config.json").read_bytes())["tokenizer"]["vqvae"]
    config = copy.deepcopy(config)
    if args.profile == "tiny":
        config.update(context_length=16, embed_dim=16, n_layers=4, vqvae_vocab_size=[4]*4, dropout=0.)
        config["vector_quantizer_config"]["levels"]["manual_levels"] = [1, 3, 7, 16]
        config["vector_quantizer_config"].update(learned_l1_cascade_norm=True, learned_l1_cascade_norm_final=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    codec = VQVAE(**OmegaConf.create(config)).eval()
    if args.profile == "sudoku":
        state = load_file(args.bundle / "codec.safetensors")
        training = load_file(args.bundle / "training.safetensors")
        assert not state.keys() & training.keys()
        state.update(training)
        codec.load_state_dict(state, strict=True)
    weights = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in codec.state_dict().items()
               if not TRAINING.fullmatch(name)}
    codec = codec.to(args.device)
    args.output.mkdir(parents=True)
    cases, all_traces = [], {}
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for name, inputs in reference_inputs(config["context_length"]).items():
            traces, decisions = capture(codec, torch.tensor(inputs, dtype=torch.int64, device=args.device))
            all_traces.update({name+"/"+key: value for key, value in traces.items()})
            cases.append({"name": name, "decisions": decisions,
                          "traces": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                            "policy": category(key), "sha256": digest(tensor_bytes(value))}
                                     for key, value in traces.items()}})
    save_file(all_traces, args.output / "traces.safetensors")
    if args.profile == "tiny":
        arrays = {}
        def intern(tensor):
            dtype = {torch.float32: "F32", torch.int64: "I64"}[tensor.dtype]
            raw = tensor_bytes(tensor)
            key = digest(json.dumps([dtype, list(tensor.shape)]).encode()+b"\0"+raw)
            arrays[key] = {"shape": list(tensor.shape), "dtype": dtype, "data_le_hex": raw.hex()}
            return key
        fixture = {"schema_version": 1, "profile": "synthetic-codec-v1",
            "upstream_commit": UPSTREAM, "config": config, "policy": POLICY,
            "weights": {name: intern(tensor) for name, tensor in weights.items()},
            "cases": [{"name": case["name"], "decisions": case["decisions"],
                       "traces": {name.split("/", 1)[1]: intern(tensor)
                                  for name, tensor in all_traces.items() if name.startswith(case["name"]+"/")}}
                      for case in cases]}
        fixture["arrays"] = arrays
        json_file(args.output / "fixture.json", fixture)
    json_file(args.output / "config.json", config)
    json_file(args.output / "report.json", {"schema_version": 1, "profile": args.profile,
        "upstream_commit": UPSTREAM, "torch": torch.__version__, "python": platform.python_version(),
        "device": args.device, "seed": 42, "tf32": False, "sdpa": "math", "policy": POLICY,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "helper_sha256": digest(Path(__file__).with_name("sudoku_artifact.py").read_bytes()),
        "trace_sha256": digest((args.output / "traces.safetensors").read_bytes()),
        "config_sha256": digest((args.output / "config.json").read_bytes()),
        "weights": {name: {"shape": list(tensor.shape), "sha256": digest(tensor_bytes(tensor))} for name, tensor in weights.items()},
        "reference_patches": [], "instrumentation": "read-only hooks; cloned observations and checked decode shadow sums",
        "rust_parity": None, "performance": None, "cases": cases})
    print(f"{args.profile}/{args.device}: {len(all_traces)} traces, {len(cases)} cases; Rust parity pending")


if __name__ == "__main__":
    main()
