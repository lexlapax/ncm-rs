"""Export the pinned Sudoku-easy pair without unrestricted pickle or weight changes.

This is a deliberately scoped Phase 1 inspection bundle, not a general model importer.
"""

import argparse
import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

UPSTREAM = "c70274273f489abc73832b2b98584a8483b6b203"
REVISION = "d342ce8aaea58830eef3b8ad473664673e41f692"
INPUT_SIZES = {"config.yaml": 1980, "generator.pt": 102560203, "tokenizer.pt": 59816209}
INPUTS = {
    "config.yaml": "68d74fe310e909ca1892338258a2d7143293e685ec1c4747265d9134013d1605",
    "generator.pt": "1cefebcd1eae0e210b7e991fdce0e86964d0e897e94a3b432d1008ab92d9926e",
    "tokenizer.pt": "3f9c87e99fd49900929cd650181f55bf00530280416992b11ddc4ea869b39d68",
}
TRAINING = re.compile(r"vq\.(ema_cluster_size|ema_w|codebook_hit)_\d+$")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(tensor):
    # Raw little-endian contiguous bytes, including signed zeros and mask infinities.
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def json_file(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def checkpoint(data):
    obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or set(obj) != {"model"} or not isinstance(obj["model"], dict):
        raise ValueError("unexpected checkpoint metadata; no silent discard")
    state = obj["model"]
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state dict must contain named tensors only")
        if tensor.dtype not in (torch.float32, torch.int64) or tensor.layout != torch.strided:
            raise ValueError(f"unsupported tensor {name}: {tensor.dtype}/{tensor.layout}")
    # This exact checkpoint has no aliases. Refuse unexpected sharing rather than flatten it.
    addresses = [t.untyped_storage().data_ptr() for t in state.values()]
    if len(set(addresses)) != len(addresses):
        raise ValueError("unexpected shared storage; add an explicit alias mapping before export")
    return state


def export(source, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = {}
    for name, expected in INPUTS.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != INPUT_SIZES[name]:
            raise ValueError(f"input type/size mismatch: {name}")
        with path.open("rb") as file:
            data = file.read(INPUT_SIZES[name] + 1)
        if digest(data) != expected:
            raise ValueError(f"untrusted or changed input: {name}")
        inputs[name] = data
    config = yaml.safe_load(inputs["config.yaml"])
    if config["task"] != "sudoku-easy":
        raise ValueError("only the pinned Sudoku-easy profile is supported")
    codec = checkpoint(inputs["tokenizer.pt"])
    generator = checkpoint(inputs["generator.pt"])
    bundles = {
        "codec.safetensors": {k: v for k, v in codec.items() if not TRAINING.fullmatch(k)},
        "training.safetensors": {k: v for k, v in codec.items() if TRAINING.fullmatch(k)},
        "generator.safetensors": generator,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ncm-export-", dir=output.parent))
    try:
        inventory = {}
        for filename, state in bundles.items():
            tensors = {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}
            save_file(tensors, temporary / filename)
            restored = load_file(temporary / filename)
            if set(restored) != set(state):
                raise ValueError("roundtrip tensor names changed")
            entries = []
            for name in sorted(state):
                original, loaded = state[name], restored[name]
                raw = tensor_bytes(original)
                if (original.dtype != loaded.dtype or original.shape != loaded.shape
                        or raw != tensor_bytes(loaded)):
                    raise ValueError(f"non-bit-exact roundtrip: {filename}/{name}")
                entries.append({"name": name, "shape": list(original.shape),
                                "dtype": "F32" if original.dtype == torch.float32 else "I64",
                                "bytes": len(raw), "sha256": digest(raw)})
            inventory[filename] = entries
        json_file(temporary / "config.json", config)
        json_file(temporary / "tokenizer.json", {
            "kind": "sudoku-v1", "vocabulary": ["_"] + list("123456789") + ["|", "<BOS>"],
            "blank_id": 0, "separator_id": 10, "bos_id": 11, "pad_id": None, "eos_id": None,
            "sequence_length": 89, "prefix_length": 91,
        })
        json_file(temporary / "provenance.json", {
            "converter_sha256": digest(Path(__file__).read_bytes()),
            "converter_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(), "torch": torch.__version__,
            "source_sha256": INPUTS, "code_license": "MIT", "weights_license": "MIT",
            "serialization": "safetensors; unchanged upstream layouts/dtypes",
            "source_loading": "torch.load(weights_only=True); no unrestricted retry",
            "aliases": [], "inference_metadata": "absent in both pinned checkpoints",
            "config_policy": "upstream config preserved; missing defaults interpreted only at pinned source commit",
            "omitted_state": [], "training_state": "EMA/count/hit buffers separated, not dropped",
            "derived_buffers": "nonpersistent buffers reconstructed by pinned Python source during oracle tests",
        })
        files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
                 for p in sorted(temporary.iterdir())}
        json_file(temporary / "manifest.json", {
            "schema_version": 1, "profile": "sudoku-easy-inspection-v1", "upstream_commit": UPSTREAM,
            "checkpoint_repository": "hazyresearch/ncm-checkpoints", "checkpoint_revision": REVISION,
            "files": files, "tensors": inventory,
        })
        # The target must not exist. The completed bundle is published only after verification.
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"profile": "sudoku-easy-inspection-v1", "source_tensors": len(codec) + len(generator),
            "bundle_tensor_counts": {k: len(v) for k, v in bundles.items()},
            "bit_exact_roundtrip": True, "manifest_sha256": digest((output / "manifest.json").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists")
    report = export(args.source, args.output)
    json_file(args.report, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
