#!/usr/bin/env python3
"""One-command quickstart wrapper for fetching and exporting a starter NCM model.

Delegates all checksum verification and export logic to the repository's pinned
scripts under `tools/reference/` without requiring a CPython source build.
For exact historical manifest-hash replay (which pins exact historical exporter
commits and Python/PyTorch versions), see `docs/USAGE.md`.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

MODELS = {
    "tinystories": {
        "tokenizer": "tinystories",
        "fetch_profile": "tinystories-15k-ema",
        "tinystories_release": "15k-ema",
        "budget_bytes": 1_200_000_000,
        "output_name": "tinystories-quickstart",
        "description": "TinyStories 15k EMA text generator (~552 MiB exported; use --precision bf16, q8, q4, or fp32)",
    },
    "tinystories-legacy": {
        "tokenizer": "tinystories",
        "fetch_profile": "tinystories",
        "tinystories_release": "v1",
        "budget_bytes": 1_200_000_000,
        "output_name": "tinystories-legacy-quickstart",
        "description": "TinyStories legacy d342ce8a text generator (~552 MiB exported; use --precision bf16, q8, q4, or fp32)",
    },
    "xsum": {
        "tokenizer": "gpt2",
        "budget_bytes": 1_500_000_000,
        "output_name": "xsum-quickstart",
        "description": "XSum BBC news summarizer (~627 MiB exported; great with --precision fp16)",
    },
    "wmt14": {
        "tokenizer": "qwen3",
        "budget_bytes": 2_500_000_000,
        "output_name": "wmt14-quickstart",
        "description": "WMT14 German-to-English translator (~1.06 GiB exported; great with --precision fp16)",
    },
    "lm1b-small": {
        "tokenizer": "bert",
        "budget_bytes": 2_200_000_000,
        "output_name": "lm1b-small-quickstart",
        "description": "LM1B Small unconditional language model (~943 MiB exported)",
    },
    "lm1b-medium": {
        "tokenizer": "bert",
        "budget_bytes": 4_000_000_000,
        "output_name": "lm1b-medium-quickstart",
        "description": "LM1B Medium unconditional language model (~1.74 GiB exported)",
    },
    "lm1b-large": {
        "tokenizer": "bert",
        "budget_bytes": 7_500_000_000,
        "output_name": "lm1b-large-quickstart",
        "description": "LM1B Large unconditional language model (~3.26 GiB exported; fits b=16 in ~2 GiB VRAM with --precision fp16)",
    },
    "sudoku-easy": {
        "tokenizer": None,
        "budget_bytes": 400_000_000,
        "output_name": "sudoku-easy-quickstart",
        "description": "Sudoku-easy solver (~155 MiB exported)",
    },
}


def build_plan(
    model: str,
    python_exe: str,
    models_dir: Path,
    upstream_dir: Path,
    repo_root: Path,
    reference_rope: Path | None = None,
) -> tuple[Path, list[list[str]]]:
    if model not in MODELS:
        raise ValueError(f"unsupported quickstart model `{model}`; choose from {', '.join(MODELS)}")
    spec = MODELS[model]
    ref = repo_root / "tools/reference"
    fetch_profile = spec.get("fetch_profile", model)
    ckpt_dir = upstream_dir / fetch_profile
    tok_raw = upstream_dir / "tokenizers-v2"
    tok_out = models_dir / "tokenizers-qualified"
    out_dir = models_dir / spec["output_name"]

    steps: list[list[str]] = []

    if spec["tokenizer"] is not None and not (tok_out / spec["tokenizer"]).is_dir():
        if not tok_raw.is_dir():
            steps.append(
                [
                    python_exe,
                    str(ref / "fetch_tokenizers.py"),
                    str(tok_raw),
                    "--lock",
                    str(ref / "tokenizer-sources.lock.json"),
                ]
            )
        steps.append(
            [
                python_exe,
                str(ref / "convert_tokenizers.py"),
                "--source",
                str(tok_raw),
                "--output",
                str(tok_out),
            ]
        )

    if not ckpt_dir.is_dir():
        steps.append(
            [
                python_exe,
                str(ref / "fetch_checkpoints.py"),
                "--profile",
                fetch_profile,
                "--output",
                str(ckpt_dir),
                "--budget-bytes",
                str(spec["budget_bytes"]),
            ]
        )

    if model == "sudoku-easy":
        steps.append(
            [
                python_exe,
                str(ref / "sudoku_artifact.py"),
                "--source",
                str(ckpt_dir),
                "--output",
                str(out_dir),
            ]
        )
    elif model in ("tinystories", "tinystories-legacy"):
        steps.append(
            [
                python_exe,
                str(ref / "tinystories_artifact.py"),
                "--release",
                spec["tinystories_release"],
                "--source",
                str(ckpt_dir),
                "--tokenizer-asset",
                str(tok_out / "tinystories"),
                "--output",
                str(out_dir),
            ]
        )
    elif model == "xsum":
        steps.append(
            [
                python_exe,
                str(ref / "xsum_artifact.py"),
                "--source",
                str(ckpt_dir),
                "--tokenizer-asset",
                str(tok_out / "gpt2"),
                "--output",
                str(out_dir),
            ]
        )
    elif model == "wmt14":
        steps.append(
            [
                python_exe,
                str(ref / "wmt14_runtime_artifact.py"),
                "--source",
                str(ckpt_dir),
                "--tokenizer-asset",
                str(tok_out / "qwen3"),
                "--output",
                str(out_dir),
            ]
        )
    elif model.startswith("lm1b-"):
        size = model.removeprefix("lm1b-")
        cmd = [
            python_exe,
            str(ref / "lm1b_small_runtime_artifact.py"),
            "--size",
            size,
            "--source",
            str(ckpt_dir),
            "--tokenizer-asset",
            str(tok_out / "bert"),
            "--output",
            str(out_dir),
        ]
        if reference_rope is not None:
            cmd.extend(["--reference-rope", str(reference_rope)])
        steps.append(cmd)

    return out_dir, steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and export one starter NCM model for quickstart CLI and serving."
    )
    parser.add_argument("--model", choices=sorted(MODELS), help="Model profile to fetch and export")
    parser.add_argument("--list", action="store_true", help="List available starter models and exit")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable with torch, safetensors, pyyaml, and tokenizers installed",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("artifacts/canonical"),
        help="Destination parent directory for exported NCM artifacts (default: artifacts/canonical)",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("artifacts/upstream"),
        help="Cache directory for verified upstream checkpoints (default: artifacts/upstream)",
    )
    parser.add_argument(
        "--reference-rope",
        type=Path,
        default=None,
        help="Optional LM1B rotary table safetensors to bundle self-contained lm1b-*-inference-v2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without downloading or writing files",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, info in MODELS.items():
            print(f"{name:12s}  {info['description']}")
        return 0

    if not args.model:
        parser.error("--model is required (or pass --list)")

    repo_root = Path(__file__).resolve().parents[2]
    out_dir, steps = build_plan(
        model=args.model,
        python_exe=args.python,
        models_dir=args.models_dir,
        upstream_dir=args.upstream_dir,
        repo_root=repo_root,
        reference_rope=args.reference_rope,
    )

    if out_dir.exists():
        print(f"Artifact already exists at {out_dir}; skipping export.")
        return 0

    print(f"Target artifact: {out_dir}")
    for idx, cmd in enumerate(steps, 1):
        print(f"[{idx}/{len(steps)}] {shlex.join(cmd)}")

    if args.dry_run:
        return 0

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.upstream_dir.mkdir(parents=True, exist_ok=True)
    for cmd in steps:
        subprocess.run(cmd, check=True)
    print(f"\nExport complete: {out_dir}")
    print(f"Next: ./target/release/ncm list --models-dir {args.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
