# Research and upstream audit

Inspected 2026-11-09. This is a source-level investigation, not a reproduction of
paper results. No checkpoints were downloaded or executed during scaffolding.

## Sources and reproducibility

- [Paper, arXiv:2610.14162v1](https://arxiv.org/html/2610.14162v1): Sections 3–4,
  architecture Appendix C, ablations/sampling Appendix F, limitations.
- [Hazy Research explainer](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm).
- [Upstream code, pinned commit](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203):
  README, all four model modules plus common layers, all evaluation configurations,
  tokenizer implementations, evaluation shard/merge scripts and registry.
- [Checkpoint repository](https://huggingface.co/hazyresearch/ncm-checkpoints/tree/d342ce8aaea58830eef3b8ad473664673e41f692),
  revision `d342ce8aaea58830eef3b8ad473664673e41f692`: API file inventory,
  model card, XSum config and OWT TODO inspected; weight contents not inspected.
- [Evaluation data](https://huggingface.co/datasets/hazyresearch/ncm-datasets):
  referenced by upstream; data revision, sizes, and licenses still need inventory.
- [mlx-rs](https://github.com/oxiglade/mlx-rs/tree/adf439bf5361d8a02955de7aaf651b0c45900945):
  README, native build documentation, attention and convolution APIs. README
  advertises 0.34.0; inspected native tuple uses MLX 0.32.2. Old `oxideai/mlx-rs`
  URL redirects here. These are research pins, not yet dependency selections.
- [Candle](https://github.com/huggingface/candle/tree/cfcbd7186d13271843ce7eb41d25659bd0835850):
  README, convolution/interpolation/device APIs, CUDA implementation and FlashAttention wrapper.

## What NCM actually does

NCM is not ordinary next-token autoregression, a sentence-embedding LCM, or diffusion.
A multiscale residual VQ-VAE learns a hierarchy of discrete code sequences. Each
level owns a codebook. At level k, downsample the remaining latent residual,
choose nearest codebook entries, upsample and refine their contribution, and add
it to the full-length cumulative reconstruction. The next level corrects what
previous levels missed.

Freeze the codec, then train a transformer to predict all codes in a level in
parallel from the cumulative reconstruction of earlier levels resized to the new
level length. Levels are sequential, but positions within one level are parallel.
All levels can be teacher-forced together using block-diagonal attention. A final
bidirectional codec decoder maps the reconstruction back to ordinary token logits.
Conditional tasks use a prefix-LM mask: prefix sees prefix, target sees prefix and
its own level, prefix never sees target. Fine-level dropout and conditioning-code
corruption are training strategies, not extra inference denoising steps.

The paper reports substantial speedups over particular diffusion baselines on an
H100. Those are **not Rust-vs-Python speedups**, nor guarantees on Apple Silicon or
an 8 GiB consumer GPU. Decode lengths are checkpoint-specific; this is not yet an
arbitrary-length streaming chat model. Intermediate levels are revised drafts,
not committed autoregressive tokens. The codec itself is lossy.

## Source map

| Upstream | Port responsibility |
| --- | --- |
| `models/common.py` | LayerNorm, QK RMSNorm, split-half RoPE, attention, MLP and residual blocks |
| `models/vae.py` | Codec encoder/decoder, bidirectional/local attention, optional sequence compression |
| `models/quant.py` | Per-level codebooks, exact resampling, Phi residual convolutions, accumulation, EMA training pieces |
| `models/vqvae.py` | Pre-quant normalization and complete encode/decode pipeline |
| `models/next_concept.py` | Prefix conditioning, level-zero branches, output heads, CFG, sampling and generation |
| `tokenization/*` | Text-to-ID adapters distinct from the learned VQ codec |
| `scripts/eval_shard.py` | Checkpoint metadata, initial unigram codes, final-token penalties, EOS policy |
| `scripts/eval_merge.py`, `evaluation/gen_ppl.py` | Task-specific scoring and perplexity protocols |
| `config/eval/*` | Actual release architecture and sampling profiles |

## Release compatibility inventory

Numbers below come from pinned **code configs**, not paper summary tables.
`P` = prefix length; `L` = codec output token length; `K` = hierarchy levels.
All inspected release configs use block-diagonal attention, RoPE, QK norm,
level embeddings, raw token prefixes, compression factor 1, and no context encoder.

| Profile | Generator depth / width / heads | P / L / K | Level zero in eval | Extra requirements |
| --- | --- | --- | --- | --- |
| Sudoku easy/medium/hard | 8 / 512 / 8 | 91 / 89 / 12 | Learned queries | 16-entry codebooks, untied heads, greedy, custom 12-symbol tokenizer |
| TinyStories | 12 / 768 / 12 | 1 / 128 / 12 | Checkpoint unigram | 4096-entry codebooks, distance-tied heads, custom byte BPE, learned cascade |
| XSum | 12 / 768 / 12 | 512 / 48 / 9 | Learned queries | GPT-2 + added PAD, distance-tied heads, cascade GroupNorm, token frequency/adjacency penalties |
| WMT14 | 12 / 768 / 12 | 96 / 64 / 11 | Prefix prediction | Qwen3 tokenizer, untied bias-free heads, cascade GroupNorm, optional CFG |
| LM1B small | 12 / 768 / 12 | 1 / 128 / 12 | Checkpoint unigram supplied despite query support | BERT tokenizer, shared head + per-level bias, post-upsample conv, four-block positional decoder, penalties |
| LM1B medium | 24 / 1024 / 16 | 1 / 128 / 12 | Same as small | Shared tokenizer checkpoint in parent directory |
| LM1B large | 36 / 1280 / 20 | 1 / 128 / 12 | Same as small | Memory budgeting essential |
| OWT (config only) | 16 / 1024 / 16 | 1 / 1024 / 32 | Checkpoint unigram | 8192-entry codebooks, tied head with bias, post-upsample conv, no EOS truncation |

Sudoku ladder: `[1,2,4,6,9,13,18,26,37,50,67,89]`.
TinyStories/LM1B ladder: `[1,8,16,24,32,40,52,64,76,92,108,128]`.
XSum ladder: `[4,6,10,14,20,26,32,40,48]`.
WMT14 ladder: `[1,4,8,14,20,26,32,40,48,56,64]`.
OWT ladder: squares 1² through 32².

Important discrepancies/blockers:

- Paper tables describe XSum with eight levels; released config (also checked on HF)
  has nine. Paper OWT dimensions/codebook size differ from repository config.
  Do not hard-code paper tables or claim reproduction from config inspection.
- Despite README links, the inspected HF tree contains only `owt/TODO.md`, with
  content `TODO`, and no OWT weights/config. Treat OWT as artifact-blocked.
- There is no complete training entry point, data pipeline, or reproducible trainer
  in the inspected tree. See [issue #10](https://github.com/HazyResearch/ncm/issues/10).
  EMA/VQ forward code is not a complete released training recipe.

## Numerical and semantic traps

1. **Resampling:** `area` means PyTorch adaptive averaging, including overlapping
   bins for non-divisible sizes; not generic fractional-overlap integration.
   `linear` defaults to `align_corners=False`. Learned first-level down/up samplers
   use Conv1d/ConvTranspose1d; the cascade factorization includes prime strides
   (Sudoku 89), not just powers of four. Later levels use interpolation, not nearest.
2. **Layouts:** model latents are BLC; PyTorch convolutions use BCL and OIK weights;
   MLX conv1d uses BLC and OKI weights. ConvTranspose layouts need independent tests.
3. **Norms:** transformer LayerNorm epsilon is 1e-6; codec pre-quant GroupNorm /
   native LayerNorm defaults are 1e-5. GroupNorm reduces over channels per group
   AND sequence, not per token. QK RMSNorm accumulates in FP32 before RoPE.
4. **Positions:** RoPE rotates two halves, not adjacent pairs. Positions reset for
   each target level; prefix positions are separate. Codec has its own positional
   embedding rules. Codec `n_layers` means attention HEAD count, not block depth.
5. **Heads:** tied heads use projected negative squared distances, codebook mean
   squared-norm scaling and learned inverse temperature—not ordinary tied dot products.
   Codebook-derived buffers are nonpersistent; generator state excludes the codec.
6. **Level zero:** supplied/unigram codes, learned queries, and prefix-derived
   predictions must be separate modes. LM1B eval overrides its query mode by passing
   sampled initial codes. Count generator calls separately from hierarchy levels.
7. **Sampling:** temperature zero is argmax; top-k threshold retains ties; top-p
   uses `(cumulative_probability - current_probability) > p`. CFG grows by level:
   `t = scale * level / max(K-1,1)`, then `(1+t)*conditional - t*null`.
   Equal seeds across Torch/MLX/CUDA do not imply equal categorical samples.
8. **Final output:** XSum/LM1B apply sequential frequency/deficit/adjacency penalties
   in FP64 *after* parallel decoder logits. This step is part of release behavior.
   Metal FP64 support cannot be assumed; CPU f64 is a compatibility fallback to
   measure, with device FP32 allowed only as an explicitly tested fast mode.
9. **Artifact loading:** upstream uses `torch.load(weights_only=False)` and pickle
   data/BPE. Production Rust must never execute pickle or load arbitrary Python.
   Export the `inference` metadata as well as `model` weights.
10. **Tokenizer identity:** GPT-2, BERT and Qwen special tokens, padded vocab rows,
    HF postprocessors and cleanup settings matter. TinyStories uses byte-valued
    vocab/merge entries, not interchangeable generic GPT-2 files.

## Upstream issues: preserve intended math, not accidental bugs

- `idx_to_zhats` appends the same in-place-mutated accumulator for every `z_hats`
  entry. Confirmed by source and [issue #11](https://github.com/HazyResearch/ncm/issues/11).
  Final reconstruction is usable; intermediate snapshots need an explicit corrected
  oracle with cloned tensors. `z_levels` is a different return value.
- The cached generation path rejects query-level-zero and omits prefix-pad bias
  in its prefix/target attention. Baseline on uncached, dense evaluation; test a
  new prefix cache against that baseline, not the upstream cache implementation.
- [Issue #23](https://github.com/HazyResearch/ncm/issues/23) reports flex-mask padding,
  query-level-zero length, and disabled level-encoding bugs; inspected source agrees.
  These paths are not used by released evaluation configurations.
- `VQVAE.reconstruct` unpacks three results from a quantizer returning five. Use
  `encode`/`decode` or explicitly corrected fixtures, not that convenience helper.
- Block-causal training masks exist, but generation currently visits only prefix
  plus the current target level. Do not advertise block-causal generation parity.
- Paper training loss/dropout descriptions and model-side forward details need
  reconciliation before promising training reproduction.

Maintain a fixture-level record distinguishing unmodified release behavior from
minimal, documented reference fixes. No silent bug-for-bug emulation.

## Runtime feasibility findings

- MLX bindings expose conv1d/transposed conv, normalization, attention, lazy
  evaluation and compilation. Pin the Rust/native tuple together; test actual
  multi-position masked attention performance rather than assume an API named
  `fast` accelerates every NCM shape. Native builds need compatible Xcode/Metal
  tools or a matched prebuilt library; package metallib/dylibs correctly.
- Candle is a Rust-native model/tensor stack with CUDA support, custom-op hooks,
  and an optional FlashAttention path. Its `interpolate1d` is nearest-neighbor,
  and that nearest CUDA path is explicitly unsupported at the inspected revision.
  NCM needs exact area/linear ops regardless: composite gather/reduce first,
  specialized kernels later. FlashAttention mask/dtype/shape support needs a spike.
- Existing [independent MLX experiments, issue #19](https://github.com/HazyResearch/ncm/issues/19)
  point to [ncm-modular](https://github.com/Red-fuchs/ncm-modular). The issue reports
  promising codec parity, not a validated Rust port; its training is explicitly a
  reimplementation. Treat these reports as leads, not acceptance evidence.
- Local hardware available during research: Linux x86_64, RTX 3060 Ti 8 GiB,
  NVIDIA driver 615.78.08, CUDA compiler on PATH. Rust 1.99.1. No Apple Silicon
  execution was available. Neither GPU runtime was built/tested in Phase 0.
