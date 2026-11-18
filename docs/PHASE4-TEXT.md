# Phase 4 text-profile contract

Phase 4 implements the released profiles in [plan order](PLAN.md#phase-4--released-text-profile-compatibility): TinyStories, XSum, WMT14 and LM1B (small, then medium and large). OpenWebText remains limited to synthetic/config/operator coverage until its checkpoint is released and validated. Complete Linux/CUDA correctness, quality and performance evidence for each profile before Apple/MLX runs; retain the full preceding regression suite. Do not infer support from tokenizer loading or a successful build.

## Per-profile acceptance

- Retain the pinned upstream code, checkpoint and tokenizer identities. Export exact inference metadata and tensor layouts to a documented, non-executable data format. Check hashes, sizes, types and profile/architecture agreement before GPU uploads; preserve aliases deliberately and classify training-only tensors without silently discarding inference state. Unknown inference behavior fails explicitly. Existing Sudoku profiles, limits, data and goldens remain unchanged.
- Qualify tokenizer IDs, padding, special-token and decode behavior; trace the teacher-forced codec and generator at every level against the pinned Python oracle. Test greedy/ties and stochastic selection with controlled inputs or draws, without assuming that identical seeds imply identical random samples across runtimes. Record numerical budgets and near ties rather than changing goldens or tolerances to make a run pass.
- Compare the official task quality protocol, preserving generated IDs and enough data to repeat scoring. For TinyStories use the configured five seeds and 256 samples per seed with the pinned GPT-2-large generative-perplexity and entropy evaluator. Define and freeze a justified non-inferiority margin before held-out quality comparisons; reference variation, baseline uncertainty and metric behavior must inform it. Do not label missing evaluator or hardware evidence as passing.
- Follow [REPORTING.md](REPORTING.md): comparable synchronized generation/readback scopes, at least five warmups and 30 measured batches in three alternating fresh processes where feasible, with fixed inputs and sampling parameters, median/p95 and dispersion, plus host RSS and device allocation/cache observations with their scope and limits. Report startup/load separately. User requires Rust to outperform the corresponding Python baseline and use less memory: compare median latency/throughput and whole-process peak RSS, and separately report native device peak and retained/cache counters for each representative batch. A cell that is slower or uses more memory remains a visible failure or exception for review, not an aggregate pass. Do not equate CUDA pool, PyTorch allocator, MLX active/cache/peak or total physical unified memory.
- CUDA uses the pinned original Python implementation on the same device. Apple uses a separately qualified same-device Python MLX adaptation or another defensible reference, clearly labeled; quality remains anchored to original Python. Desktop activity, GPU telemetry limitations, OOMs and lack of concurrent-request throughput measurement must be disclosed. Shared-generator request serialization from Phase 3 is a correctness safeguard, not evidence of parallel throughput.

## TinyStories inputs and current boundary

The pinned source revision is `d342ce8aaea58830eef3b8ad473664673e41f692`; upstream code remains `c70274273f489abc73832b2b98584a8483b6b203`. Offline source checkpoint identities are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `generator.pt` | 356,160,651 | `0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff` |
| `tokenizer.pt` | 221,861,835 | `76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81` |
| `config.yaml` | 2,483 | `4009baccad5a159bfc12cd48f3da17c22c4a281b86f71f2fc70b7c1bbb62004b` |

The existing tokenizer-source lock pins the BPE vocab and merge inputs; they are not executed by production. A CPU-only `torch.load(weights_only=True, mmap=True)` inspection of these exact files found 197 generator tensors (356,071,472 bytes), one 4,096-element FP32 `inference.level0_unigram` (16,384 bytes), 70 codec inference tensors (145,739,264 bytes), and 36 EMA/count/hit tensors (76,087,296 bytes). This inspection is not a quality claim. The unigram is finite, nonnegative and sums to one in the inspected checkpoint. Tensor inventory and model metadata must be verified in the exporter, not inferred from checkpoint file sizes.

The separate TinyStories artifact profile uses 384 MiB per-file and 640 MiB total limits; the Sudoku profile retains 256 MiB and 512 MiB. The exact local export at `artifacts/canonical/tinystories-phase4-v1/` has manifest SHA-256 `0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1`. The exporter verifies bit-exact roundtrips and classifies inference metadata separately. Host checks accept the actual 70 codec inference tensors, 195 generator parameters, two persistent generator buffers and unigram, and embedded tokenizer bytes match the pinned TinyStories fixtures. Width-384 codec configuration is separately validated. A fixed-code TinyStories teacher-forced prototype and pinned Python CUDA trace have an [initial comparison](results/2026-11-15-phase4-tinystories/cuda-teacher-probe-v1.md). CUDA stochastic generation is implemented and qualified on controlled draws; padded prefixes remain unsupported. The opt-in strict-cuDNN path has a further [numerical investigation](results/2026-11-15-phase4-tinystories/cuda-numerical-investigation-v1.md), including 80 generated-input development traces; it does not imply exhaustive parity.

The existing exact pinned bytes are held locally in `artifacts/upstream/tinystories-phase4/`; weights and data are not Git content. Python and Rust teacher-forced diagnostics and stochastic quality comparisons have run on CUDA. Results and limitations are recorded in the [held-out quality report](results/2026-11-15-phase4-tinystories/heldout-quality-v1.md) and [initial performance report](results/2026-11-15-phase4-tinystories/cuda-performance-v1.md). The [strict-path performance comparison](results/2026-11-15-phase4-tinystories/cuda-strict-performance-v1.md) also favors Rust in the measured latency and RSS cells; native allocator peak/retained observations are documented with differing coverage. Together with the expanded strict-path numerical checks these satisfy the scoped Linux/CUDA evidence for TinyStories, while leaving the documented activity and allocator limitations. Mac inference, quality and performance investigations have since run; the [Metal observations](results/2026-11-15-phase4-tinystories/mlx-development-v1.md) and [lifecycle comparison](results/2026-11-15-phase4-tinystories/mlx-lifecycle-performance-v3.md) document remaining numerical and latency-tail questions. The user subsequently [accepted the scoped unresolved TinyStories exceptions for advancement](results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md); the failed numerical band and historical slowdown remain recorded.

## TinyStories CLI candidate

The `sample-tinystories` command exposes one seeded batch of 1–128 one-token prefixes as a JSON array of arrays. It returns the raw code and token-ID rows and the requested seed. It uses `Artifact::load_inference`, explicitly selects the requested backend and returns errors rather than falling back. The `cuda-cudnn` device requires the `ncm-cli/cuda-cudnn` build feature and native cuDNN; ordinary `cuda` is a separate, non-strict path. For example, after building with `cargo build -p ncm-cli --release --features cuda-cudnn --locked`, run `ncm sample-tinystories ARTIFACT cuda-cudnn PREFIXES_JSON 42`. On Apple Silicon, build with `--features mlx` and select `mlx`.

Sampling is the existing temperature-1, top-k-0, top-p-1, CFG-0 implementation using SplitMix64 and inverse CDF. Its seed does not reproduce PyTorch's RNG stream; padded prefixes and other sampling settings remain unsupported. The CLI returns IDs rather than decoded text. It drops the unused logits before readback, but does not set a Metal process-global cache limit or promise the benchmark's cleanup barrier; the dedicated-process lifecycle timings and memory figures must not be attributed to this command without a separate measurement. At commit `45b5252`, `tools/bench/check_tinystories_cli.py` compared the CLI output with the first 128 rows of the retained scored seed-42 files on strict cuDNN and Metal: all codes and IDs matched. Result manifests are retained at `reports/phase4-tinystories-cli-cuda-v2/result.json` (SHA-256 `e92b581c4690ce1357a16945b98f2e27500275ec791863ac9c92fb4d0cf73327`) and `reports/phase4-tinystories-cli-mlx-v1/result.json` (SHA-256 `ba4fcfd8e6a8f6d5f5b2215672abbf3396386b112d8dc9f56d16330313db0d8f`). The full CUDA and Metal checks passed, including 18 real-device operator cases on each; raw logs and limitations are recorded in the context map.

## TinyStories quality evaluator preparation

The upstream metric names `gpt2-large`. For this Phase 4 comparison its evaluator is newly pinned to [`openai-community/gpt2-large` revision `32b71b12589c2f8d625668d2335a01cac3249519`](https://huggingface.co/openai-community/gpt2-large/tree/32b71b12589c2f8d625668d2335a01cac3249519); its 3,247,159,078-byte `model.safetensors` SHA-256 is `5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce`. The evaluation-only Transformers version is `4.57.3`; installed dependencies are recorded in ignored `reports/phase4-evaluator-install-v1.log`. The pinned model and tokenizer files are in ignored `artifacts/upstream/gpt2-large-phase4/`, and evaluator dependencies are isolated in `.reference/phase4-eval-deps/` rather than the production Rust runtime.

`tools/reference/tinystories_quality.py` verifies all six evaluator files and checks both implementations' raw IDs, decoded text and re-encoded BPE tokens against pinned upstream before applying the upstream `generation_perplexity` and entropy functions. A four-sample Python/Rust development pipeline passed that preflight. The official evaluator batch is 32. The official seeds `[42, 0, 1, 2, 3]` were evaluated after separately generated development seeds informed a fixed non-inferiority threshold. The [quality report](results/2026-11-15-phase4-tinystories/heldout-quality-v1.md) identifies the original comparison and the subsequent strict-path retest; the latter is not a previously unseen held-out test.

## Later-profile checkpoint preparation

Read-only repository metadata at the existing checkpoint revision confirms XSum requires a 354,355,211-byte generator and a 299,626,079-byte tokenizer checkpoint. These and its 2,345-byte config are downloaded and SHA-256 verified under ignored `artifacts/upstream/xsum-phase4/`; their respective hashes are `f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4`, `1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7`, and `5dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493`. CPU `weights_only=True` inspection finds 194 generator model tensors and required `inference.token_unigram`, plus 80 tokenizer checkpoint tensors. The XSum profile requires padded 512-token prefixes, query level zero and sequential output penalties; this inspection does not implement or qualify them. A separately bounded [XSum artifact export and embedded tokenizer check](results/2026-11-15-phase4-xsum/artifact-export-v1.md) now preserve the exact checkpoint tensors and GPT-2 assets; they do not implement model inference. The [first-level codec cascade](results/2026-11-15-phase4-xsum/cuda-cascade-v1.md) now has a limited real-CUDA component comparison; the full profile gate remains open.

The later XSum strict-CUDA implementation now covers the pinned top-1 setting, padded prefixes, output penalties and restricted CFG. Its [tie-enabled full and held-out quality comparisons](results/2026-11-15-phase4-xsum/cuda-reference-ties-v1.md) match all Python output IDs and quality metrics; one intermediate-code disagreement was investigated as a Python head tie. Current-source [repeated performance](results/2026-11-15-phase4-xsum/cuda-reference-tie-repeated-v1.md) favors Rust versus Python in measured latency, peak process RSS and sampled driver GPU memory at batches 1/4/32. Rust was 7–18% slower than in the preceding run, but a same-condition paired control did not reproduce a tie-change slowdown; the earlier cause remains unknown. Experimental stochastic sampling has [controlled-draw differences](results/2026-11-15-phase4-xsum/cuda-stochastic-investigation-v1.md) and unqualified quality. These results do not establish the full XSum or MLX gate.

On Apple Silicon, a CLI built with the `mlx` feature accepts an explicit
positive CFG scale for experimental XSum top-1 generation:

```sh
ncm sample-xsum artifacts/canonical/xsum-inference-rope-v2-final mlx prefixes.json 1
```

The input is a JSON array of 1 to 64 padded 512-token prefixes. The command
returns one `codes` row and one raw `ids` row per prefix. Metal uses the
CPU FP64 final selector. Omitting the scale uses zero, the setting of the
official XSum evaluation. Seeded Metal generation requires scale zero.
The [retained guidance fixtures](results/2026-11-15-phase4-xsum/mlx-cfg-probe-v1.md)
cover scales 0, 1 and 2. The official full-corpus Metal exact gate remains open.
The user approved [advancing to WMT14 with the listed retained differences](results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md).

WMT14 top-1 generation is available through the experimental CLI.
Build with `cuda-cudnn,wmt14-experimental` for the explicit `cuda` device
or `mlx,wmt14-experimental` for `mlx` on Apple Silicon.

```sh
ncm sample-wmt14 artifacts/canonical/wmt14-inference-v1-final cuda prefixes.json 1
```

Supply a JSON array of padded 96-token WMT14 prefixes. The CLI accepts up
to 32 prefixes at the default scale zero and up to 16 at a positive scale.
It emits 313 intermediate codes and 64 raw token IDs per row. CUDA uses the
[one-pass final selector](results/2026-11-16-phase4-wmt14/cuda-fast-argmax-plan-v1.md).
On Apple Silicon, select `mlx-retry-padded` to opt into one bounded
retry for batches of one to four prefixes. A tied or nonfinite head
repeats input rows to eight internally. The same head check runs on
the padded batch and the command returns only the original rows.
Successful recovery is reported on standard error.

```sh
ncm sample-wmt14 artifacts/canonical/wmt14-inference-v1-final mlx-retry-padded prefixes.json 1
```

WMT14 Metal qualification is in progress.

The WMT14 and LM1B checkpoints at the same immutable revision have been fetched to ignored `artifacts/upstream/wmt14-phase4/` and `artifacts/upstream/lm1b-phase4/` and verified against their published SHA-256 values:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `wmt14/generator.pt` | 480,582,539 | `ae580cbbb400a27b30c616634745d4c9cfa8b66806f936c6c26041e5b9640d4f` |
| `wmt14/tokenizer.pt` | 645,790,351 | `74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702` |
| `lm1b/small/generator.pt` | 390,138,443 | `b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe` |
| `lm1b/medium/generator.pt` | 1,268,014,283 | `5aa4621472c2e658ab343e50d98abc808e4c2c6dd26eaf8535897b6a540c1604` |
| `lm1b/large/generator.pt` | 2,901,184,267 | `ec34d6c56a81f428303b2fa3a8e1615ceee81b7dec7f59380427f46808ec224c` |
| `lm1b/tokenizer.pt` | 597,840,331 | `d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c` |

At the same existing checkpoint revision, the local WMT14 and LM1B config
files were independently fetched and matched byte-for-byte. Their SHA-256
identities and source links are recorded in the [WMT14](results/2026-11-16-phase4-wmt14/inventory-preparation-v1.md)
and [LM1B](results/2026-11-16-phase4-lm1b/inventory-preparation-v1.md) preparation
records. This verifies those config identities; it does not change the revision
or qualify a runtime artifact.

These downloads need explicit per-profile memory bounds, implementation and GPU validation; they do not establish support. The pinned OWT directory still contains only `TODO.md`; no released checkpoint or quality gate is asserted.

The original GPT-2-large scorer ran out of memory on this 8 GiB GPU at batch 32 with the desktop present. A proposed memory-bounded scorer keeps the upstream batch-32 tokenizer, attention masks and FP32 transformer forward pass, but projects/log-softmaxes output rows in smaller chunks and disables unused KV-cache output. Before using it for any held-out score, compare each sample's token count and mean NLL with the unmodified upstream scorer on development inputs at a batch size that fits. Require exactly equal token counts and maximum absolute per-token mean-NLL difference at most `1e-4`; record the measured differences. This is a newly declared evaluator-implementation tolerance, not a change to a previously passed oracle. Preserve and disclose the original batch-32 OOM and the chunked evaluator rather than calling it an unmodified run.

The separate development set was subsequently scored with the qualified memory-bounded batch-32 implementation. Results, the retained unmodified-scorer OOM, and the thresholds fixed before official-seed sampling are in [the development report](results/2026-11-15-phase4-tinystories/development-quality-v1.md).

For later XSum and WMT14 evaluation, read-only Hugging Face dataset metadata at the already used Phase 3 revision `6580ba41a4b0a976dde876fbe84d25327617b0e6` identifies these files; this is metadata preparation, not an evaluation or a new dataset revision:

| Dataset file | Bytes | SHA-256 |
| --- | ---: | --- |
| `xsum/passages.pickle` | 18,506,302 | `39b820f0a71ddd6ad178b2700eeae6d035729053b1b2f115d3a92810c6ebf4f8` |
| `xsum/reference_summaries.pickle` | 1,456,199 | `db624e065b6685dfea25db9106752e837f06f26b799763593759efd150fe554a` |
| `wmt14/sources.pickle` | 406,100 | `62254653c32ce9e8b93576797597381babaab301b10ee0a2a3babec89576ea43` |
| `wmt14/reference_translations.pickle` | 366,263 | `3dcef81d8aeded13f79917cd6b794d2ca9f33223bf35c1fff1c5643eae0df52b` |

All four files were fetched into ignored `artifacts/upstream/text-evaluation-6580ba41/` and verified against those sizes and hashes. `tools/reference/text_phase4_data.py` checks an opcode allowlist, forbids pickle globals and persistent references, checks exact row shapes and writes non-executable JSON. The locally converted XSum file has 11,334 560-token rows and references (SHA-256 `cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a`); WMT14 has 3,003 source/reference text pairs (SHA-256 `36035390ce542834f8dfeeac004c1b532aff247a4ff39910f4a16efd009ca47c`). Outputs are ignored under `artifacts/canonical/text-phase4-v1/`; conversion does not establish model support or an evaluation result. The pinned XSum upstream decoder applies frequency, deficit and adjacency adjustments sequentially in float64 before argmax, so [MLX's lack of GPU FP64 support](https://ml-explore.github.io/mlx/build/html/python/data_types.html) requires a qualified exact or explicitly reviewed alternative rather than silent float32 substitution.
