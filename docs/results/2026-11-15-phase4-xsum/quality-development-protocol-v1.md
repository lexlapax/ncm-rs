# XSum development baseline and held-out quality rule

The development set is input indices 0–255 of the pinned 11,334-row XSum data.
It includes rows already used in performance and initial correctness work and is
not held out. The original two-shard stride and batch-32 ordering are preserved.
Upstream conditional generation does not set an evaluation seed; this reference
explicitly records seeds 42 and 314159 for reproducibility. On all 256 rows,
the two pinned-Python CUDA runs produced identical codes and raw IDs. This is
limited evidence of stability on these inputs, not a claim for every input.

| Python development metric | Value |
| --- | ---: |
| Stemmed ROUGE-1 / ROUGE-2 / ROUGE-L | 34.5944 / 12.7460 / 27.9333 |
| Mean output words | 20.9219 |
| GPT-2 generative perplexity | 80.3858 |
| GPT-2 scored tokens | 6,346 |

For baseline sampling uncertainty, a 10,000-replicate nonparametric bootstrap
over these 256 rows with NumPy `default_rng(20261116)` gave 95% percentile
intervals of 32.913–36.287, 11.320–14.269 and 26.351–29.579 ROUGE points,
and 20.418–21.434 words. These describe the small development sample, not a
held-out confidence interval or a quality margin. Perplexity uncertainty has
not been estimated on the development sample.

The scorer's decoded text agreed with the pinned upstream `HFTokenizer` for
all 256 outputs, and a direct independent aggregation using the upstream
stemmed ROUGE scorer reproduced the values to floating-point rounding. The
pinned upstream `generation_perplexity` called directly on all 256 texts with
batch size 32 returned exactly 80.3857757865936 and 6,346 scored tokens,
matching the batch-wise scoring wrapper. The evaluator dependencies and GPT-2
model are explicitly pinned in [the preparation record](quality-evaluator-preparation-v1.md).
The Python generator's resumed and uninterrupted two-batch outputs were
byte-identical, including saved random states. These checks do not establish a
held-out result. The initial Rust development run matched all 51,200 codes and
12,288 IDs across 256 rows; its runner was then changed to address a lint
finding, so that result is retained as pre-fix evidence. The rebuilt runner
also matched all 51,200 codes and 12,288 IDs on those rows; its per-row scores,
ROUGE, perplexity, token count and mean words matched the Python version-2
scorer result exactly.

## Frozen rule before held-out comparison

For this deterministic top-k-1 profile, require **exact raw output-ID equality**
for every paired row, and record all intermediate-code disagreements and near
ties for investigation. On the held-out indices 256–11,333, this makes the
predeclared allowed ROUGE deterioration **0.0 points**, the allowed increase
in generative perplexity **0.0**, and the allowed change in mean words **0.0**:
the same texts scored by the same pinned evaluator must produce the same
metrics. Any output-ID difference or unequal score is a visible failure to
investigate, not an automatic tolerance change. Report both the held-out
comparison and the official full 11,334-row result, including the development
rows, along with disagreement counts and uncertainty where meaningful.

This deliberately strict rule uses the observed identical seed outputs and the
upstream algorithm's deterministic top-k-1 setting; the wide bootstrap
intervals quantify variation across articles rather than license a difference
between implementations on the same articles. The native CUDA path explicitly
rejects exact head ties and non-finite scores, so such cases remain failures to
investigate rather than being silently assigned an output. The quality rule is
not a change to an existing numerical tolerance and does not by itself satisfy
the separate CFG, stochastic/tie, performance, device-memory or MLX gates.

The ignored baseline outputs are `reports/phase4-xsum-quality-python-smoke-s0/`,
`reports/phase4-xsum-quality-python-dev-s1/` and
`reports/phase4-xsum-quality-python-dev-seed314159-s{0,1}/`. The seed-42 ROUGE
result SHA-256 is `a8694db4eee569b3463b68e4c30dd9fb33efd5e2ed8b1df1190f7abf4e65beb9`,
the seed-314159 ROUGE result is `00f7827124c83880373646d3e61a6cb794f0376eccd39512ddd339324c93fccb`,
and the seed-42 full metric result is `17d2587cb7e0de6aa697cb07f730ef320115b8dc8cc22c3d460db03843b9a19f`.
The direct perplexity log is `reports/phase4-xsum-quality-ppl-direct-v1.log`
(SHA-256 `8fd4c660252a19d1535f48c24552841d2ac7380af5fd5b81206094bba9e518c5`).
Generator and scorer source SHA-256 values are respectively
`dfda5bb67e576878406540fafc8089941b788191ac79d0242123bb20314514d2` and
`69df36b15aad77649784a2afdbf141ef04a867913cae79016a3d01eefc0b948e`.

The scorer was subsequently extended with explicit `--start` and `--stop`
row bounds for held-out scoring. Its output schema is explicitly version 2,
adding the start index; the current scorer SHA-256 is
`58d9b22ccae22d9e428c94034d58882a1474986e72ac6e46738f82587fca95a0`.
On the same 256 development rows it produced exactly the same per-row outputs
and full metrics, including perplexity, as version 1. The version-2 full result
SHA-256 is `dff6ea06ab7107cba004376a542d0a51e7b0e98b705798e566e3314e21aa2927`;
an additional `--start 1 --stop 256` check selected exactly the expected 255
rows. Version 1 and its results remain retained, not overwritten.
The Rust development version-2 full result is
`reports/phase4-xsum-quality-rust-dev-v2-full.json` (SHA-256
`a49db6347bdad137c161d3fa6d702ebe216e477631455a44e9ce7d461093ef49`).
The explicit comparison below reported zero code, ID, text and per-row ROUGE
differences and equal aggregate metrics on those development results. Its
output SHA-256 is `090714881203eab501b1e44d9ee78dc2a759417655e5779e56671370c7a75241`.
Two additional CPU-only synthetic tests check the comparison's exact-output
rule, code-difference reporting and rejection of mismatched inputs.

```sh
.reference/venv/bin/python tools/reference/xsum_quality_compare.py --python reports/phase4-xsum-quality-python-dev-seed42-full-v2.json --rust reports/phase4-xsum-quality-rust-dev-v2-full.json --output reports/phase4-xsum-quality-dev-comparison-v1.json
PYTHONPATH=tools/reference .reference/venv/bin/python -m unittest -v test_xsum_quality_compare
```

The generator accepts `--upstream .reference/upstream-ncm`,
`--source artifacts/upstream/xsum-phase4`,
`--data artifacts/canonical/text-phase4-v1/xsum.json`, `--output`, `--shard`,
`--seed`, and `--stop 256`; each output directory is source-bound and resumable.
Run `tools/reference/xsum_quality_score.py` with both shard directories, the
same data and upstream paths, `--stop 256`, and `--with-ppl` for the full
metric. Run Python through `.reference/venv/bin/python` with `OMP_NUM_THREADS=1`.

The first Rust quality-tool build failed because the linker could not find
`libcudnn.so`; its log is `reports/phase4-xsum-rust-quality-build-v1.log`.
The successful build used the existing local cuDNN link:

```sh
LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 cargo build --release --locked -p ncm-backend-cuda --features cuda-cudnn --example xsum_quality_batches > reports/phase4-xsum-rust-quality-build-v2.log 2>&1
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_rust.py --data artifacts/canonical/text-phase4-v1/xsum.json --output reports/phase4-xsum-quality-rust-smoke-s0 --shard 0 --stop 64 > reports/phase4-xsum-quality-rust-smoke-s0.log 2>&1
PYTHONPATH=tools/reference .reference/venv/bin/python -m unittest -v test_xsum_quality_generate test_xsum_quality_score
```

The Rust tool's pre-fix output is under the indicated ignored report directory;
the seven Python tests are CPU-only synthetic input-integrity checks. The full
`CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 tools/check.sh --cuda`
passed, including 18 real-device operator cases, in
`reports/phase4-xsum-quality-tool-check-v1.log`. A separate strict-cuDNN example
Clippy run first found two `chunks_exact` style warnings
(`reports/phase4-xsum-rust-quality-clippy-v1.log`); after changing only output
chunk serialization, the targeted lint and rebuild passed in the corresponding
`-v2` and `build-v3` logs. The revised Rust development output is retained in
`reports/phase4-xsum-quality-rust-dev-v2-s{0,1}/`.
