# XSum quality evaluator preparation

The pinned upstream `scripts/eval_merge.py` computes stemmed ROUGE-1, ROUGE-2 and ROUGE-L, and also reports GPT-2 generative perplexity and mean output length. Its SHA-256 is `6452cb5cc06a2355ab7cd16b9809b7144a557418e9041e4617f3c50aef5655fc`; the upstream `evaluation/gen_ppl.py` SHA-256 is `31cde7d31aadc8cd873f7a853e818270e27e4789af4fb573d1821c8a0072ef5f`. The upstream script names `gpt2` without a revision; this project now explicitly pins the evaluation-only `openai-community/gpt2` model and tokenizer to [revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e). This is a new evaluator pin, not a change to NCM weights or the previously pinned GPT-2-large TinyStories evaluator.

The ignored local files are in `artifacts/upstream/gpt2-evaluator-phase4-v1/`. The 548,105,171-byte `model.safetensors` SHA-256 is `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`; the six small configuration and tokenizer files were checked against the repository's Git blob IDs at that revision. The local `SHA256SUMS` SHA-256 is `e8f29b1554a0b0588477a6c2e91267e707580f64df2fc2a9cdd437cc0a282036`. The download shell reported exit 1 because its cleanup trap expanded a nonexistent `.part` glob after successfully checking the weights; all seven downloaded files were subsequently verified, and no partial file remained. An offline CPU load with Transformers 4.57.3 succeeded, reporting a fast 50,257-token tokenizer and 124,439,808 model parameters. This is not a GPU evaluation.

The evaluation-only scorer environment is `.reference/phase4-xsum-rouge-v1/`, used alongside the existing isolated `.reference/phase4-eval-deps/`. [Google's `rouge-score` 0.1.2 source distribution](https://pypi.org/project/rouge-score/0.1.2/) was downloaded and verified against PyPI SHA-256 `c7d4da2683e68c9abf0135ef915d63a46643666f848e558a1b9f7ead17ff0f04`; only its package Python files were extracted after path and size checks. Its Python-only dependencies were installed from separately verified local wheels without dependency resolution: NLTK 3.9.2 (`1e209d2b3009110635ed9709a67a1a3e33a10f799490fa71cf4bec218c11c88a`), absl-py 2.3.1 (`eeecf07f0c2a93ace0772c92e596ace6d3d3996c042b2128459aaae2a76de11d`), six 1.17.0 (`4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274`), joblib 1.5.2 (`4e1f0bdbb987e6d843c70cf43714cb276623def372df3c22fe5266b2670bc241`) and click 8.2.1 (`61a3265b914e850b85317d0b3109c7f8cd35a670f963866005d6ef1d5175a12b`). Existing environment versions include NumPy 2.5.4, regex 2026.11.9 and tqdm 4.70.1; their installed bytes were not separately fingerprinted here. The upstream requirements did not pin these packages, so the chosen versions are explicit rather than a claim about an original historical environment.

An offline stemmed-ROUGE smoke check returned ROUGE-1 0.875, ROUGE-2 0.714286 and ROUGE-L 0.875 for reference `The quick brown dog jumps on the log.` and hypothesis `The quick brown dog jumps on a log.` No XSum passages have been scored, no quality margin has been frozen, and no ROUGE or perplexity gate is claimed. The held-out protocol still needs reference-variation evidence and a predeclared margin before comparing Rust with Python.

## Decoder and scorer preparation

The experimental scorer's source-package verification passed, and six CPU-only
batch and scorer integrity tests passed. On the 32 IDs from the first Python
batch-32 performance process plus one synthetic EOS/padding example, the
scorer's decoded text matched the pinned upstream `HFTokenizer` using the
locally pinned GPT-2 tokenizer (33 of 33 examples; EOS 50256, padding 50257).
The first attempt could not import `transformers` because the isolated evaluator
dependency directory was absent from `PYTHONPATH`; adding that existing local
directory made the comparison pass. These are development checks, not a quality
score. CUDA generation, restart behavior, ROUGE/genPPL execution and held-out
quality remain unqualified.

The synthetic tests and package check were run with:

```sh
PYTHONPATH=tools/reference .reference/venv/bin/python -m unittest -v test_xsum_quality_generate test_xsum_quality_score
PYTHONPATH=tools/reference .reference/venv/bin/python -c 'import xsum_quality_score as s; s.verify_scorer_files()'
```

The decoder comparison read `reports/phase4-xsum-cuda-performance-b32-v2/python-b32-r0.json`
and used `PYTHONPATH=tools/reference:.reference/upstream-ncm:.reference/phase4-eval-deps`:

```sh
PYTHONPATH=tools/reference:.reference/upstream-ncm:.reference/phase4-eval-deps .reference/venv/bin/python -c 'import json; from transformers import AutoTokenizer; from tokenization.hf_tokenizer import HFTokenizer; from xsum_quality_score import decode_ids,MODEL; a=AutoTokenizer.from_pretrained(MODEL,local_files_only=True); b=HFTokenizer(str(MODEL),pad_token="<|pad|>",eos_token="<|endoftext|>",local_files_only=True); d=json.load(open("reports/phase4-xsum-cuda-performance-b32-v2/python-b32-r0.json")); samples=d["ids"]+[[100,50257,200,50256,300]]; expected=lambda ids:b.decode([t for t in (ids[:ids.index(50256)] if 50256 in ids else ids) if t not in (50256,50257)],skip_special_tokens=True); print("upstream decoder matches:",all(decode_ids(a,ids)==expected(ids) for ids in samples),"rows:",len(samples),"eos/pad:",b.eos_token_id,b.pad_token_id)'
```
