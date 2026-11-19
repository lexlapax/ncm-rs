# LM1B-medium and LM1B-large CUDA timing

Rust (`cuda_lm1b_efficient_experimental` with the reference rotary tables and device penalties, the
`sample-lm1b` CUDA path) is compared with pinned upstream Python on the RTX 3060 Ti. The method is the
same as the [small repeated timing](small-cuda-capacity-screen-v1.md#repeated-cuda-timing-and-allocator-measurements):
development seed 11, three fresh processes per runtime and batch in alternating order, five warmups and
30 synchronized timed requests each (90 per row), model load excluded. Python and Rust use independent random
streams. Output signatures matched across the three processes in every row. All runs were at commit
`d5f2de3` on 2026-11-19.

**Result: Rust is faster than Python at the median and p95 for every batch both runtimes can run, and it
uses less host and GPU memory.**

## Medium (batches 1, 4, 16)

| Batch | Runtime | Median | p95 | Samples/s | Median peak RSS | Peak reserved GPU memory |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Python | 0.141959 s | 0.142576 s | 7.04 | 3,936,888 KiB | 2,228 MiB |
| 1 | Rust | 0.118780 s | 0.119364 s | 8.42 | 2,430,680 KiB | 1,696 MiB |
| 4 | Python | 0.333684 s | 0.334455 s | 11.99 | 3,938,348 KiB | 2,786 MiB |
| 4 | Rust | 0.303277 s | 0.304094 s | 13.19 | 2,431,640 KiB | 1,888 MiB |
| 16 | Python | 1.099324 s | 1.102031 s | 14.55 | 3,936,172 KiB | 5,576 MiB |
| 16 | Rust | 1.049562 s | 1.051559 s | 15.24 | 2,419,104 KiB | 2,528 MiB |

Rust's median is 16.3%, 9.1% and 4.5% lower at batches 1, 4 and 16.

## Large (batches 1, 4, 8)

The first large sweep (batches 1, 4, 16) stopped when pinned Python ran out of GPU memory at batch 16:
`torch.OutOfMemoryError ... Tried to allocate 478.00 MiB` in upstream `eval_shard.py` `pick_tokens`
(`logits.double()`), with 5.33 GiB held by the process on the 8 GiB card. That run is retained in
`reports/phase4-lm1b-large-cuda-timing-v1/`. Large was rerun one batch per directory at 1, 4 and 8,
the 80-seed campaign batch.

| Batch | Runtime | Median | p95 | Samples/s | Median peak RSS | Peak reserved GPU memory |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Python | 0.287064 s | 0.296723 s | 3.48 | 7,128,684 KiB | 3,910 MiB |
| 1 | Rust | 0.251922 s | 0.264652 s | 3.97 | 4,025,024 KiB | 3,264 MiB |
| 4 | Python | 0.682478 s | 0.684737 s | 5.86 | 7,127,140 KiB | 4,382 MiB |
| 4 | Rust | 0.638896 s | 0.641059 s | 6.26 | 4,027,832 KiB | 3,424 MiB |
| 8 | Python | 1.227692 s | 1.230858 s | 6.52 | 7,127,408 KiB | 5,154 MiB |
| 8 | Rust | 1.168629 s | 1.171923 s | 6.85 | 4,027,572 KiB | 3,680 MiB |

Rust's median is 12.2%, 6.4% and 4.8% lower at batches 1, 4 and 8. Rust at large batch 16 was not
measured, since there is no Python comparison on this card.

## Limits

GPU memory columns use each runtime's own allocator counter (PyTorch peak reserved; Rust CUDA default-pool
peak reserved), so they are comparable in scale, not in exact scope. Desktop processes held about 450 MiB of
GPU memory during the runs. This is a development performance comparison at one seed, not an exclusive-GPU
benchmark; quality is established separately by the [80-seed campaigns](cohort-campaign-v1.md#prospective-cuda-results-medium-and-large).

| Item | SHA-256 |
| --- | --- |
| Medium result | `b86abc29d23ffdf090df7585c76acd5e8fd87ee1ba5be851f63ca3ab466da3a1` |
| Large batch 1 result | `fbb436f4255e826c300ef18ef464bffd93ec0712d42693c2b11b2fdf6a118a21` |
| Large batch 4 result | `3be5edef5475e858933c7f0d8ac9ebbada45ef8323e23d2f94428dd10f9d1c71` |
| Large batch 8 result | `ae8f6550ecf2de191ebe89d9709f76c81a932be6b4ce01a958889240a0ae3ac0` |
| Failed large sweep manifest | `3cafd51848ec1f278d0c53f3b3564eefe86a5b539327c19d3a003add81597f04` |

Raw results are in ignored `reports/phase4-lm1b-medium-cuda-timing-v1/` and
`reports/phase4-lm1b-large-cuda-timing-v1-b{1,4,8}/`.

```sh
py=.reference/venv/bin/python
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
  $py tools/bench/run_lm1b_small_capacity_screen.py --size medium --repeated \
  --reference-rope-diagnostic reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors \
  --rust-cudnn-graph-efficient --output reports/phase4-lm1b-medium-cuda-timing-v1
# Large: add --batch 1, 4 or 8 and use --size large with one output directory per batch.
```
