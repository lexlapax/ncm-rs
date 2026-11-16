# XSum CUDA tie-order screen

The pinned PyTorch CUDA build (`2.15.0+cu132`, build commit
`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`) sorted 4,096 synthetic 4096-way
rows on an RTX 3060 Ti. The rows had 2, 3, 4, 8, 16, 64, 512 or 4096 equal
maxima, 512 rows per count; other entries were negative infinity. For every
row, the first sorted maximum had the lowest tied index. This reproduces the
relevant sorted, top-k-1-masked score shape, not complete model generation.

That observed CUDA ordering is not a portable guarantee: the pinned upstream
uses default `torch.sort` without requesting stable sorting, and the
[PyTorch 2.15 sort documentation](https://docs.pytorch.org/docs/2.15/generated/torch.sort.html)
only guarantees preservation of equal-value order with stable sorting enabled.
The screen cannot prove behavior for all values, layouts, batch shapes,
devices or future runtime builds. The existing XSum top-1 selector therefore
continues to reject exact maximum ties explicitly; no tie policy was changed.

Run `OMP_NUM_THREADS=1 .reference/venv/bin/python
tools/reference/xsum_tie_order_probe.py` on the CUDA device. The deterministic
probe has SHA-256
`d3b802f5a3e8b0bdc15f0959720020b01e98a94b7b8b8759aa92f8e7bb8612df`;
the captured JSON in `reports/phase4-xsum-tie-order-screen-v1/result.json`
hashes to `8d4221872e0830f4e3b4d78eaad0753cc9dfba77c5ae87460c37e11689c3e6c8`.
The screen used synthetic data and did not fingerprint linked libraries or
monitor desktop activity; it is not a performance measurement or full gate.
