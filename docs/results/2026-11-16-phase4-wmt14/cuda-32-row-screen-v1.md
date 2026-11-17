# WMT14 CUDA 32-row development screen

The opt-in experimental CFG-zero output API returns device-resident codes,
decoder logits and greedy IDs. On the two-row top-1 fixture, untraced and
traced codes, logits and IDs agree exactly, and the trace is byte-identical to
the earlier qualified development capture. The normal WMT14 path remains
unsupported.

The following commands use the first 32 independently verified WMT14 source
prefixes, with identical contiguous batches of four. The upstream run uses the
pinned released generation settings and unchanged sampler, with TF32 disabled
and math SDPA. The native run checks the artifact and prefix hashes and uses
the experimental API. Neither result is an official quality or performance
gate. Desktop GPU processes were present.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/wmt14_cuda_screen.py --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json --output reports/phase4-wmt14-python-cuda-screen-v1.json
setpriv --inh-caps=-all env RUSTC_WRAPPER= CUDA_COMPUTE_CAP=86 cargo build --release -p ncm-backend-cuda --features wmt14-experimental --example wmt14_screen
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 target/release/examples/wmt14_screen artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-tokenizer-reference-v1.json reports/phase4-wmt14-rust-cuda-screen-v1.json
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_wmt14_cuda_screen.py reports/phase4-wmt14-python-cuda-screen-v1.json reports/phase4-wmt14-rust-cuda-screen-v1.json reports/phase4-wmt14-cuda-screen-comparison-v1.json
```

Two independent upstream runs produced identical output SHA-256
`f86979d47a76b789889c835fed5a7c3425768cf97ffc1d7ab19ab4daa998a852`.
The Rust output SHA-256 is `d14be2eadd209502a94c0442e879709edf40c869d700721615aa855e176b47ea`;
the comparator SHA-256 is `80eada1c2646d1561376e510524e757e4a57a04775b72a18bb177b171beb91e6`.
All 10,016 codes and 2,048 raw IDs match. The Rust executable SHA-256 is
`966188d29faaafced57153f1b6dc29ce2991371a8112a272e72f7a620c4cb333`;
the Python capture script SHA-256 is
`e892525cbe9ba1717796154037d717a6a23357b866466b901268c59ffb050f70`.
The Rust screen verifies inputs but does not perform a source postflight.
The comparator's shape/range and row-attribution regression, feature model
tests and feature-specific Clippy pass. Full `tools/check.sh --cuda` passed,
including 18 real-device cases (log SHA-256
`0362b8b286e12798b6f285068a47e36bcd014d8800110a84f460f72224c0169c`).
Full-corpus BLEU/chrF, uncertainty,
performance and Metal remain pending.
