# WMT14 fixed-history CUDA reference

Two clean pinned-upstream captures used the first two independently verified
96-token WMT14 source prefixes. The upstream generator executed with eleven
deterministic forced code levels, no KV cache, CFG zero, TF32 disabled and math
SDPA. The trace contains all eleven generator head logits, all eleven codec
reconstruction levels, prefixes, forced codes, final decoder logits and greedy
IDs. Sampling and task quality are not evaluated. Source, checkpoint and prefix
hashes were checked before and after each run.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/wmt14_teacher_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json --output reports/phase4-wmt14-teacher-cuda-reference-v1
```

An independent run with output suffix `v2` produced the identical 26-tensor
payload SHA-256 `368d60d5960686faec87a00b73c9521c4dce15430e7748c33d0b611b3f6797e2`
and report SHA-256 `95cfc76db86032dfeedc73cbcff41b121644c1e7f767e84e937eb2afc95ffba8`.
The script SHA-256 is `5c5aefa4998844df46858b00fb3a29b1f2cf1ed4e050a45106c55033879672aa`.
The prefix bounds test passes in the pinned reference environment. GPU desktop
processes were present; this capture makes no timing claim. Reports and large
trace files remain ignored under `reports/`. No Rust generator comparison or
WMT14 quality, performance or Metal gate has yet passed.

## Experimental Rust teacher comparison

The non-default `wmt14-experimental` feature now provides a fixed-history
teacher path. It checks the pinned generator tensor inventory before upload,
handles the prefix-only first level, prefix padding and ten untied later heads,
and does not provide free generation. Its CUDA capture used the reference and
artifact manifest hashes embedded in the example:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= CUDA_COMPUTE_CAP=86 cargo build --release -p ncm-backend-cuda --features wmt14-experimental --example wmt14_teacher_trace
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 target/release/examples/wmt14_teacher_trace artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-teacher-cuda-reference-v1 reports/phase4-wmt14-teacher-rust-cuda-v2
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_wmt14_teacher_trace.py reports/phase4-wmt14-teacher-cuda-reference-v1/trace.safetensors reports/phase4-wmt14-teacher-rust-cuda-v2/traces.safetensors reports/phase4-wmt14-teacher-cuda-comparison-v2.json
```

All 26 tensors pass the existing comparison bands; all 626 head argmax choices,
recorded forced codes, prefix IDs and greedy decoder IDs match exactly. The largest absolute
difference is 0.000041961669921875 in `generator.final_logits`. The native trace
SHA-256 is `cff57cecc1ba1e486fdad62ed1050754266cdfff126e94111063d3dd9db718ee`,
comparison SHA-256 `c51d810e979a1e27fed5b3b37280f131c6c0766cd805740522e447c8ae1bc66a`,
and executable SHA-256 `79791650cbf5b171be8c72cfcd9e16d02ca8b3f4d5c8e87d85176e5f4c5b4421`.
The example binds artifact and reference identities but does not perform a
source postflight. The initial capture had the same trace bytes before three
Clippy style corrections and a prefix-position regression test. The corrected
feature-enabled model tests, comparator test and Clippy pass. Full
`tools/check.sh --cuda` passed 18 real-device operator cases (log SHA-256
`3270fdbcea8ea5089d48fcd08ed084c38a36392d1113c0e27b22669e30e4c033`);
its default checks began before the feature-only corrections, while its CUDA
build and device cases ran afterward. This is a two-prefix
development fixture, not a sampling, translation-quality, performance or Metal
gate; the normal WMT14 inference path remains unsupported.

## Metal comparison

At `8bf8e3e`, the shared two-prefix example reproduced the CUDA trace
byte-for-byte. The Metal build ran the same fixed inputs and teacher
histories against the same pinned Python CUDA trace. The unchanged
comparator passed all 26 tensors, including the 626 head argmax choices,
forced codes, prefix IDs and greedy decoder IDs. The largest absolute
difference was `0.000396728515625` in `generator.level_07.logits`.
The Metal example passed release build and Clippy. This check covers
fixed-history numerical parity. Free-generation and performance have
separate comparisons.

| Metal evidence | SHA-256 |
| --- | --- |
| Trace | `ffb88a7a15d908b4aa6bb29e5f07e6e0723c694963a959f499155a0310026556` |
| Comparison | `155d4c19d6364e8d300d665d0735e0f96a1f019e59c2d15894e6326542816c70` |

The trace, comparison and source identity receipt are retained in the
ignored `reports/phase4-wmt14-guided-head-v1.vo6Wpv/corpus/teacher/`.
