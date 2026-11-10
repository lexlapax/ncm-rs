# Phase 2 shared loading remediation — Linux revalidation

**Shared candidates pass the extended Linux gate; Apple candidate validation is
next.** Do not infer Apple performance or close Phase 2 from these Linux results.

## Measured causes and reviewed changes

Apple was rechecked after Linux acceptance `7aee602`. Its main checkout was clean;
old release binaries were saved, a clean `5dc9626` baseline worktree was created,
and main was fast-forwarded. The unrelated `.reference/phase2-mlx-candidate`
worktree was inspected and left untouched. No global tools, allocator, clocks,
power settings or applications were changed.

Fresh unprofiled old Apple codec load is **944.17 ms**, with final host HWM
**4,808,605,696 bytes**, versus qualified Python **270.58 ms / 309,100,544 bytes**.
The fresh measurements reproduce the deficits rather than assuming historical
workstation results still apply.

Three independently motivated mechanisms:

1. `ncm_io::bounded_read` previously grew an initially empty `Vec` geometrically.
   Apple's `malloc_history` attributes the largest live allocation stack
   (201,364,480 bytes at the sampled instant) to `bounded_read → RawVec::finish_grow
   → realloc`; `vmmap` shows large freed malloc regions remaining resident.
   This is not evidence that MLX leaks. Reserve the checked file length plus one
   byte for EOF/growth detection, retaining bounded reads, symlink/path checks,
   hashes, every payload and tensor, and complete preflight. Tests cover empty,
   tiny, large, exact-bound and over-bound files and allocation capacity.
2. Rust Tokenizers 0.22.2 `ModelWrapper` deserialization builds an intermediate
   generic JSON tree before deserializing its model. The authenticated canonical
   profiles already fix BPE versus WordPiece. Deserialize that concrete model
   with the **same** `TokenizerImpl` visitor, then move the complete object into
   the same runtime `Tokenizer` wrapper. Normalization, pre/postprocessing, added
   vocabulary, padding, truncation and model state are preserved—not rebuilt by
   hand. Tests compare complete serialized state and special/Unicode behavior;
   all pinned real-asset goldens and rejection cases pass. No dependency, asset
   format, source pin, or model algorithm changes for this optimization.
3. Apple `sample` attributes **2,187/2,222 main-thread samples** to SHA-256
   compression during host-only loading. Cargo's active features were only
   `default/std`; the Mac reports `hw.optional.arm.FEAT_SHA256: 1`. Inspection of
   the pinned **sha2 0.10.9** shows AArch64 hardware dispatch is gated by `asm`.
   Enable that feature **only on macOS/AArch64**, retaining runtime feature
   detection and the same SHA-256 algorithm. This deliberately adds locked
   **sha2-asm 0.6.4**, checksum
   `b845214d6175804686b2bd482bcffe96651bb2d1200742b712003504a2dac1ab`
   (MIT; build script compiles bundled assembly, no download or global changes).
   sha2 itself remains 0.10.9 (MIT/Apache-2.0). No project-local unsafe code or
   new cryptographic implementation. Known-answer tests include empty, `abc`,
   and the standard million-`a` multi-block vector; pinned artifact digests remain
   authoritative. Apple hardware behavior still requires its own validation.

Research inspected on 2026-11-10: installed Tokenizers 0.22.2
`models/mod.rs`, `tokenizer/serialization.rs`, and moving `From<TokenizerImpl>`;
[sha2 0.10.9 source](https://docs.rs/crate/sha2/0.10.9/source/src/sha256.rs) and its
AArch64 implementation, and sha2-asm 0.6.4 build script;
[Apple malloc diagnostics](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/ManagingMemory/Articles/MallocDebug.html);
[MLX memory definitions](https://ml-explore.github.io/mlx/build/html/python/memory_management.html).
MLX active/peak/cache do not measure process RSS or malloc's freed regions.
No upstream source was copied; technical provenance and dependency licenses remain.

Apple diagnostics used process-local `MallocStackLogging=1` and
`MallocStackLoggingNoCompact=1`, `vmmap -summary PID`, `malloc_history PID -allBySize`,
and a separate `sample PID 3 1 -file ...` capture. These are not acceptance timings.
`raw/apple-diagnosis.tar.gz` preserves both profiler outputs and unprofiled probes.

## Linux correctness and metrics

Changes were validated on Linux before Apple candidate testing. Full
`./tools/check.sh --cuda` passes (`raw/cuda-check.log`), including new allocation,
SHA known-answer and tokenizer-state regressions. All **642 trained traces and
2,500 decisions** pass unchanged, with the same two near ties and exact trace
bytes. All four real tokenizer profiles pass goldens/rejections in a cleared
environment without Python (`raw/tokenizer-parity.json`).

Fresh old/candidate/Python triplets use the unchanged workload with the same
process-local CPU-2 affinity for every runtime. Representative warm loads:

| ms | Previous Rust | Candidate Rust | Python |
| --- | ---: | ---: | ---: |
| Codec | 209.26 | 203.00 | 312.05 |
| BERT | 10.32 | 6.76 | 20.01 |
| GPT-2 | 47.34 | 23.83 | 57.93 |
| Qwen | 194.80 | 107.75 | 172.34 |
| TinyStories | 3.88 | 3.85 | 28.90 |

Qwen native peak host memory is 109–112 MiB; codec HWM is 576–579 MiB.
All codec/loader/tokenizer latency and host-memory comparisons pass in
`linux-acceptance.json` (**78 comparisons**). All medians, p95, dispersions,
process medians, throughput, first calls, allocator counters, telemetry and
absolute memory values are retained in `raw/linux-shared-load-v1.tar.gz`.

The initial three-process follow-up's point estimates all passed, but observed
process envelopes for BERT/GPT-2 encode and TinyStories crossed the 1.05 ceiling.
A fixed 30-triplet affinity-controlled follow-up for those three profiles closes
uncertainty. These are **actual changed binaries**, not identity substitution.
The audit now binds follow-up source fingerprints to its matrix and only permits
executable-identity controls for hash-identical binaries. No negative run or
threshold is removed/relaxed. Both initial and successful audit records and every
raw measurement are public.

The Apple-only SHA feature and known-answer tests were added after this Linux
matrix. Both Linux `codec_bench` and `asset_bench` executables remain **byte-exact**
before/after this addition (`raw/linux-binaries.sha256` verified with
`sha256sum --check`); the Linux feature tree still has no `asm`. Cargo.lock gains
only sha2-asm, with no unrelated updates. The matrix's pre-feature source hashes
remain unchanged evidence; no hidden rerun/relabeling. Final full CUDA checks pass.

## Apple first candidate and tokenizer-buffer follow-up

After pulling `74c2ed3`, full `tools/check.sh --mlx` passes. The first unprofiled
Apple codec candidate loads in **180.06 ms**, host HWM **293,634,048 bytes**, versus
old **944.17 ms / 4,808,605,696 bytes** and Python **270.58 ms / 309,100,544 bytes**.
Host-only artifact load/drop remains approximately 182 MB over 36 iterations.
These are diagnostics, not a completed three-process Apple acceptance matrix.

Qwen load improves to **93.79 ms**, but HWM remains **634,519,552 bytes**. Its
allocation capture independently identifies the tokenizer loader's separate
`assets::read → RawVec::finish_grow → realloc` path. Apply the same checked-length
plus EOF reservation there; new boundary tests retain all byte/path/hash checks.
The preceding candidate and its negative Qwen memory result remain archived in
`raw/apple-shared-diagnostics.tar.gz`, with the Metal check log adjacent.

This additional shared fix returned through Linux first: full CUDA checks and
all real tokenizer goldens/rejections pass. A new complete three-way matrix and
fixed 30-triplet BERT/GPT-2/TinyStories controls yield **78 passing comparisons**
in `raw/phase2-linux-token-buffer-acceptance-v3.json`. Initial failed/uncertain
v1/v2 audits remain beside it. The control summarizer now optionally includes
checked-load trials with exactly the same whole-process bootstrap and unchanged
bands; the audit opts into that evidence explicitly through `load_controls`.
Default historical control output remains unchanged. Malformed load-trial counts
are regression-tested. No identity substitution is used for changed binaries.
`raw/phase2-linux-token-buffer-config.json` records every input; unpack the two
`linux-token-buffer-*.tar.gz` archives to their configured report directories.
Lifecycle and trained codec qualification remain from the preceding shared change;
this follow-up changes only the separate tokenizer reader, not codec execution.

## Remaining work

The tokenizer-buffer follow-up is pushed as `9058df8`, but its Mac pull/test
command could not authenticate: the local SSH agent became unavailable, and a
batch-mode retry was denied. Mac was last observed at `74c2ed3`. No Apple result
for the additional tokenizer-buffer change is claimed. Restore authorized SSH
access, recheck competing work/state, and safely pull the reviewed commit.

Then qualify all trained Metal traces, real tokenizer goldens and repeated
load/request stability. Collect the complete old/new/qualified-Python matrix
and lifecycle evidence. Both extended gates are required; Phase 3 stays blocked.
