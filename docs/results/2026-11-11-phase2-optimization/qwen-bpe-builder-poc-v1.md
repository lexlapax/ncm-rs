# Qwen BPE builder allocation — isolated proof of concept

**Review requested; production dependency unchanged.** The pinned
`tokenizers` 0.22.2 BPE builder currently formats the two pieces of each merge
into a new string. In an isolated temporary workspace and separate Cargo target,
the proof of concept changed that operation to reserve the exact combined byte
length and append the same two pieces. The new token is still looked up in the
same vocabulary, and rank, prefix handling, errors and runtime BPE behavior
remain with the pinned implementation. This experiment did not change the
repository, dependency pin, tokenizer assets or goldens.

Three fresh matched Qwen benchmark processes per binary used CPU 2, five
warmups and 30 trials. Previous checked-load process medians were 65.908,
67.854 and 66.101 ms; proof-of-concept medians were 63.420, 63.916 and 62.889
ms. The median of process medians was **66.101 to 63.420 ms** (ratio 0.95943).
Host HWM overlapped at roughly 76 MB. One allocation capture per binary reported
**60,376 to 128 reallocations** and **55,700,878 to 54,630,078 requested Rust
allocation bytes** per checked load; allocation requests remained 758,101.
These are Rust allocator requests, not native allocation or physical memory.
All three encode/decode scope output hashes matched. The real-asset probe
passed all four profiles and rejection cases, and 18 tokenizer tests passed.
This small diagnostic is not a Linux acceptance result or Apple qualification.

The exact [patch and raw results](raw/qwen-bpe-builder-poc-v1.tar.xz) archive
has SHA-256 `93216be589999a1c9dec564537e987c53be6ee5b38c43ac76784fc1cd891a431`.
The patch itself is available for review at
`/tmp/ncm-qwen-bpe-poc-results-v1/tokenizers-bpe-builder.patch` (SHA-256
`eb630e4adc7e0f78f11bdc13228b75ed1dee8b3f0100ee8bc9fd8e2d8316fcf5`).
The existing registry lock pins version 0.22.2 and checksum
`b238e22d44a15349529690fb07bd645cf58149a1b1e44d6cb5bd1641ff1a6223`.
The temporary Cargo patch changes that package's lockfile source from registry
to local path while keeping the version; the exact lockfile difference is also
in the archive. The upstream package declares Apache-2.0 and includes its license.

An offline provenance check hashed the cached `.crate` and confirmed the exact
lockfile checksum above. All 99 regular files in that archive match the local
registry source byte-for-byte; the registry adds only `.cargo-ok`. Comparing
that 100-file registry directory to the proof-of-concept copy found no missing
or extra files and exactly one changed file, `src/models/bpe/model.rs`. The
unchanged `LICENSE` SHA-256 in both directories is
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
This verifies the candidate copy, not a future vendored tree or a final gate.

The proposed production path is to vendor that exact package with its license,
apply only the reviewed patch, and explicitly update Cargo's source binding and
lockfile. It changes dependency provenance even though the version is unchanged,
so the approved plan requires user review before implementation. If approved,
fresh Linux correctness, resource and performance gates remain required before
Apple work. The current combined candidate still has a failing 77/78 Linux audit.
