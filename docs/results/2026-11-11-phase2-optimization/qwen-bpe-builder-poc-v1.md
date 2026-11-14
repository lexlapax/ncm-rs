# Qwen BPE builder allocation — isolated proof of concept

**Historical proof of concept; production integration was subsequently approved.** The pinned
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

An exact proposed workspace is prepared at `/tmp/ncm-tokenizers-review-v1`.
It was populated from the checksum-verified `.crate`, adds the 99-file package
at `vendor/tokenizers-0.22.2`, applies the one-file patch, and uses a relative
`[patch.crates-io]` override. Its lockfile changes only the registry source and
checksum lines shown in the archived lockfile patch. The `REVIEW.md` SHA-256 is
`87fea598bda8c30b3841d3d18e3c7a6aca2d1daf11b55139380e7f9250eaaa72`;
`review-verification.json` SHA-256 is
`40f3a992b7adf29bce53b1da4fdd2e1e698d3abbb23614a0739a2a89182320cf`.
An offline `cargo check --locked -p ncm-tokenizers` passed in this workspace.
The first unrestricted offline `cargo metadata` attempt stopped because the
unneeded `r-efi` package was absent from the cache; `metadata --no-deps` and
the targeted locked check passed. Neither check is a final performance gate.
The production checkout and dependency source remain unchanged pending review.

The proposed production path is to vendor that exact package with its license,
apply only the reviewed patch, and explicitly update Cargo's source binding and
lockfile. It changes dependency provenance even though the version is unchanged,
so the approved plan requires user review before implementation. If approved,
fresh Linux correctness, resource and performance gates remain required before
Apple work. The current combined candidate still has a failing 77/78 Linux audit.

## Approved production integration

The user approved the exact vendored package and source binding. The repository
now contains the same 99 files as the verified 0.22.2 archive, with only
`src/models/bpe/model.rs` changed and the Apache-2.0 license unchanged. The
workspace Cargo override selects that path; the lockfile drops the registry
source and checksum for this package without changing its version. The other
`tokenizers` version remains untouched. A byte comparison against the cached
archive confirmed the 99-file inventory and single changed source file.

The tokenizer package's 19 tests and Clippy pass; a release real-asset probe
passes all four profiles and their rejection cases. A regression test covers
Unicode, continuing prefixes and a missing merged vocabulary entry. The three
asset examples now include the patched vendor source in their existing
`source_sha256` maps, an explicit addition to that inventory; the report schema
is otherwise unchanged. These checks do not establish the performance gate.
The fresh Linux performance comparison, followed by Apple only after Linux
acceptance, remains required. Earlier statements above describe
the historical review state and isolated experiment.

The first `tools/check.sh --cuda` run under umask 022 stopped before CUDA:
three existing clock-finalization tests failed because their temporary evidence
directories were not private. The same success test passed under umask 077; the
full check passes under that umask, including 482 reporting tests, real CUDA
tests and 18 CUDA operator cases. The passing log SHA-256 is
`3850f0328bc3d8d2d363dd99bf6edb6d5df1e2b5c3fa9d899d9234686c42f500`; the
failed log SHA-256 is `8a4955ef6d426254e5721ad06d0a0db23b4476a49031da037561deb7af999ab0`
and both are retained under `/tmp/ncm-vendored-check-cuda*-v1.log`. Staged whitespace review found only
pre-existing whitespace in the unmodified vendored `CHANGELOG.md` and
`Cargo.toml.orig`; it is retained to preserve the exact upstream archive.
