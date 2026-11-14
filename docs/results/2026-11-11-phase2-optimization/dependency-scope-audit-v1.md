# Dependency and pin scope audit

At `87992c1`, the checksum-pinned `tokenizers` 0.22.2 crate archive contains
99 regular files. The vendored tree has the same file inventory; 96 files match
byte-for-byte and exactly three differ: the BPE model builder, BERT normalizer,
and byte-level implementation. Each changed file matches its previously
reviewed integration commit (`582e8fe`, `4ab4d3a`, or `211df63`, respectively).
The archive SHA-256 matches the frozen Cargo registry checksum, and the upstream
license file is unchanged.

Comparing `Cargo.lock` to accepted baseline `3307dff` finds the same 218
unique package name/version pairs. The only package-record difference is removal
of the registry source and checksum for `tokenizers` 0.22.2, consistent with
the user-approved local vendoring patch; its version and dependencies remain the
same. `upstream.toml`, both tokenizer source/manifest lock files, and tracked
fixture files are unchanged from the baseline. This check does not evaluate
untracked files or establish general semantic/security correctness.

[Reproduction script and JSON report](raw/dependency-scope-audit-v1.tar.xz)
have SHA-256 `32e31a4edf585d5b54d966bf417cd29f855ba4ace299a19b76177405e075b051`.
Reproduction requires git history and the checksum-pinned `.crate` archive:

```sh
python3 ncm-phase2-scope-audit-v1.py REPOSITORY TOKENIZERS_0_22_2_CRATE NEW_REPORT
```

The script rejects tracked worktree changes and unexpected vendored-source, package,
selected pin or tracked-fixture differences. This static audit does not close the current Apple
76/78 performance control or Phase 2.
