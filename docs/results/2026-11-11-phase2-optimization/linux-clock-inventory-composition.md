# Clock preparation inventory composition

`clock_preparation_inventory.compose` joins supplied native, Python, standalone
loader and preload producer records with explicitly named tool, upstream, utility
and timezone identity groups. It calls the existing pure library compiler and the
same early declaration admission used by the finalizer. It rejects missing groups,
identity conflicts, missing injection/native image bindings, alias conflicts and
count/byte overflow without opening files or creating processes. It copies inputs.
The resulting scope flags explicitly deny current-file recheck, complete inventory,
producer authentication and launch readiness. The eventual fresh entry must
construct and authenticate each group, independently verify interpreter links,
recheck aliases/files and bind live roles; this component is not that entry.

Compatibility against retained preparation records exposed two pre-existing
integration failures. Four upstream source files are empty; previously the clock
file declaration reused an executable/library-only identity check requiring
positive length. Frozen regular-file declarations now permit zero length and the
final rehash still checks bytes, inode and digest. Libraries, benchmark images,
interpreter, taskset and injection still require positive length. A real temporary
empty file passes final rehash and later mutation fails.

The loader record also contains twelve absolute paths with `..`. Lexically
collapsing those paths would change meaning in the presence of symlink parents.
Only aliases from the supplied library contract accept this spelling; explicit
extra aliases, paths of frozen files, roots and output paths remain strictly
normalized. Postflight resolves each original alias to its declared pinned target.
A regression exercises a symlink parent, successful resolution and changed-target
rejection. The future fresh preparation must also resolve and bind these aliases
before launch. This is not continuous symlink or immutable-mapped-byte proof.

A file-only compatibility run against the retained Python v3, loader v1, preload
v1, read-only preparation v2, and upstream v2 records passed declaration admission:
267 unique files, 4,019,703,432 declared bytes and 55 aliases. Its old tool group
contains 96 files and is not the current tool inventory; the result is neither a
fresh preflight nor evidence that complete current inputs fit. The earlier 303-path
planning count used a newer tool inventory and different declarations; neither is
a complete launch inventory. No GPU query, policy command, benchmark or fresh
hardware evidence was generated.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_preparation_inventory test_verify_clock_files test_clock_library_contract test_finalize_clock_diagnostic -v
python -S /tmp/ncm-clock-inventory-compat.py
```

The focused suite passed 28 tests. The first synthetic run failed because the
fixture accidentally shared its native and loader identity objects; the fixture
was made independent, after which native drift was rejected. The initial retained
record composition failed on the empty-file check; after that repair it exposed
the loader `..` mismatch, then passed after exact-path handling. These failures
were not workload runs or accepted measurements. The compatibility script under
`/tmp` reads the five retained local records and reports aggregate counts only;
it is not a standalone reproducibility artifact or launch entry.

`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed outside the sandbox,
including 403 reporting tests. The local log is
`/tmp/ncm-phase2-inventory-composition-check.log`.
