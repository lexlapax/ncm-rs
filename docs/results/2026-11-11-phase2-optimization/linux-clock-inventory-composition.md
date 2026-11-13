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

## Upstream producer binding

The composer now requires an upstream producer record and explicit tree path.
`upstream_group` checks the pinned commit, scope flags, zero untracked/ignored
counts, source path/mode/blob-shape and count/byte consistency, and includes the
Git executable, checkout config and index identities as well as every source.
It is a pure check of supplied declarations: it does not authenticate the producer,
recompute Git blobs or establish current bytes. The future fresh entry must call
the producer and recheck the resulting inventory. Two tests cover the complete
source/Git group and ten malformed or overclaimed variants, without file IO.

The v4 actual snapshot yields 42 upstream-group identities (39 sources and three
Git identities), including four empty sources. The earlier retained-input
compatibility exercise now admits 269 files / 4,019,707,424 bytes and 55 aliases;
it still uses the older 96-file tool inventory and is neither a fresh preparation
nor complete inventory. No workload or GPU operation occurred.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_preparation_inventory test_prepare_clock_upstream -v
```

Nine focused tests pass. `umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh`
passes all default checks, including 408 reporting tests; local log:
`/tmp/ncm-phase2-upstream-inventory-check.log`.

## Precollection file recheck

`verify_clock_preparation_files.verify` reuses declaration admission, resolves all
canonical paths and the exact supplied aliases, inventories every declared file's
regular type/device/inode/size before hashing any file, checks all SHA-256 values,
then rechecks aliases and visible metadata. Its output explicitly denies complete
inventory, producer authentication, atomic snapshot and launch readiness. It is
not an entry and is not yet wired into a real collection. It performs no subprocess,
GPU or policy action; trusted ancestors and races between sampled checks remain.

Five temporary-file tests cover empty files, changed size, same-size changed bytes,
same-byte replacement, changed aliases during hashing and impossible declarations
before path access. The first test run failed on an incorrect expected error-text
substring; the fixture was corrected and replacement was made deterministic by
keeping the old inode alive. The focused combined command passes 25 tests:

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_verify_clock_preparation_files test_clock_preparation_inventory test_verify_clock_files -v
```

The future frozen entry must call this on a freshly assembled and authenticated
inventory and still enforce the final postflight after all collection validation.

The first default check ran 413 reporting tests and failed two previously
intermittent CPU-only observer integrations: the clock default-budget handoff
and read-only lifecycle. The first retains job 10's `live child command mismatch`;
fake reset and cleanup succeeded. The second records an observation mismatch but
does not retain a more specific reason in the sequence receipt. Both tests passed
one direct focused rerun; this does not establish the cause or repair it. Guards,
timings and test expectations were not relaxed. Logs are local at
`/tmp/ncm-phase2-precollection-files-check.log` and
`/tmp/ncm-phase2-observer-focused-rerun.log`. No GPU or real policy action occurred.
One unchanged full rerun, `umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh`, then
passed all default checks including 413 reporting tests; its local log is
`/tmp/ncm-phase2-precollection-files-recheck.log`. This does not explain the failure.
