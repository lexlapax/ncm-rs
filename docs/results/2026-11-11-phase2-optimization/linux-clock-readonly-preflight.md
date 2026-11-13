# Read-only clock preparation preflight

`clock_preflight.run` accepts two explicitly hash-bound private reviewed records:
a file-only snapshot and read-only GPU metadata. It requires a fresh single-threaded
Linux `-S` interpreter and its own reviewed checkout/Python image. It generates a
new file snapshot and GPU metadata, compares the stable fields and utility identity,
then repeats the worktree guard and rehashes all declared files/aliases **after**
the GPU queries. The new GPU record is exclusively written and re-read by hash;
its digest and the current snapshot digest are included in the private preflight
result. Output and collection paths must be fresh and disjoint.

This is an import-only **read-only preflight**, not a launch entry. It performs
bounded Git/loader/NVIDIA identity and supported-clock probes but never runs a
benchmark, starts telemetry, changes clocks or checks privilege by mutation. It
leaves producer authentication, role ancestry, permissions, hardware policy,
restoration, availability and readiness false. The caller-supplied reviewed hashes
and the future live entry must still be authenticated. File/path checks are sampled,
not atomic or proof of mapped bytes. A successful comparison would not close any
Linux acceptance cell. The user has not supplied a fresh diagnostic quiet window.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_preflight test_compare_clock_snapshots test_compare_clock_gpu -v
```

Ten updated focused tests pass, including bad reviewed digest, each producer/
comparison/late-check failure, reviewed-evidence output isolation and worktree
drift; the tests stub producers and do not query the GPU. An earlier 19-test
focused run also covered snapshot and file-check regressions. Private current-GPU
record publication and reviewed-evidence path isolation were added afterward and
passed the updated rerun. The actual assembled preflight below used a clean
committed snapshot; later source states require fresh preparation.

`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed the default checks,
including 436 reporting tests, before the two refinements described above; their
updated focused tests passed afterward. Local log:
`/tmp/ncm-phase2-clock-preflight-check.log`.

## First actual read-only preflight

At clean pushed commit `6f118be49bef991ceb1808613323b05b66f92750`, a new
file-only snapshot v2 rechecked 329 declared files. The actual preflight then
generated another file snapshot and read-only GPU metadata, matched the stable
file and GPU fields and the NVIDIA utility identity, repeated the worktree
guard, and rechecked **329 files / 4,020,276,376 bytes and 56 aliases** after the
GPU queries. The collection directory remained absent. Permission, hardware
policy, restoration, availability, producer authentication and launch readiness
remain false. No lock, reset, telemetry, Torch import or benchmark ran. This
evidence is bound to that clean commit and must be refreshed for later HEADs.

```sh
umask 077; PYTHONPATH=tools/bench python3 -S -c 'import json,sys; from pathlib import Path; from prepare_clock_snapshot import prepare; root=Path.cwd(); result=prepare(root,root/"reports/phase2-linux-clock-diagnostic-v3",root/"reports/phase2-clock-snapshot-v2",candidate_head="6f118be49bef991ceb1808613323b05b66f92750",home=Path.home().resolve(strict=True),python=Path(sys.executable).resolve(strict=True),git=Path("/usr/bin/git").resolve(strict=True),taskset=Path("/usr/bin/taskset").resolve(strict=True),nvidia_smi=Path("/usr/bin/nvidia-smi").resolve(strict=True)); print(json.dumps(result))' > /tmp/ncm-phase2-clock-snapshot-v2.log 2>&1
umask 077; env -u PYTHONPATH -u PYTHONHOME -u PYTHONTZPATH python3 -S -c 'import json,sys; from pathlib import Path; root=Path.cwd(); sys.path.insert(0,str(root/"tools/bench")); from clock_preflight import run; r=run(root/"reports/phase2-clock-snapshot-v2/snapshot.json","f48c6e94e204797c7e2cbdce8087011b8f959b7b221ffb4741e141980e04f674",root/"reports/phase2-clock-gpu-preparation-v1/result.json","bc96c5020034f59066fefa3030f0b7636b0ab608532fe073e0eb1d351b38a011",root/"reports/phase2-clock-preflight-v1"); print(json.dumps({k:r[k] for k in ("collection_started","permissions_verified","restoration_verified","availability_confirmed","launch_ready")}))' > /tmp/ncm-phase2-clock-preflight-v1.log 2>&1
```

The second command ran outside the sandbox for read-only NVIDIA driver access.

The private ignored reviewed snapshot `reports/phase2-clock-snapshot-v2/snapshot.json`
has SHA-256 `f48c6e94e204797c7e2cbdce8087011b8f959b7b221ffb4741e141980e04f674`;
the resulting `reports/phase2-clock-preflight-v1/preflight.json` has SHA-256
`3942c556b7d89682310a86901dd7d1875e0bb67bf44e808629c5f94adcdab4e1`, and its
`gpu/result.json` has SHA-256
`07652ac40f70fcee6acf7638ad3052d862477e7d8447fc6f09299f89d8381120`.
The local snapshot and preflight command logs have SHA-256 values
`b316ecb1ab57b6b31ef8983c035fdc8759af7b2fc19f8bd41f1451dac11ded4d` and
`b485bf61352d1e849c6f7d06c11594a528832efcd18d8f22d3cf7c777515bdac`.
