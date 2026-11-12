# Clock preparation inventory admission blocker

**Status: approval requested, no limit changed. Linux 72/78; Apple pending.**
Before assembling fresh complete preparation, a path-count audit at `e4ba006`
found that the planned frozen inventory cannot fit the clock postflight's existing
**256-file** admission limit. A subset already contains **261** unique paths; the
current planned inventory contains **303**. This is not a failed hardware run.

## Evidence and scope

[Count evidence](linux-clock-preparation-admission.json) records input-record hashes,
group counts, independently recomputed cumulative set unions and private audit-log
hashes. The audit joins the current top-level Python tool listing with retained
Python, library/preload, upstream and native-preparation declaration paths. It does
**not** rehash target libraries/payloads, prove fresh identities or establish a complete
future import/role inventory. No GPU query, Torch import, benchmark, policy command
or runtime environment change occurred.

| Group | Declared paths | Newly added | Cumulative unique files |
| --- | ---: | ---: | ---: |
| All `tools/bench/*.py` and `tools/reference/*.py` | 136 | 136 | 136 |
| Python metadata, startup and interpreter | 43 | 43 | 179 |
| Known libraries, including optional candidates | 40 | 40 | 219 |
| Three workload images and taskset | 4 | 3 | 222 |
| Tracked upstream files | 39 | 39 | **261** |
| Reviewed Torch preload sources | 3 | 2 | 263 |
| Native source declarations and workload fixtures | 26 | 26 | 289 |
| Codec payloads and manifest | 7 | 7 | 296 |
| Reference/profiling declarations, archive and injection identity | 4 | 4 | 300 |
| Utility and timezone identities | 4 | 3 | **303** |

Overlap is deduplicated by declared canonical pathname, not ignored to reduce the
count. The 261 figure is a lower bound **for this planned inventory**, not a claim
that every possible redesign needs that many files. The broad tool inventory follows
`prepare_readonly_launch.tool_inventory`: freeze both Python tool directories rather
than pretend a short hand-written list proves the validator dependency closure.
Current source declarations retain upstream assets and known optional libraries;
none were silently discarded to fit. Future entry/preparation sources and additional
mapped libraries can add paths, so 303 is not a proof of final inventory completeness.
No current total-byte or largest-file qualification is implied by this count audit.

## Proposed bounded change — awaiting user approval

Raise **only** `verify_clock_files.FILE_COUNT_LIMIT` from **256 to 512** and have fresh
clock preparation enforce the same bound before launch. Add tests at the actual new
boundary and with the larger composed inventory. Keep unchanged:

- **8 GiB total / 2 GiB per-file** postflight byte limits and 512 alias bindings;
- the separate read-only postflight limit and the 128-file / 1 GiB evidence reader;
- eighteen processes, five warmups/thirty trials, no diagnostic retry;
- 1 MiB workload/control captures, 16 MiB reports and all collection/disk checks;
- original 600/590/10-second restoration/work/reserve deadlines and 660/60 outer capture;
- 901-sample/900-second telemetry, all numerical tolerances and acceptance criteria.

This is a postflight file-count admission proposal, **not** permission to increase
failed read-only v1's trace quota, retry either diagnostic or change GPU policy.
512 would leave headroom for the currently planned inventory; actual fresh counts,
bytes, aliases and added paths still must pass. Do not evade the current bound with
multiple batches, dropped tests/assets/optional candidates or a green partial manifest.
If the change is declined, a separately reviewed inventory/guard redesign is needed.

## Checks and next input

```sh
python -S reports/phase2-clock-preparation-inventory-v1.py
PYTHONPATH=tools/bench python -m unittest test_verify_clock_files -v
```

All **9 existing file-postflight tests pass** with the limit still 256. Cumulative
set counts were independently recomputed from the retained per-path group membership.
Default checks were not rerun for this docs/count checkpoint; the prior checkpoint
passed 391 reporting tests. No green test here makes the 303-file plan admissible.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-preparation-inventory-v1.py` | `40f05f2dbed585ffe1fcb627d8052a9211f78ec044b5a484dcfcc416a88d05de` |
| `phase2-clock-preparation-inventory-v1.json` | `203f9b56400a4802f06de53fa755804c9620f780917450bf5202472da223cfba` |
| `phase2-clock-preparation-inventory-v1.log` | `cc3aa34a1209665e957f9e6963fea0b54f5c81b3ef9f0e323346336e459ecd83` |
| `phase2-clock-preparation-admission-tests-v1.log` | `804559bbb2cde037dc9d2355606004f9051e582bf36bdada658d39eb3bf1d6b6` |

Requested input: approval for the **clock-only 512-file postflight count**, and the
still-unanswered independent question whether GPU clocks are currently at defaults,
with no custom lock or competing controller (“unsure” is acceptable). Continuing the
goal is not either answer. No quiet window is requested yet: full preparation, frozen
restoration-aware entry, permissions/enforcement/restoration and readiness remain
pending. Phase 2 completion and both runtime acceptance gates remain unachieved.
