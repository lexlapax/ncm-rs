# Read-only trace v1 — failed fixed attempt, data insufficient

**Linux remains 72/78; Apple remains pending. Phase 2 is incomplete.**
One attempt ran after the user explicitly confirmed “quiet window now”. The first
traced job exceeded the predeclared 1 GiB capture bound. The sequence stopped, no
job was retried, and the full 900-second telemetry window completed. No GPU policy
was changed. This is not a completed twelve-process diagnostic or acceptance run.

## Frozen launch and observed outcome

Tooling/checkout: `b7f61b5d4a616b80f94b1c0c06d5d1917d7ab14e`.
Accepted checkout: `3307dff82a08f7251b64de0866a01d131528b74f`.
Two metadata-only preparations passed and matched through the entry's comparison;
the entry repeated that preflight before starting collection. Native executable,
source/payload, tool/library, GPU UUID/PCI/name, timezone, command/environment and
worktree guards passed. The explicit sanitized environment differs from v2.
The reviewed preparation digest was
`060b736951e123c42f9b56c10acf488155f054be5fbf109f852ef52ae01f1c3e`.
The evidence archive contains both preparations and the launch-time manifest;
raw Git status/session names remain private and are **not** in the archive.

Exact launch (from repository root; stdout/stderr retained with mode 0600):

```sh
python -S tools/bench/readonly_trace_entry.py launch \
  "$PWD/reports/phase2-readonly-entry-review-v2/manifest.json" \
  060b736951e123c42f9b56c10acf488155f054be5fbf109f852ef52ae01f1c3e \
  "$PWD/reports/phase2-linux-readonly-trace-v1-launch" \
  --quiet-window-confirmed \
  > reports/phase2-linux-readonly-trace-v1-entry.log 2>&1
```

Outer capture: **2026-11-12 00:28:13.637956–00:43:14.443194 UTC**, exit 1.
The sequence stopped after 30.092573 seconds. Fixed job order was not changed:

| Job | PID | Outcome |
| --- | ---: | --- |
| 01 AP, accepted/plain | 3880627 | Exit 0; capture/report complete; 176 process samples |
| 02 CP, candidate/plain | 3880700 | Exit 0; capture/report complete; 177 process samples |
| 03 AT, accepted/traced | 3880940 | `output_limit`, killed with SIGKILL, reaped; no report; 228 process samples |
| 04–12 | — | Not started, not excluded completed runs |

AT ran 11.803864 seconds including cleanup. Capture retained **1,073,741,824 bytes**;
collector observed 1,073,750,016 bytes before stopping (8,192 bytes above the retained
limit). `capture_eof=false`; the last line is partial. The original failed receipt,
process observation and truncated capture are retained. No observation mismatch was
reported: AT's process record is failed because the child/capture is incomplete,
not because an observed mismatch was disregarded.

The independent supervisor recorded complete cleanup, no remaining children or
cleanup errors, and restored its process-local subreaper role. Outer, worker, CPU
sampler and all attempted workloads were reaped; GPU observer shutdown was verified.
A later `/proc` check found none of the eight recorded child PIDs present. This is
specific evidence for this attempt, not a guarantee for supervisor SIGKILL, host
failure, uninterruptible tasks or all CUDA failure paths.

The finalizer correctly rejected the failed outer receipt with
`ValueError: incomplete owned-child receipt`. Neither `completion/` nor
`entry-result.json` was created. **Frozen-input postflight did not execute.** No
postflight, complete twelve-job lifecycle/process verification, output qualification,
trace-to-trial qualification, independent provenance or acceptance pass is claimed.
Two successful plain reports are retained, but do not replace the required groups
of three or supply a selectively accepted timing result.

## Offline inspection and insufficiency

Streaming inspection of the retained 1 GiB capture verified its receipt hash:
`dfb1043085d792fe992cca63bff18ae94a49103bd7388c81340785cff9c24b5f`.
It contains 8,933,480 complete lines plus one partial line. Counts include
4,442,927 DRIVER lines, 184,587 CONCURRENT_KERNEL lines and 643,104 MEMORY2 lines.
These are serialized prefix counts, **not** complete activity or unique allocations.
There are **1,433 `cuEventElapsedTime_v2` queries**, versus the required 2,008 for the
full harness. No complete traced report or candidate trace exists; no six-trace
comparison or qualified ordinal joins can be made. Even where early windows may be
present, this does not qualify the failed process or authorize accepting a prefix.
The run therefore provides **insufficient data for the stated attribution question**.
It establishes a capture-volume limitation, not the cause of B1 timing clusters.

Independent existing telemetry-content validators (`cpu_records`, `gpu_records`)
passed without changing their checks:

- CPU: 901 samples; elapsed 900.002083346 seconds; no optional-data gaps or unavailable
  activity intervals; sample gaps 0.997999–1.002008 seconds.
- GPU: 900 samples; 00:28:13.777–00:43:13.220 UTC; gaps 1.000–1.018 seconds, retained.
- Both streams bracket all **three attempted** receipt windows, including the failed
  one. This is not the twelve-successful-job coverage gate, continuous observation,
  quietness proof or per-call GPU-clock attribution.

The initial offline inspector matched only `cuEventElapsedTime`, incorrectly reporting
zero elapsed queries while its full API-name counts already showed 1,433 `_v2` calls.
Both initial script/result are retained. The corrected inspector imports the existing
boundary parser's `ELAPSED` name set, then re-reads the same capture; it does not rerun
any benchmark or alter a production validator. A generic file-reader attempt also
failed on the 1 GiB text with `Cannot create a string longer than 0x1fffffe8 characters`;
subsequent inspection used bounded reads/streaming rather than loading it all.

## Evidence, reproduction and next boundary

[Evidence index](linux-readonly-trace-v1-evidence.json) binds the compact archive:
36 members, 7,255,877 uncompressed bytes / 354,184 compressed bytes, SHA-256
`ad6812bb1668c93fd1cf906f44c6180a0b2ffde1110148827521b5ae4ba0e479`.
It includes all attempted-job receipts/reports/process records, full telemetry,
collector/supervisor evidence, preparation summaries, launch error and both offline
inspections. No weights, executables, session names or large trace tensors are
published. The **1 GiB truncated CUPTI text remains local**, indexed by exact hash;
public archive reproduction can verify receipts/telemetry/summary consistency but
cannot independently recompute its streamed line/API counts without that retained
capture. This limitation is not represented as full public raw-trace reproduction.

After extracting into a **separate clean scratch checkout**, never over existing
attempt evidence, telemetry contents can be rechecked from the archived manifest:

```python
import json, sys
from pathlib import Path
sys.path.insert(0, 'tools/bench')
from validate_readonly_telemetry import cpu_records, gpu_records
root = Path('reports/phase2-linux-readonly-trace-v1')
manifest = json.loads(Path(str(root) + '-launch/preparation.json').read_bytes())
source = next(x['sha256'] for x in manifest['tool_identities']
              if x['path'].endswith('/tools/bench/sample_linux_cpu.py'))
cpu = cpu_records((root / 'telemetry/cpu.jsonl').read_bytes(), source)
gpu = gpu_records((root / 'telemetry/gpu.csv').read_bytes(),
                  manifest['gpu']['uuid'], manifest['timezone']['zone'])
assert len(cpu['samples']) == 901 and len(gpu) == 900
```

The archived `reports/phase2-readonly-failure-inspect-v2.py` requires the retained
capture at its indexed path and a fresh inspection-output path; it writes
exclusively. Inspection versions are offline re-analysis, **not diagnostic retries**.
No quota increase, sampling retry, selective exclusion, diagnostic-v2 rerun or
acceptance substitution is authorized by this failure.

Post-attempt checks (no further hardware workload or policy command):

| Command / check | Result | Retained log SHA-256 |
| --- | --- | --- |
| `CARGO_BUILD_JOBS=4 ./tools/check.sh` | Requested checks pass; 243 reporting tests | `12b26595d813e5647dc351b6895bcbb0d649c815cf3b8f340a8046cc9999d5f4` (`reports/phase2-readonly-failure-check-v1.log`) |
| Archive member hashes, extraction into clean `git archive b7f61b5`, telemetry code above, summary arithmetic | 36 hashes verified; 901 CPU/900 GPU rows validate; no raw-trace count reproduction claim | `e39846b4fe30db6a6dbd22de8da9bf894ec96f72c500ad87fcb412af4ac4d8b5` (`reports/phase2-readonly-failure-repro-v1.log`) |

The already selected [clock fallback](linux-clock-diagnostic-v3-proposal.md) now has
documented read-only insufficiency. It is **not ready to run**: default policy and
absence of competing controllers must be confirmed, noninteractive set/reset
permissions and enforcement must be established without credential/global changes,
and clock-restoration supervision must be implemented/tested with fake commands.
A fresh quiet window is required afterward. No lock/reset command or permission
mutation test ran here. Separate unchanged-criteria Linux 78/78 acceptance remains
required, followed by actual Apple work and the final requirement-level audit.
