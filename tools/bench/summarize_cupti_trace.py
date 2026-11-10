"""Summarize NVIDIA CUPTI sample text; diagnostic activity sums are NOT wall time."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re

EVENT = re.compile(r'^(DRIVER|RUNTIME|CONCURRENT_KERNEL) \[ .*? \] duration (\d+), "(.*?)"', re.M)
COPY = re.compile(r'^MEMCPY "(.*?)" \[ .*? \] duration (\d+), size (\d+),', re.M)
MEMORY = re.compile(r'^MEMORY2 \[ .*? \] memoryOperationType (\w+), memoryKind (\w+), size (\d+),', re.M)


def summarize(raw):
    text = raw.decode()
    # The sample prints a message when records are dropped; do not silently
    # attribute an incomplete capture. Reject even a zero-count message for review.
    if "dropped" in text.lower():
        raise ValueError("capture reports dropped records; review before attribution")
    groups = collections.defaultdict(list)
    for kind, duration, name in EVENT.findall(text):
        groups[kind, name].append(int(duration))
    if not any(kind == "CONCURRENT_KERNEL" for kind, _ in groups):
        raise ValueError("no kernel events")
    rows = [{"kind": kind, "name": name, "count": len(values), "sum_ns": sum(values),
             "max_ns": max(values), "over_100us": sum(t > 100000 for t in values)}
            for (kind, name), values in sorted(groups.items())]
    copies = collections.defaultdict(lambda: {"count": 0, "bytes": 0, "sum_ns": 0})
    for direction, duration, size in COPY.findall(text):
        row = copies[direction]; row["count"] += 1; row["bytes"] += int(size); row["sum_ns"] += int(duration)
    memory = collections.defaultdict(lambda: {"count": 0, "bytes": 0})
    for operation, kind, size in MEMORY.findall(text):
        row = memory[f"{operation}/{kind}"]; row["count"] += 1; row["bytes"] += int(size)
    return {"schema_version": 1, "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "Diagnostic CUPTI activity counts/sums, not acceptance timings or process peaks; API/kernel spans can overlap.",
            "totals": {kind: {"count": sum(r["count"] for r in rows if r["kind"] == kind),
                              "sum_ns": sum(r["sum_ns"] for r in rows if r["kind"] == kind)}
                       for kind in ("DRIVER", "RUNTIME", "CONCURRENT_KERNEL")},
            "events": rows, "memcpy": dict(copies), "memory_events": dict(memory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    if args.trace.suffix == ".gz": raw = gzip.decompress(raw)
    result = summarize(raw)
    result["summarizer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as file:
        json.dump(result, file, indent=2); file.write("\n")


if __name__ == "__main__":
    main()
