"""Qualify reported CUDA codec outputs against frozen v2 Python reports.

Diagnostic only: no acceptance ratios, process-freshness proof, new oracle run,
operator/trace parity or unreported floating-output checks are implied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

from bind_cupti_trials import LAYOUT_SOURCES
from phase2_summary import codec, require

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/results/2026-11-11-phase2-optimization/linux-diagnostic-v2-evidence.json'
NATIVE_SOURCES = set(LAYOUT_SOURCES) | {
    'Cargo.lock', 'crates/ncm-backend-cuda/src/runtime.rs',
    'crates/ncm-backend-cuda/src/codec.rs', 'crates/ncm-backend-cuda/src/ops.rs',
    'crates/ncm-backend-cuda/src/resample.rs', 'crates/ncm-core/src/config.rs',
    'crates/ncm-model/src/lib.rs', 'crates/ncm-model/src/blocks.rs',
    'crates/ncm-model/src/codec.rs', 'crates/ncm-io/src/lib.rs',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen_references():
    """Use exactly one copy of each of the three published Python reports."""
    evidence_raw = EVIDENCE.read_bytes()
    archive = json.loads(evidence_raw)['archive']
    path = ROOT / archive['path']
    require(path.stat().st_size == archive['bytes'] and digest(path.read_bytes()) == archive['sha256'],
            'frozen reference archive mismatch')
    members = {m['path']: m for m in archive['members']}
    reports, identities = [], []
    with tarfile.open(path, 'r:xz') as capture:
        for repeat in (1, 2, 3):
            name = f'phase2-linux-diagnostic-v2/candidate/codec-python-{repeat}.json'
            matches = [m for m in capture.getmembers() if m.name == name]
            require(len(matches) == 1 and matches[0].isfile(), 'invalid reference archive member')
            raw = capture.extractfile(matches[0]).read()
            expected = members[name]
            require(len(raw) == expected['bytes'] and digest(raw) == expected['sha256'],
                    'frozen reference report mismatch')
            reports.append(json.loads(raw))
            identities.append({'member': name, 'sha256': digest(raw)})
    return reports, {'evidence_sha256': digest(evidence_raw), 'archive_sha256': archive['sha256'],
                     'reports': identities}


def validate_group(runs, runtime, expected_sources):
    require(len(runs) == 3, 'expected three reports per group')
    canonical = [json.dumps(run, sort_keys=True, separators=(',', ':'), allow_nan=False) for run in runs]
    require(len(set(canonical)) == 3, 'duplicate reports are not independent evidence')
    for run in runs:
        require(type(run['schema_version']) is int and run['schema_version'] == 1,
                'unsupported report schema')
        require(run['runtime'] == runtime and run['device'] == 'cuda' and run['precision'] == 'fp32',
                'unsupported diagnostic runtime/precision')
        require(run['warmups'] == 5 and run['trials'] == 30, 'unsupported diagnostic trial protocol')
        require(run['source_sha256'] == expected_sources, 'diagnostic source inventory mismatch')
    # Profiling may affect latency, not the report-output numerical contract.
    if runtime == 'rust-candle':
        require(all(type(r['diagnostic_profiling']) is bool and r['index_dtype'] == 'u32' for r in runs),
                'invalid native profiling/index metadata')
        require(len({r['diagnostic_profiling'] for r in runs}) == 1, 'mixed profiling modes in group')


def qualify(raw_reports, expected_sources):
    require(len(raw_reports) == 3, 'expected three native reports')
    require(isinstance(expected_sources, dict) and set(expected_sources) == NATIVE_SOURCES,
            'incomplete declared native source inventory')
    require(all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) for h in expected_sources.values()),
            'invalid declared source hash')
    require(all(expected_sources[p] == h for p, h in LAYOUT_SOURCES.items()),
            'unreviewed diagnostic harness layout')
    native = [json.loads(raw) for raw in raw_reports]
    validate_group(native, 'rust-candle', expected_sources)
    reference, binding = frozen_references()
    validate_group(reference, 'pytorch-pinned-eager-snapshot-fix', reference[0]['source_sha256'])
    # Reuse all existing report/output checks and the unchanged FP32 tolerances.
    # Explicit runs bypass the acceptance-only file reader, not codec validation.
    validated = codec(ROOT, 'cuda', runs={'rust': native, 'python': reference})
    return {
        'schema_version': 1, 'diagnostic_only': True, 'reported_codec_outputs_qualified': True,
        'performance_acceptance_evaluated': False, 'launch_provenance_verified': False,
        'native_report_sha256': [digest(raw) for raw in raw_reports],
        'declared_native_source_sha256': expected_sources, 'frozen_reference_binding': binding,
        'quality': [{key: row[key] for key in ('batch', 'scope', 'quality', 'output_ids_sha256')}
                    for row in validated['measurements']],
        'scope': 'Existing full codec report validation: all 12 scopes, IDs/retained counts, '
                 'first decode/reconstruct logits, stage/load report structure. No new Python run, '
                 'independent-process proof, operator/642-trace parity, unreported block/snapshot '
                 'float checks, or performance acceptance. Distinct reports alone do not prove freshness.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('native_reports', type=Path, nargs=3)
    parser.add_argument('--expected-sources', type=Path, required=True)
    args = parser.parse_args()
    source_raw = args.expected_sources.read_bytes()
    result = qualify([p.read_bytes() for p in args.native_reports], json.loads(source_raw))
    result['source_declaration_sha256'] = digest(source_raw)
    result['tool_source_sha256'] = {
        name: digest(Path(__file__).with_name(name).read_bytes())
        for name in ('qualify_diagnostic_codec.py', 'phase2_summary.py', 'bind_cupti_trials.py')}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
