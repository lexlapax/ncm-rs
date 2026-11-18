"""Compare WMT14 MLX processes or recheck retained results against frozen outputs."""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path

from run_phase3_mlx_bench import ROOT, run
from run_process_resources import digest_file as digest

sys.path.insert(0, str(ROOT / 'tools/reference'))
from compare_wmt14_corpus_capture import REFERENCE_SHA256

ARTIFACT = ROOT / 'artifacts/canonical/wmt14-inference-v1-final'
PREFIX = ROOT / 'reports/phase4-wmt14-tokenizer-reference-v1.json'
BINARY = ROOT / 'target/release/examples/wmt14_perf'
PYTHON = ROOT / '.reference/mlx-venv/bin/python'
ADAPTER = ROOT / 'tools/reference/wmt14_mlx_reference_bench.py'
CELLS = ((0, 1), (0, 4), (0, 16), (0, 32), (1, 1), (1, 4), (1, 16))
MEMORY_FIELDS = ('active', 'peak', 'cache')
REPETITIONS = 3
WARMUPS = 5
TRIALS = 30
CONTROL_FACTORS = {
    'checkpoint': ('materialize_after', 'checkpoint_screen_level', range(1, 11)),
    'decoder': ('decoder_rows', 'decoder_screen_rows', range(1, 33)),
}


def ordered_runs(cell, with_control=False):
    cfg, batch = cell
    cell_index = CELLS.index(cell)
    for rep in range(1, REPETITIONS + 1):
        if with_control:
            cycle = ('rust_control', 'rust', 'python')
            if cell_index % 2:
                cycle = cycle[::-1]
            offset = (cell_index + rep - 1) % len(cycle)
            runtimes = cycle[offset:] + cycle[:offset]
        else:
            runtimes = ('rust', 'python') if (cell_index + rep) % 2 else ('python', 'rust')
        for runtime in runtimes:
            yield runtime, f'{cfg}-{batch}-{runtime}-{rep}'


def checked_output(report, runtime, cfg, batch, expected):
    start_key = 'start' if runtime == 'rust' else 'case_start'
    times = report['trials_seconds']
    if ((report['cfg'], report['batch'], report['warmups'], report[start_key])
            != (cfg, batch, WARMUPS, 0) or len(times) != TRIALS
            or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times)):
        raise ValueError('unexpected timing scope or samples')
    load = report['load_seconds']
    if type(load) not in (int, float) or not math.isfinite(load) or load < 0:
        raise ValueError('invalid model load time')
    memory = report['mlx_memory']
    if any(type(memory[key]) is not int or memory[key] < 0 for key in MEMORY_FIELDS):
        raise ValueError('invalid MLX allocator counters')
    for key in ('codes', 'raw_ids'):
        values = report[key]
        rows = [row[key] for row in expected]
        wanted = [value for row in rows for value in row] if runtime == 'rust' else rows
        if runtime == 'rust':
            correctly_typed = isinstance(values, list) and all(type(value) is int for value in values)
        else:
            correctly_typed = isinstance(values, list) and all(
                isinstance(row, list) and all(type(value) is int for value in row) for row in values)
        if not correctly_typed or values != wanted:
            raise ValueError(f'{runtime} WMT14 {key} differ from the frozen reference')
    return {
        'trials_seconds': times, 'load_seconds': load,
        'mlx_memory': {key: memory[key] for key in MEMORY_FIELDS},
    }


def checked_allocator_trace(report):
    boundaries = report.get('allocator_boundaries')
    starts = report.get('trial_start_unix_seconds')
    if (type(boundaries) is not list or len(boundaries) != TRIALS + 1
            or type(starts) is not list or len(starts) != TRIALS
            or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in starts)
            or any(a >= b for a, b in zip(starts, starts[1:]))):
        raise ValueError('incomplete WMT14 allocator trace or invalid trial timestamps')
    for counters in boundaries:
        if (type(counters) is not dict or counters.keys() != set(MEMORY_FIELDS)
                or any(type(n) is not int or n < 0 for n in counters.values())
                or counters['active'] > counters['peak']):
            raise ValueError('invalid WMT14 allocator trace counters')
    if (any(a['peak'] > b['peak'] for a, b in zip(boundaries, boundaries[1:]))
            or boundaries[-1] != report['mlx_memory']):
        raise ValueError('WMT14 allocator trace disagrees with process high water')
    return {'allocator_boundaries': boundaries, 'trial_start_unix_seconds': starts}


def checked_rss(directory, stem, replay, system):
    if replay and not (directory / f'{stem}.meta.json').exists():
        data = (directory / f'{stem}.time').read_text()
        values = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', data, re.MULTILINE)
        if len(values) != 1 or int(values[0]) <= 0:
            raise ValueError(f'{stem} lacks a unique macOS RSS measurement')
        return int(values[0])
    if system != 'Darwin':
        raise ValueError('macOS process RSS source required')
    metadata = json.loads((directory / f'{stem}.meta.json').read_bytes())
    value = metadata['child_peak_rss_platform_units']
    if metadata['returncode'] != 0 or type(value) is not int or value <= 0:
        raise ValueError(f'{stem} has invalid process status or RSS')
    return value


def summarize(records):
    pooled = sorted(t for record in records for t in record['trials_seconds'])
    return {
        'median_ms': 1000 * statistics.median(pooled),
        'p95_ms': 1000 * pooled[math.ceil(.95 * len(pooled)) - 1],
        'process_median_ms': [1000 * statistics.median(r['trials_seconds']) for r in records],
        'median_load_ms': 1000 * statistics.median(r['load_seconds'] for r in records),
        'median_peak_rss_bytes': statistics.median(r['peak_rss_bytes'] for r in records),
        'median_mlx_memory': {
            key: statistics.median(r['mlx_memory'][key] for r in records) for key in MEMORY_FIELDS
        },
        'process_measurements': records,
    }


def compare_metrics(candidate, reference):
    return {
        'median_not_faster': candidate['median_ms'] >= reference['median_ms'],
        'p95_not_faster': candidate['p95_ms'] >= reference['p95_ms'],
        'rss_not_lower': candidate['median_peak_rss_bytes'] >= reference['median_peak_rss_bytes'],
        'mlx_peak_higher': candidate['median_mlx_memory']['peak'] > reference['median_mlx_memory']['peak'],
    }


def cell_arg(value):
    if value not in [f'{cfg}:{batch}' for cfg, batch in CELLS]:
        raise argparse.ArgumentTypeError('expected a supported CFG:BATCH cell')
    return tuple(map(int, value.split(':')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='read retained raw reports without running inference')
    parser.add_argument('--cell', type=cell_arg, action='append')
    parser.add_argument('--allocator-trace', action='store_true',
                        help='retain Rust allocator counters between measured trials (diagnostic)')
    parser.add_argument('--lifetime', choices=('retain', 'drop', 'release'))
    parser.add_argument('--cache-mib', type=int, metavar='0..16384')
    parser.add_argument('--materialize-after', type=int, choices=range(1, 11))
    parser.add_argument('--decoder-rows', type=int, choices=range(1, 33))
    parser.add_argument('--control-decoder-rows', type=int, choices=range(1, 33),
                        help='use this tile size for --control decoder (default is the full projection)')
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument('--control', choices=CONTROL_FACTORS,
                          help='compare one Rust setting against a control using the same workload')
    controls.add_argument('--with-control', dest='control', action='store_const', const='checkpoint',
                          help='shorthand for --control checkpoint')
    for mode in ('base', 'cfg'):
        parser.add_argument(f'--{mode}-reference', type=Path,
                            default=ROOT / f'reports/phase4-wmt14-{mode}-cuda-full-v1/python.json')
    args = parser.parse_args()
    cells = args.cell or CELLS
    if args.cache_mib is not None and not 0 <= args.cache_mib <= 16 * 1024:
        parser.error('cache limit must be in 0..16384 MiB')
    if len(set(cells)) != len(cells):
        parser.error('duplicate workload cell')
    if args.replay and (args.allocator_trace or any(option is not None for option in (
            args.control, args.lifetime, args.cache_mib, args.materialize_after,
            args.decoder_rows, args.control_decoder_rows))):
        parser.error('replay reads existing settings')
    factor = args.control
    allocator_trace = args.allocator_trace
    control_value = args.control_decoder_rows
    if factor:
        field, manifest_key, _ = CONTROL_FACTORS[factor]
        screen_value = getattr(args, field)
        if screen_value is None:
            parser.error(f'--control {factor} requires --{field.replace("_", "-")}')
    if control_value is not None and (factor != 'decoder' or control_value == screen_value):
        parser.error('--control-decoder-rows requires --control decoder and a distinct tile size')
    if args.decoder_rows is not None and args.lifetime == 'retain':
        parser.error('--decoder-rows requires dropping logits')
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        parser.error('output must be a new directory under reports')
    reference_paths = {0: args.base_reference.resolve(), 1: args.cfg_reference.resolve()}
    references = {}
    for cfg in {cell[0] for cell in cells}:
        payload = reference_paths[cfg].read_bytes()
        if hashlib.sha256(payload).hexdigest() != REFERENCE_SHA256[cfg]:
            raise ValueError(f'published CFG={cfg} reference differs')
        references[cfg] = json.loads(payload)['outputs']
    watched = [reference_paths[cfg] for cfg in sorted(references)]
    options = []
    control_options = []
    packages = None
    system = platform.platform()
    rss_platform = platform.system()
    if args.replay:
        original_manifest = args.replay / 'manifest.json'
        system, rss_platform = '', ''
        if original_manifest.is_file():
            watched.append(original_manifest)
            original = json.loads(original_manifest.read_bytes())
            system, rss_platform = original.get('system', ''), original.get('rss_platform', '')
            allocator_trace = original.get('allocator_trace', False)
            if type(allocator_trace) is not bool:
                raise ValueError('invalid allocator trace setting in the source manifest')
            screens = [name for name, (_, key, _) in CONTROL_FACTORS.items() if key in original]
            if len(screens) > 1:
                raise ValueError('source manifest has multiple Rust control settings')
            if screens:
                factor = screens[0]
                field, manifest_key, allowed = CONTROL_FACTORS[factor]
                screen_value = original[manifest_key]
                if type(screen_value) is not int or screen_value not in allowed:
                    raise ValueError(f'invalid {factor} screen setting in the source manifest')
            if 'decoder_control_rows' in original:
                control_value = original['decoder_control_rows']
                if (factor != 'decoder' or type(control_value) is not int
                        or control_value not in allowed or control_value == screen_value):
                    raise ValueError('invalid decoder control tile size in the source manifest')
    else:
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise RuntimeError('new timing runs require Apple Silicon')
        if any(os.environ.get(key) for key in (
                'DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
            raise ValueError('instrumented environment is unsupported')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                   cwd=ROOT, text=True).strip():
            raise ValueError('tracked source must be clean')
        os.environ.update(OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        if digest(PREFIX) != '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f':
            raise ValueError('WMT14 prefixes differ')
        if digest(ARTIFACT / 'manifest.json') != '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81':
            raise ValueError('WMT14 artifact differs')
        options = ['reusable', 'deferred-head']
        if allocator_trace:
            options.append('allocator-trace')
        options += {None: ['drop-logits'], 'drop': ['drop-logits'],
                    'retain': [], 'release': ['release-outputs']}[args.lifetime]
        if args.cache_mib is not None:
            options.append(f'cache-mib={args.cache_mib}')
        if args.decoder_rows is not None:
            options.append(f'decoder-rows={args.decoder_rows}')
        if args.materialize_after is not None:
            options.append(f'materialize-after={args.materialize_after}')
        if factor:
            selected_option = f'{field.replace("_", "-")}={screen_value}'
            control_options = [option for option in options if option != selected_option]
            if control_value is not None:
                control_options.append(f'decoder-rows={control_value}')
        requested = {'materialize_after': args.materialize_after, 'decoder_rows': args.decoder_rows,
                     'reusable_loader': True, 'deferred_head': True,
                     'drop_logits': args.lifetime != 'retain', 'release_outputs': args.lifetime == 'release',
                     'cache_limit_bytes': None if args.cache_mib is None else args.cache_mib * 1024**2}
        packages = json.loads(subprocess.check_output(
            [str(PYTHON), '-c', 'import importlib.metadata as m, importlib.util as u, json; '
             'print(json.dumps({"core": u.find_spec("mlx.core").origin, "mlx": m.version("mlx"), '
             '"mlx_metal": m.version("mlx-metal"), "numpy": m.version("numpy")}))'],
            cwd=ROOT, text=True))
        core = Path(packages['core'])
        watched += [core, core.parent / 'lib/libmlx.dylib', core.parent / 'lib/mlx.metallib',
                    BINARY, PYTHON, PREFIX, ARTIFACT / 'manifest.json', ADAPTER,
                    ROOT / 'tools/bench/phase3_process.py',
                    ROOT / 'tools/bench/run_phase3_mlx_bench.py',
                    ROOT / 'tools/reference/phase3_mlx_generator.py',
                    ROOT / 'tools/reference/mlx_codec.py',
                    ROOT / 'tools/reference/mlx_operators.py',
                    ROOT / 'tools/reference/mlx_codec_assets.py',
                    ROOT / '.reference/mlx-native/lib/libmlx.dylib',
                    ROOT / '.reference/mlx-native/lib/mlx.metallib']
    watched += [Path(__file__), ROOT / 'tools/bench/run_process_resources.py']
    before = {str(path): digest(path) for path in watched}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = {'git_head': head, 'sha256': before, 'cells': cells, 'replay': bool(args.replay),
                'rust_options': options, 'warmups': WARMUPS, 'trials': TRIALS,
                'repetitions': REPETITIONS, 'status': 'incomplete',
                'system': system, 'rss_platform': rss_platform, 'python_packages': packages,
                'limitations': ['RSS includes load. Native allocator counters are separate.',
                                'No exclusive GPU reservation. Python is the MLX adaptation.']}
    if allocator_trace:
        manifest['allocator_trace'] = True
    with_control = factor is not None
    if with_control:
        manifest[manifest_key] = screen_value
        if control_value is not None:
            manifest['decoder_control_rows'] = control_value
        manifest['rust_control_options'] = control_options
        manifest['process_order'] = {
            f'{cfg}:{batch}': [stem for _, stem in ordered_runs((cfg, batch), with_control=True)]
            for cfg, batch in cells
        }
        if args.replay and any(original.get('process_order', {}).get(cell) != order
                               for cell, order in manifest['process_order'].items()):
            raise ValueError(f'invalid {factor} screen process order in the source manifest')
    out.mkdir()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    source = args.replay.resolve() if args.replay else out
    settings = {}
    summary = {}
    for cfg, batch in cells:
        per_runtime = {'rust': [], 'python': []}
        if with_control:
            per_runtime['rust_control'] = []
        for variant, stem in ordered_runs((cfg, batch), with_control=with_control):
            runtime = 'python' if variant == 'python' else 'rust'
            report_path = source / f'{stem}.json'
            if not args.replay:
                if runtime == 'rust':
                    variant_options = control_options if variant == 'rust_control' else options
                    command = [str(BINARY), str(ARTIFACT), str(PREFIX), str(cfg), str(batch),
                               str(WARMUPS), str(TRIALS), *variant_options]
                else:
                    command = [str(PYTHON), str(ADAPTER), str(ARTIFACT), str(PREFIX), str(report_path),
                               str(cfg), '0', str(batch), str(WARMUPS), str(TRIALS)]
                run(command, stem, out)
                if runtime == 'rust':
                    report_path.write_bytes((out / f'{stem}.stdout').read_bytes())
            data = report_path.read_bytes()
            report = json.loads(data)
            if runtime == 'rust':
                actual = {key: report.get(key, default) for key, default in (
                    ('reusable_loader', True), ('deferred_head', True),
                    ('drop_logits', report.get('release_outputs', False)),
                    ('release_outputs', False), ('cache_limit_bytes', None), ('materialize_after', None),
                    ('decoder_rows', None))}
                if not args.replay:
                    expected = requested
                    if variant == 'rust_control':
                        expected = {**requested, field: control_value}
                    if (report.get('backend') != 'mlx' or any(key not in report for key in expected)
                            or actual != expected):
                        raise ValueError('Rust execution differs from the requested timing settings')
                if settings.setdefault(variant, actual) != actual:
                    raise ValueError('mixed Rust execution settings')
            record = checked_output(report, runtime, cfg, batch, references[cfg][:batch])
            if runtime == 'rust':
                if (report.get('allocator_boundaries') is not None) != allocator_trace:
                    raise ValueError('WMT14 allocator trace differs from the requested timing settings')
                if allocator_trace:
                    record.update(checked_allocator_trace(report))
            record['source_sha256'] = hashlib.sha256(data).hexdigest()
            record['peak_rss_bytes'] = checked_rss(source, stem, bool(args.replay), rss_platform)
            per_runtime[variant].append(record)
        cell = {variant: summarize(records) for variant, records in per_runtime.items()}
        rust, python = cell['rust'], cell['python']
        cell.update(compare_metrics(rust, python))
        if with_control:
            control = settings['rust_control']
            if control[field] != control_value or settings['rust'] != {**control, field: screen_value}:
                raise ValueError(f'Rust candidate and control settings differ beyond the requested {factor}')
            cell['control_vs_python'] = compare_metrics(cell['rust_control'], python)
            cell[f'{factor}_vs_control'] = compare_metrics(rust, cell['rust_control'])
        summary[f'{cfg}:{batch}'] = cell
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'Completed CFG {cfg} batch {batch}', flush=True)
    if before != {str(path): digest(path) for path in watched}:
        raise RuntimeError('watched input or source changed during comparison')
    if not args.replay and (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head
                            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                                       cwd=ROOT, text=True).strip()):
        raise RuntimeError('tracked source changed during comparison')
    manifest.update(status='complete', rust_settings=settings['rust'],
                    full_timing_matrix=not (with_control or allocator_trace) and set(cells) == set(CELLS))
    if with_control:
        manifest['rust_control_settings'] = settings['rust_control']
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
