"""Fixed read-only diagnostic entry. Execution requires reviewed metadata AND a fresh quiet window.

No clock/power commands, installations, rebuilds or diagnostic retries. Internal
roles require the frozen private plan and their expected live parent command.
Preparation and tests alone are not permission to use --quiet-window-confirmed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import stat
import sys
import threading

from analyze_readonly_trace import read_bounded, REPORT_LIMIT
from bounded_child import run_bounded
from finalize_readonly_trace import equal, parsed, publish, run_then_finalize, sha
from observe_readonly_process import ProcessObservation, process_stat, read_proc
from prepare_readonly_launch import prepare_contract, validate_contract
from readonly_trace_coordinator import coordinate
from readonly_trace_preflight import require
from readonly_trace_runtime import clean_environment
from readonly_trace_sequence import ORDER
from readonly_trace_telemetry import collect
from run_phase2_linux_comparison import observer_affinity
from supervised_worker import direct_children, subreaper, supervise

ENTRY = Path(__file__).resolve()
ROOT = ENTRY.parents[2]
WORKER_TIMEOUT = 970  # coordinator's 950 s plus bounded observer/worker cleanup
OUTER_TIMEOUT = 990
SUPERVISOR_GRACE = 10  # additional cleanup; native 120/600/900 s limits unchanged


def fresh_interpreter():
    require(sys.platform == 'linux' and sys.flags.no_site == 1
            and threading.current_thread() is threading.main_thread() and threading.active_count() == 1
            and len(list(Path('/proc/self/task').iterdir())) == 1
            and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL and not direct_children()
            and subreaper() == 0, 'fresh -S Linux interpreter with exclusive reaping required')
    require(not any(os.environ.get(k) for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONTZPATH')),
            'Python path/timezone overrides forbidden in launch interpreter')


def load_contract(path, expected):
    raw = read_bounded(path, REPORT_LIMIT)
    require(sha(raw) == expected, 'reviewed preparation digest mismatch')
    contract = validate_contract(parsed(raw))
    require(Path(contract['project_root']) == ROOT, 'entry must use its own reviewed checkout')
    tools = {item['path']: item for item in contract['tool_identities']}
    require(str(ENTRY) in tools and str(ROOT / 'tools/bench/sample_linux_cpu.py') in tools,
            'entry/sampler missing from frozen tool inventory')
    image = os.stat('/proc/self/exe')
    python = contract['utilities']['python']
    require((image.st_dev, image.st_ino) == (python['device'], python['inode']), 'wrong Python interpreter image')
    return raw, contract


def private_directory(path):
    require(path.is_dir() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o700,
            'private non-symlink launch directory required')


def role_command(contract, launch, role, parent_pid, plan_sha):
    require(role in ('_supervisor', '_coordinator', '_telemetry')
            and type(parent_pid) is int and parent_pid > 0, 'invalid internal role/parent')
    argv = [contract['libraries']['targets']['taskset']['path'], '-c', '0',
            contract['utilities']['python']['path'], '-S', str(ENTRY), role,
            str(launch / 'preparation.json'), contract['_manifest_sha256'], str(launch),
            '--plan-sha256', plan_sha, '--parent-pid', str(parent_pid)]
    if role == '_telemetry':
        argv += ['--telemetry-dir', str(Path(contract['collection_root']) / 'telemetry')]
    return argv


def preflight_matches(reviewed, current):
    """Ignore only probe times/PIDs and allowed .pi file counts, never measured pins."""
    for key in ('project_root', 'collection_root', 'expected_candidate_head', 'prepared',
                'utilities', 'tool_identities', 'gpu', 'library_roots', 'telemetry_commands', 'limits'):
        equal(reviewed[key], current[key], 'current preflight differs: ' + key)
    for key in ('loader_identity', 'targets', 'libraries', 'listings', 'environment',
                'static_loader_resolution_verified'):
        equal(reviewed['libraries'][key], current['libraries'][key], 'current loader evidence differs: ' + key)
    for key in ('zone', 'localtime_path', 'link_target', 'file_identity'):
        equal(reviewed['timezone'][key], current['timezone'][key], 'current timezone differs: ' + key)
    for key in ('git_identity', 'production_base', 'manifest_base', 'worktree_state_verified'):
        equal(reviewed['worktrees'][key], current['worktrees'][key], 'current Git guard differs: ' + key)
    for name in ('accepted', 'candidate'):
        for key in ('head', 'tracked_clean'):
            equal(reviewed['worktrees']['states'][name][key], current['worktrees']['states'][name][key],
                  'current worktree state differs')


def process_factory(contract):
    """Mandatory for every job, with native AND traced-injection dependencies."""
    prepared, libraries = contract['prepared'], contract['libraries']
    targets = libraries['targets']
    def make(ordinal, job):
        require(type(ordinal) is int and 1 <= ordinal <= len(ORDER), 'invalid observation ordinal')
        equal(job, prepared['jobs'][ordinal - 1], 'observer command differs from frozen job')
        condition = ORDER[ordinal - 1]
        runtime = 'accepted' if condition[0] == 'A' else 'candidate'
        keys = [runtime] + (['injection'] if condition.endswith('T') else [])
        paths = {value['canonical_path'] for key in keys
                 for value in libraries['listings'][key]['dependencies'].values()}
        return ProcessObservation(workload=targets[runtime], workload_argv=job['argv'][3:],
                                  expected_env=job['env'], cpu=2, injection=targets['injection'],
                                  traced=condition.endswith('T'), required_libraries=[libraries['libraries'][p] for p in sorted(paths)],
                                  launcher=targets['taskset'], launcher_argv=job['argv'])
    return make


def check_parent(contract, launch, role, parent_pid, plan_sha):
    require(type(parent_pid) is int and parent_pid > 0 and os.getppid() == parent_pid,
            'internal role parent missing or lost')
    raw = read_bounded(launch / 'entry-plan.json', REPORT_LIMIT)
    require(sha(raw) == plan_sha, 'private launch plan digest mismatch')
    plan = parsed(raw)
    require(plan['schema_version'] == 1 and plan['quiet_window_confirmed'] is True
            and plan['preparation_sha256'] == contract['_manifest_sha256']
            and plan['worker_timeout_seconds'] == WORKER_TIMEOUT and plan['outer_timeout_seconds'] == OUTER_TIMEOUT
            and plan['supervisor_grace_seconds'] == SUPERVISOR_GRACE, 'launch plan mismatch')
    pid, grandparent, _ticks = process_stat(read_proc(f'/proc/{parent_pid}/stat', 65536))
    require(pid == parent_pid, 'parent stat PID mismatch')
    if role == '_supervisor':
        require(parent_pid == plan['root_pid'], 'supervisor not owned by entry')
        expected = plan['root_argv']
    elif role == '_coordinator':
        require(grandparent == plan['root_pid'], 'coordinator ancestry mismatch')
        expected = role_command(contract, launch, '_supervisor', grandparent, plan_sha)[3:]
    else:
        expected = role_command(contract, launch, '_coordinator', grandparent, plan_sha)[3:]
    wanted = b'\0'.join(value.encode() for value in expected) + b'\0'
    require(read_proc(f'/proc/{parent_pid}/cmdline', 65536) == wanted, 'unexpected parent command')
    image = os.stat(f'/proc/{parent_pid}/exe')
    python = contract['utilities']['python']
    require((image.st_dev, image.st_ino) == (python['device'], python['inode']), 'unexpected parent image')
    require(os.getppid() == parent_pid, 'parent lost during role preflight')
    # Unexpected command/environment values are never serialized.


def internal_role(role, manifest, expected_sha, launch, parent_pid, plan_sha, telemetry_dir=None):
    require(role in ('_supervisor', '_coordinator', '_telemetry'), 'unknown internal role')
    fresh_interpreter()
    private_directory(launch)
    require(manifest == launch / 'preparation.json', 'internal roles require the private preparation copy')
    _raw, contract = load_contract(manifest, expected_sha)
    contract['_manifest_sha256'] = expected_sha
    require(telemetry_dir == (Path(contract['collection_root']) / 'telemetry' if role == '_telemetry' else None),
            'internal telemetry directory mismatch')
    equal(dict(os.environ), clean_environment(contract['prepared']), 'internal environment differs from isolation contract')
    require(os.getcwd() == str(ROOT) and os.sched_getaffinity(0) == {0}, 'internal cwd/affinity mismatch')
    check_parent(contract, launch, role, parent_pid, plan_sha)
    env = clean_environment(contract['prepared'])
    if role == '_supervisor':
        result = supervise(role_command(contract, launch, '_coordinator', os.getpid(), plan_sha),
                           cwd=ROOT, env=env, output_dir=launch / 'supervision',
                           timeout_seconds=WORKER_TIMEOUT, parent_pid=parent_pid)
    elif role == '_coordinator':
        source = next(v['sha256'] for v in contract['tool_identities']
                      if v['path'] == str(ROOT / 'tools/bench/sample_linux_cpu.py'))
        result = coordinate(contract['prepared']['jobs'],
                            role_command(contract, launch, '_telemetry', os.getpid(), plan_sha),
                            cwd=ROOT, env=env, output_dir=Path(contract['collection_root']),
                            expected_cpu_source=source, gpu_uuid=contract['gpu']['uuid'],
                            process_observer_factory=process_factory(contract))
    else:
        source = next(v['sha256'] for v in contract['tool_identities']
                      if v['path'] == str(ROOT / 'tools/bench/sample_linux_cpu.py'))
        result = collect(contract['telemetry_commands']['cpu'], contract['telemetry_commands']['gpu'],
                         cwd=ROOT, env=env, output_dir=Path(contract['collection_root']) / 'telemetry',
                         expected_cpu_source=source, gpu_uuid=contract['gpu']['uuid'])
    return 0 if result['status'] == 'complete' else 1


def launch_collection(manifest, expected_sha, launch, *, quiet_window_confirmed=False):
    require(quiet_window_confirmed is True, 'fresh quiet-window confirmation required; prior selection is insufficient')
    fresh_interpreter()
    raw, contract = load_contract(manifest, expected_sha)
    launch = Path(launch).absolute()
    require(not os.path.lexists(launch), 'fresh launch directory required')
    launch = launch.resolve()
    collection = Path(contract['collection_root'])
    require(launch.parent.is_dir() and launch != collection and not launch.is_relative_to(collection)
            and not collection.is_relative_to(launch), 'separate launch/collection paths required')
    launch.mkdir(mode=0o700)
    # Revalidate the current clean checkout, all inputs/tools, loader resolution,
    # GPU UUID and timezone BEFORE any sampler or workload. Never do this in roles.
    fresh = prepare_contract(ROOT, collection, launch / 'preflight',
                             expected_candidate_head=contract['expected_candidate_head'],
                             expected_gpu_pci=contract['gpu']['pci_bus_id'], expected_gpu_name=contract['gpu']['name'],
                             home=Path(contract['prepared']['observer_env']['HOME']),
                             python=Path(contract['utilities']['python']['path']),
                             nvidia_smi=Path(contract['utilities']['nvidia_smi']['path']),
                             taskset=Path(contract['libraries']['targets']['taskset']['path']))
    current_raw = read_bounded(Path(fresh['manifest']), REPORT_LIMIT)
    require(sha(current_raw) == fresh['sha256'], 'current preflight changed during read')
    preflight_matches(contract, parsed(current_raw))
    plan = dict(schema_version=1, quiet_window_confirmed=True, preparation_sha256=expected_sha,
                current_preflight_sha256=fresh['sha256'], root_pid=os.getpid(),
                root_argv=read_proc('/proc/self/cmdline', 65536).decode().rstrip('\0').split('\0'),
                worker_timeout_seconds=WORKER_TIMEOUT, outer_timeout_seconds=OUTER_TIMEOUT,
                supervisor_grace_seconds=SUPERVISOR_GRACE,
                scope='Explicit invocation flag records caller confirmation, not independent quietness proof.')
    plan_id = publish(launch, 'entry-plan.json', plan)
    command_contract = {**contract, '_manifest_sha256': expected_sha}
    argv = role_command(command_contract, launch, '_supervisor', os.getpid(), plan_id['sha256'])
    def runner():
        return run_bounded(argv, cwd=ROOT, env=clean_environment(contract['prepared']),
                           capture_path=launch / 'supervisor.txt', receipt_path=launch / 'supervisor.receipt.json',
                           timeout_seconds=OUTER_TIMEOUT, output_limit_bytes=1024**2,
                           termination_grace_seconds=SUPERVISOR_GRACE)
    with observer_affinity(0):
        result = run_then_finalize(raw, expected_sha, launch, runner)
    require(result['status'] == 'complete' and result['validation_composition_complete'] is True
            and result['frozen_rehash_after_runner_and_validation'] is True
            and result['performance_acceptance_evaluated'] is False,
            'diagnostic finalization did not complete')
    # The returned completion binds all consumed raw outputs. Bind the entry plan
    # and fresh preflight too, without publishing raw Git status/session names.
    require(read_bounded(launch / 'entry-plan.json', REPORT_LIMIT) == json.dumps(plan, indent=2, allow_nan=False).encode() + b'\n',
            'entry plan changed during collection')
    equal(parsed(read_bounded(launch / 'supervisor.receipt.json', REPORT_LIMIT))['argv'], argv,
          'outer supervisor command binding mismatch')
    entry_result = dict(schema_version=1, status='complete', preparation_sha256=expected_sha,
                        entry_plan=plan_id, current_preflight_sha256=fresh['sha256'],
                        completion_sha256=sha(read_bounded(launch / 'completion/completion.json', REPORT_LIMIT)),
                        reported_codec_outputs_qualified=result['validation_composition_complete'],
                        performance_acceptance_evaluated=False, launch_provenance_verified=False,
                        scope='Fixed reviewed entry and successful diagnostic validation only; not performance acceptance, '
                              'immutable mapped bytes, continuous observation or independent quietness proof.')
    publish(launch, 'entry-result.json', entry_result)
    return entry_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('launch', '_supervisor', '_coordinator', '_telemetry'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('sha256')
    parser.add_argument('launch_dir', type=Path)
    parser.add_argument('--quiet-window-confirmed', action='store_true')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--telemetry-dir', type=Path)
    args = parser.parse_args()
    require(sys.flags.no_site == 1, 'invoke entry with python -S')
    if args.role == 'launch':
        require(args.plan_sha256 is None and args.parent_pid is None and args.telemetry_dir is None,
                'internal flags forbidden on launch')
        result = launch_collection(args.manifest, args.sha256, args.launch_dir,
                                   quiet_window_confirmed=args.quiet_window_confirmed)
        print(json.dumps(result, indent=2))
        return 0
    require(not args.quiet_window_confirmed and args.plan_sha256 is not None and args.parent_pid is not None,
            'internal role requires explicit parent/plan, not an availability flag')
    return internal_role(args.role, args.manifest, args.sha256, args.launch_dir, args.parent_pid,
                         args.plan_sha256, args.telemetry_dir)


if __name__ == '__main__':
    raise SystemExit(main())
