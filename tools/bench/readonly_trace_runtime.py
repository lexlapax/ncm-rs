"""Bounded metadata-only loader/Git probes for the read-only diagnostic.

The glibc --list mode does not run benchmark main. DT_NEEDED listings are NOT
coverage of later dlopen/driver/JIT libraries or proof of live process mappings.
Only trusted, identity-rechecked inputs and local installed utilities are allowed.
"""
import json
import os
from pathlib import Path
import re

from bounded_child import run_bounded
from readonly_trace_preflight import file_identity, require

ACCEPTED_HEAD = '3307dff82a08f7251b64de0866a01d131528b74f'
PRODUCTION_HEAD = 'd44beaa82bbf026b353cb44461da44d25efbfb87'
# The measured v2 tree already includes the approved codec_memory example entry.
MANIFEST_HEAD = 'fa040ed34a44a577bc4e6e297c944393cdf6cfda'
ENV_KEYS = {'PATH', 'HOME', 'LANG', 'LC_ALL', 'PYTHONNOUSERSITE'}


def parse_loader_listing(text):
    dependencies, virtual = {}, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.fullmatch(r'linux-vdso\.so\.1 \(0x[0-9a-f]+\)', line)
        if match:
            require(not virtual, 'duplicate virtual dependency')
            virtual.append('linux-vdso.so.1')
            continue
        match = re.fullmatch(r'(?:(\S+) => )?(/[^\s]+) \(0x[0-9a-f]+\)', line)
        require(match is not None, 'unsupported or unresolved loader listing')
        alias, path = match.groups()
        name = Path(alias or path).name
        require(name not in dependencies, 'duplicate loader dependency')
        dependencies[name] = path
    require(dependencies and 'libc.so.6' in dependencies, 'incomplete native loader listing')
    return dependencies, virtual


def metadata_probe(argv, cwd, env, directory, name):
    """Retain bounded command/capture receipts; caller creates a private directory."""
    receipt = run_bounded(argv, cwd=cwd, env=env,
                          capture_path=directory / (name + '.txt'),
                          receipt_path=directory / (name + '.receipt.json'),
                          timeout_seconds=10, output_limit_bytes=1024 ** 2)
    require(receipt['status'] == 'complete', 'metadata command failed; inspect retained receipt')
    return (directory / (name + '.txt')).read_text(), receipt


def clean_environment(prepared):
    env = dict(prepared['observer_env'])
    require(set(env) == ENV_KEYS and env['PATH'] == os.defpath and env['LANG'] == env['LC_ALL'] == 'C'
            and env['PYTHONNOUSERSITE'] == '1' and Path(env['HOME']).is_absolute(),
            'unexpected metadata environment')
    return env


def inspect_libraries(prepared, output_dir, *, loader=Path('/usr/lib/ld-linux-x86-64.so.2')):
    """Recheck inputs, list trusted ELF dependencies, hash canonical library files.

    Records current system libraries; no historical library pin is invented. The
    resulting listing and hashes must be frozen/rechecked for actual collection.
    """
    env = clean_environment(prepared)
    identities = {entry['path']: entry for entry in prepared['file_identities']}
    jobs = prepared['jobs']
    require(len(jobs) == 12 and jobs[0]['condition'] == 'AP' and jobs[1]['condition'] == 'CP'
            and jobs[2]['condition'] == 'AT', 'invalid prepared job inventory')
    targets = {'accepted': jobs[0]['argv'][3], 'candidate': jobs[1]['argv'][3],
               'injection': jobs[2]['env']['CUDA_INJECTION64_PATH'], 'taskset': jobs[0]['argv'][0]}
    # All target identities checked before the first utility process is launched.
    target_ids = {key: file_identity(path, identities[path]['sha256'], identities[path]['bytes'])
                  for key, path in targets.items()}
    loader = Path(loader).resolve(strict=True)
    loader_identity = file_identity(loader)
    directory = Path(output_dir)
    directory.mkdir(mode=0o700)
    listings, library_ids, receipts = {}, {}, {}
    for key, path in targets.items():
        text, receipt = metadata_probe([str(loader), '--list', path], directory, env, directory, key)
        dependencies, virtual = parse_loader_listing(text)
        required = {'libc.so.6'}
        if key in ('accepted', 'candidate'):
            required |= {'libcuda.so.1', 'libcurand.so.10', 'libcublas.so.13'}
        elif key == 'injection':
            required |= {'libcuda.so.1', 'libcupti.so.13'}
        require(required <= dependencies.keys(), 'missing required native dependency')
        resolved = {}
        for name, listed in dependencies.items():
            canonical = str(Path(listed).resolve(strict=True))
            if canonical not in library_ids:
                library_ids[canonical] = file_identity(canonical)
            resolved[name] = {'listed_path': listed, 'canonical_path': canonical}
        listings[key] = {'dependencies': resolved, 'virtual_dependencies': virtual}
        receipts[key] = receipt
        file_identity(path, target_ids[key]['sha256'], target_ids[key]['bytes'])
    require(listings['accepted']['dependencies'] == listings['candidate']['dependencies'],
            'accepted/candidate native library resolution differs')
    file_identity(loader, loader_identity['sha256'], loader_identity['bytes'])
    # Recheck paths/hashes after all listings, including mutable library symlinks.
    for listing in listings.values():
        for dependency in listing['dependencies'].values():
            require(str(Path(dependency['listed_path']).resolve(strict=True)) == dependency['canonical_path'],
                    'library symlink changed during inspection')
    for path, identity in library_ids.items():
        file_identity(path, identity['sha256'], identity['bytes'])
    result = {'schema_version': 1, 'diagnostic_only': True, 'static_loader_resolution_verified': True,
              'runtime_loaded_libraries_verified': False, 'launch_ready': False,
              'environment': env, 'loader_identity': loader_identity, 'targets': target_ids,
              'listings': listings, 'libraries': library_ids, 'receipts': receipts,
              'scope': 'glibc --list DT_NEEDED resolution and current file hashes only; not benchmark '
                       'execution, CUDA initialization, historical library equivalence, lazy dlopen, '
                       'driver/JIT dependencies, runtime mapping proof or collection readiness.'}
    with (directory / 'libraries.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    return result


def verify_worktrees(root, expected_candidate_head, output_dir, *, git=Path('/usr/bin/git')):
    """Require accepted HEAD and clean candidate at the caller's predeclared HEAD.

    Only .pi/ untracked session files are allowed in the candidate. Never serialize
    their names in the summary; raw command logs stay in a private local directory.
    No checkout, fetch, reset, index refresh request, hook or credential operation.
    """
    require(re.fullmatch('[0-9a-f]{40}', expected_candidate_head) is not None, 'invalid candidate HEAD')
    root, git = Path(root).resolve(strict=True), Path(git).resolve(strict=True)
    git_identity = file_identity(git)
    directory = Path(output_dir)
    directory.mkdir(mode=0o700)
    env = {'PATH': os.defpath, 'HOME': str(root), 'LANG': 'C', 'LC_ALL': 'C',
           'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
           'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0'}
    def probe(tree, arguments, label):
        command = [str(git), '--no-optional-locks', '-c', 'core.fsmonitor=false',
                   '-c', 'core.untrackedCache=false', '-C', str(tree), *arguments]
        return metadata_probe(command, root, env, directory, label)[0]
    states = {}
    for key, tree, expected in (('accepted', root / '.reference/phase2-optimization-baseline', ACCEPTED_HEAD),
                                ('candidate', root, expected_candidate_head)):
        head = probe(tree, ['rev-parse', 'HEAD'], key + '-head').strip()
        require(head == expected, 'worktree HEAD mismatch')
        raw = probe(tree, ['status', '--porcelain=v1', '-z', '--untracked-files=all'], key + '-status')
        entries = raw.split('\0')[:-1] if raw.endswith('\0') else []
        require(not raw or raw.endswith('\0'), 'truncated Git status')
        allowed = key == 'candidate' and all(e.startswith('?? .pi/') for e in entries)
        require(not entries or allowed, 'worktree contains unrelated/uncommitted work')
        states[key] = {'head': head, 'tracked_clean': True, 'allowed_untracked_count': len(entries)}
    changed = probe(root, ['diff', '--no-ext-diff', '--no-textconv', '--name-only', PRODUCTION_HEAD, 'HEAD',
                          '--', 'Cargo.lock', ':(glob)crates/*/src/**'], 'production-diff')
    require(not changed.strip(), 'production changed since frozen candidate')
    manifest_changes = probe(root, ['diff', '--no-ext-diff', '--no-textconv', '--name-only', MANIFEST_HEAD, 'HEAD',
                                   '--', 'Cargo.toml', ':(glob)crates/*/Cargo.toml'], 'manifest-diff')
    require(not manifest_changes.strip(), 'Cargo manifests changed since measured v2 tree')
    file_identity(git, git_identity['sha256'], git_identity['bytes'])
    result = {'schema_version': 1, 'worktree_state_verified': True, 'launch_ready': False,
              'git_identity': git_identity, 'states': states, 'production_base': PRODUCTION_HEAD,
              'manifest_base': MANIFEST_HEAD,
              'scope': 'Read-only current Git state; recheck at launch. Ignored artifacts are '
                       'covered separately by frozen hashes, not by Git cleanliness.'}
    with (directory / 'worktrees.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    return result
