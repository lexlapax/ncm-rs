"""Precollection recheck of declared clock files; not a complete preflight.

No workload, telemetry, hardware, policy or subprocess action. This checks only
supplied declarations and pathnames, not completeness, ancestry or mapped bytes.
"""
from pathlib import Path
import stat

from readonly_trace_preflight import file_identity, require
from verify_clock_files import canonical_spelling, declared_inventory


def verify(contract, *, frozen_identities, path_bindings, injection):
    """Check the whole admitted inventory before collection, and aliases twice.

    Inventory all file identities/sizes before hashing; then hash all admitted files
    and recheck observable metadata and alias resolution. Never claim an atomic
    filesystem snapshot, runtime provenance or collection readiness. Future entry
    must use the same frozen declarations and separately verify producer coverage,
    Git/worktree status, command roles, policy and permissions.
    """
    pinned, aliases = declared_inventory(contract, frozen_identities=frozen_identities,
                                         path_bindings=path_bindings, injection=injection)
    names = sorted(pinned)

    def check_paths():
        for name in names:
            path = canonical_spelling(name)
            require(str(path.resolve(strict=True)) == name, 'prepared pathname became noncanonical')
        for spelling, target in aliases.items():
            require(str(Path(spelling).resolve(strict=True)) == target, 'prepared alias target changed')

    check_paths()
    before = {}
    for name in names:
        current = Path(name).stat(follow_symlinks=False)
        expected = pinned[name]
        require(stat.S_ISREG(current.st_mode)
                and (current.st_dev, current.st_ino, current.st_size)
                == (expected['device'], expected['inode'], expected['bytes']),
                'prepared file identity changed')
        before[name] = current
    for name in names:
        expected = pinned[name]
        observed = file_identity(name, expected['sha256'], expected['bytes'])
        require((observed['device'], observed['inode']) == (expected['device'], expected['inode']),
                'prepared file replaced while hashing')
    check_paths()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    for name in names:
        current = Path(name).stat(follow_symlinks=False)
        require(all(getattr(before[name], key) == getattr(current, key) for key in fields),
                'prepared file changed during recheck')
    return {'schema_version': 1, 'declared_file_count': len(names),
            'declared_file_bytes': sum(pinned[name]['bytes'] for name in names),
            'declared_alias_count': len(aliases), 'supplied_files_rechecked': True,
            'supplied_aliases_rechecked': True, 'inventory_completeness_verified': False,
            'producer_authenticity_verified': False, 'atomic_snapshot_verified': False,
            'launch_ready': False,
            'scope': 'Current supplied pathname identities/digests and alias resolutions only; '
                     'not complete preparation, atomic bytes, producer or runtime provenance, '
                     'policy, permission, quietness or launch authorization.'}
