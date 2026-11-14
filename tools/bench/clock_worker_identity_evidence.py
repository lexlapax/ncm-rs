"""Private sampled worker identity evidence, not launch or provenance proof.

The caller supplies an independently authenticated expected identity. The
version-1 record is a separate artifact; it does not change an existing schema.
"""
import json
import os
from pathlib import Path
import stat

import diagnostic_clock
from clock_unprivileged_identity import STATUS_LIMIT, require, verify

NAME = 'worker-identity.json'
RECORD_LIMIT = 16384
MAX_LINUX_ID = 2**32 - 2


def target(uid, gid, groups, require_no_new_privs):
    require(type(uid) is int and 0 < uid <= MAX_LINUX_ID
            and type(gid) is int and 0 < gid <= MAX_LINUX_ID
            and isinstance(groups, (list, tuple)) and len(groups) <= 1024
            and all(type(value) is int and 0 <= value <= MAX_LINUX_ID for value in groups)
            and type(require_no_new_privs) is bool, 'explicit worker identity declaration required')
    return {'uid':uid, 'gid':gid, 'groups':sorted(groups),
            'require_no_new_privs':require_no_new_privs}


def no_new_privs():
    with open('/proc/self/status', 'rb') as stream:
        raw = stream.read(STATUS_LIMIT+1)
    require(len(raw) <= STATUS_LIMIT, 'oversized proc status')
    matches = [line.partition(b':')[2].strip() for line in raw.splitlines()
               if line.partition(b':')[0] == b'NoNewPrivs']
    require(len(matches) == 1 and matches[0] in (b'0', b'1'), 'invalid no_new_privs state')
    return matches[0] == b'1'


def write(directory, uid, gid, groups, *, require_no_new_privs):
    """Sample this process, then exclusively write one bounded private record.

    The output directory must already exist, be canonical, private and owned by
    this process. Nothing here authenticates the declared identity, parent,
    process source, continued credentials or the final filesystem consumer.
    """
    declared = target(uid, gid, groups, require_no_new_privs)
    directory = Path(directory)
    require(directory.is_absolute() and directory.resolve(strict=True) == directory,
            'canonical identity evidence directory required')
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISDIR(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o700
                and before.st_uid == os.getuid(), 'owned private identity directory required')
        life = diagnostic_clock.begin()
        result = verify(uid, gid, groups)
        nnp = no_new_privs()
        require(result['sampled_unprivileged_identity_verified'] is True
                and result['launch_ready'] is False and (not require_no_new_privs or nnp),
                'worker credential sample incomplete')
        require(os.getpid() == life['pid'] and os.getppid() == life['parent_pid'],
                'worker or parent changed during credential sample')
        record = {'schema_version':1, 'kind':'clock-worker-identity', 'target':declared,
                  'no_new_privs':nnp, 'lifetime':diagnostic_clock.finish(life),
                  'sampled_unprivileged_identity_verified':True,
                  'identity_declaration_authenticated':False, 'continuous_identity_verified':False,
                  'launch_ready':False}
        raw = json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode()+b'\n'
        require(len(raw) <= RECORD_LIMIT, 'worker identity record exceeds limit')
        current = os.stat(directory, follow_symlinks=False)
        require((before.st_dev, before.st_ino) == (current.st_dev, current.st_ino)
                and stat.S_IMODE(current.st_mode) == 0o700 and current.st_uid == os.getuid(),
                'worker identity directory changed')
        out = os.open(NAME, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=fd)
        with os.fdopen(out, 'wb') as stream:
            output_stat = os.fstat(stream.fileno())
            require(stat.S_ISREG(output_stat.st_mode) and stat.S_IMODE(output_stat.st_mode) == 0o600
                    and output_stat.st_uid == os.getuid() and output_stat.st_nlink == 1,
                    'owned private worker identity file required')
            require(stream.write(raw) == len(raw), 'incomplete worker identity write')
        return record
    finally:
        os.close(fd)
