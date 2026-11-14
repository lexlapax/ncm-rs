"""Sample Linux worker credentials; never change rights or launch work.

Expected IDs/groups must be authenticated by the future entry. A sample does not
prove continuous credentials or authorize workload execution or a GPU action.
"""
import os
import re
import sys
import threading
from pathlib import Path

STATUS_LIMIT = 65536
CAPABILITY_FIELDS = ('CapInh', 'CapPrm', 'CapEff', 'CapAmb')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _credentials(raw):
    require(isinstance(raw, bytes) and len(raw) <= STATUS_LIMIT, 'bounded proc status required')
    fields = {}
    wanted = {'Uid', 'Gid', 'Groups', *CAPABILITY_FIELDS}
    for line in raw.splitlines():
        name, separator, value = line.partition(b':')
        try:
            key = name.decode('ascii')
        except UnicodeDecodeError:
            continue
        if key in wanted:
            require(separator == b':' and key not in fields, 'duplicate or invalid credential field')
            fields[key] = value.strip()
    require(set(fields) == wanted, 'incomplete proc credential fields')
    def numbers(field, count=None):
        parts = fields[field].split()
        require((count is None or len(parts) == count)
                and all(re.fullmatch(rb'[0-9]+', part) for part in parts),
                'invalid proc credential IDs')
        return tuple(int(part) for part in parts)
    ids, gids, groups = numbers('Uid', 4), numbers('Gid', 4), numbers('Groups')
    caps = []
    for name in CAPABILITY_FIELDS:
        value = fields[name]
        require(re.fullmatch(rb'[0-9a-fA-F]{1,32}', value) is not None,
                'invalid proc capability field')
        caps.append(int(value, 16))
    return ids, gids, groups, tuple(caps)


def verify(expected_uid, expected_gid, expected_groups):
    """Check sampled IDs/groups and active capabilities against supplied identity.

    The four proc IDs include filesystem credentials. Effective, permitted,
    inheritable and ambient capabilities must be empty; bounding capabilities
    are not active capabilities. This helper does not drop rights.
    """
    require(sys.platform == 'linux' and threading.current_thread() is threading.main_thread()
            and threading.active_count() == 1 and len(list(Path('/proc/self/task').iterdir())) == 1
            and type(expected_uid) is int and expected_uid > 0
            and type(expected_gid) is int and expected_gid > 0
            and isinstance(expected_groups, (tuple, list))
            and all(type(g) is int and g >= 0 for g in expected_groups),
            'single-threaded Linux worker and explicit non-root identity required')
    with open('/proc/self/status', 'rb') as stream:
        raw = stream.read(STATUS_LIMIT + 1)
    uids, gids, groups, caps = _credentials(raw)
    require(uids == (expected_uid,) * 4 and gids == (expected_gid,) * 4
            and tuple(sorted(groups)) == tuple(sorted(expected_groups))
            and os.getresuid() == (expected_uid,) * 3
            and os.getresgid() == (expected_gid,) * 3
            and tuple(sorted(os.getgroups())) == tuple(sorted(expected_groups))
            and not any(caps), 'worker identity or capabilities differ')
    return {'schema_version': 1, 'sampled_unprivileged_identity_verified': True,
            'identity_declaration_authenticated': False, 'continuous_identity_verified': False,
            'launch_ready': False}
