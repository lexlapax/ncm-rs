"""Irreversible credentials transition for a dedicated Linux worker process.

Import-only; NOT a diagnostic entry. The caller must authenticate target IDs,
all code and descriptors before invoking this in a disposable child. Any failure
after attempting a security change terminates the process without returning to
privileged caller code. No GPU or policy commands are present.
"""
import ctypes
import errno
import os
from pathlib import Path
import sys
import threading

from clock_unprivileged_identity import require, verify

PR_SET_NO_NEW_PRIVS = 38
PR_GET_NO_NEW_PRIVS = 39
DROP_FAILURE_EXIT = 125
MAX_LINUX_ID = 2 ** 32 - 2
LINUX_CAPABILITY_VERSION_3 = 0x20080522


class _CapabilityHeader(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]


class _CapabilityData(ctypes.Structure):
    _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32),
                ('inheritable', ctypes.c_uint32)]


def _no_extra_fds():
    # listdir's own temporary descriptor is already closed on return.
    for value in os.listdir('/proc/self/fd'):
        if not value.isdecimal() or int(value) <= 2:
            continue
        try:
            os.fstat(int(value))
        except OSError as error:
            if error.errno == errno.EBADF:
                continue
            raise
        raise ValueError('unexpected inherited descriptor')


def _prctl(option, value):
    library = ctypes.CDLL(None, use_errno=True)
    fn = library.prctl
    fn.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    fn.restype = ctypes.c_int
    result = fn(option, value, 0, 0, 0)
    if result == -1:
        raise OSError(ctypes.get_errno(), 'prctl failed')
    return result


def _clear_capabilities():
    """Drop all capability sets on this thread; never request an added bit.

    Linux capability ABI v3 uses two 32-bit words for each capability set.
    Clearing permitted and inheritable also lowers corresponding ambient bits;
    the later procfs verification independently requires all four sets empty.
    """
    header = _CapabilityHeader(LINUX_CAPABILITY_VERSION_3, 0)
    data = (_CapabilityData * 2)()  # ctypes zero-initializes all six fields.
    library = ctypes.CDLL(None, use_errno=True)
    fn = library.capset
    fn.argtypes = (ctypes.POINTER(_CapabilityHeader), ctypes.POINTER(_CapabilityData))
    fn.restype = ctypes.c_int
    if fn(ctypes.byref(header), data) != 0:
        raise OSError(ctypes.get_errno(), 'capset failed')


def drop_and_verify(uid, gid, groups):
    """Permanently drop root in this process; caller must already be trusted.

    Sets no_new_privs, supplementary groups, all GIDs and all UIDs, then clears
    capability sets and verifies the resulting credentials. The no_new_privs bit is inherited
    by future children. Any failure after the first change is fatal to this
    dedicated process, since privileges may already have changed. The supplied
    identity is not authenticated here and successful sampling does not prove
    continuous identity, restored GPU policy or launch readiness.
    """
    require(sys.platform == 'linux' and sys.flags.no_site == 1
            and threading.current_thread() is threading.main_thread()
            and threading.active_count() == 1 and len(list(Path('/proc/self/task').iterdir())) == 1,
            'dedicated single-threaded Linux -S process required')
    require(os.getresuid() == (0, 0, 0) and os.getresgid() == (0, 0, 0),
            'fully root process required')
    require(type(uid) is int and 0 < uid <= MAX_LINUX_ID and type(gid) is int and 0 < gid <= MAX_LINUX_ID
            and isinstance(groups, (list, tuple))
            and len(groups) <= 65536 and all(type(g) is int and 0 <= g <= MAX_LINUX_ID for g in groups),
            'explicit non-root target identity required')
    target_groups = tuple(groups)
    _no_extra_fds()
    try:
        if _prctl(PR_SET_NO_NEW_PRIVS, 1) != 0:
            raise ValueError('could not set no_new_privs')
        os.setgroups(target_groups)
        os.setresgid(gid, gid, gid)
        os.setresuid(uid, uid, uid)
        _clear_capabilities()
        result = verify(uid, gid, target_groups)
        if _prctl(PR_GET_NO_NEW_PRIVS, 0) != 1:
            raise ValueError('no_new_privs not retained')
    except BaseException:
        # No Python-level recovery or caller continuation after a partial drop.
        os._exit(DROP_FAILURE_EXIT)
    return {**result, 'no_new_privs_verified': True,
            'privilege_drop_authenticated': False, 'launch_ready': False}
