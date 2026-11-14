"""Bounded post-collection evidence reads, not live supervision or finalization.

Trusted owned private directories, no symlink traversal, regular private leaves,
visible-change checks and explicit aggregate read budget. Current file identities
cannot authenticate producers or prove an atomic/immutable filesystem snapshot.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from clock_codec_sequence import CAPTURE_BYTES, REPORT_BYTES, MIN_FREE_BYTES
from clock_process_observation import ORDER
from readonly_trace_preflight import require
from validate_clock_restoration import finite
from validate_readonly_lifecycle import digest, receipt_complete
from validate_readonly_telemetry import unique_object, reject_constant
from verify_readonly_processes import key, number

FILE_LIMIT = 128
TOTAL_BYTES = 1024**3
META_BYTES = 16*1024**2
FIELDS = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_uid','st_gid')


def parts(name):
    require(isinstance(name,str) and name and '\0' not in name and '\\' not in name,
            'invalid relative evidence path')
    path=PurePosixPath(name)
    require(not path.is_absolute() and str(path)==name and all(p not in ('.','..') for p in path.parts)
            and path.parts, 'normalized relative evidence path required')
    return path.parts


def same(first,second):
    return all(getattr(first,k)==getattr(second,k) for k in FIELDS)


def private(value,directory=False):
    require((stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode))
            and value.st_uid==os.getuid() and value.st_mode & 0o077 == 0,
            'owned private regular evidence file/directory required')


class Evidence:
    """Read each relative path once; keep identities and recheck metadata explicitly.

    Caller must wait for all writer branches before constructing this object. No
    automatic cleanup, process signalling or frozen preparation-file hashing occurs.
    """
    def __init__(self,root):
        self.root=Path(root).absolute()
        require(self.root.resolve(strict=True)==self.root, 'canonical evidence root required')
        self.fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            self.initial=os.fstat(self.fd); private(self.initial,True)
        except BaseException:
            os.close(self.fd); raise
        self.inputs={}; self.snapshots={}; self.total=0

    def __enter__(self): return self

    def __exit__(self,*_):
        os.close(self.fd); self.fd=None

    def _open(self,name):
        components=parts(name)
        require(self.fd is not None and self.root.resolve(strict=True)==self.root, 'evidence root changed/closed')
        current=self.root.stat(follow_symlinks=False)
        require((current.st_dev,current.st_ino)==(self.initial.st_dev,self.initial.st_ino), 'evidence root replaced')
        fd=os.dup(self.fd)
        try:
            private(os.fstat(fd),True)
            for component in components[:-1]:
                child=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                os.close(fd); fd=child; private(os.fstat(fd),True)
            return os.open(components[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        finally:
            os.close(fd)

    def raw(self,name,limit=META_BYTES):
        require(type(limit) is int and 0 < limit <= META_BYTES and name not in self.inputs
                and len(self.inputs)<FILE_LIMIT, 'invalid/duplicate/oversized evidence read inventory')
        fd=self._open(name)
        try:
            before=os.fstat(fd); private(before)
            require(before.st_size<=limit and self.total+before.st_size<=TOTAL_BYTES,
                    'evidence read byte budget exceeded')
            chunks=[]; size=0
            while chunk:=os.read(fd,min(65536,limit+1-size)):
                size+=len(chunk); require(size<=limit,'evidence grew beyond read bound'); chunks.append(chunk)
            require(size==before.st_size and same(before,os.fstat(fd)), 'evidence changed during read')
            current=self._open(name)
            try: require(same(before,os.fstat(current)), 'evidence pathname changed during read')
            finally: os.close(current)
        finally:
            os.close(fd)
        raw=b''.join(chunks)
        self.total+=size; self.snapshots[name]=before
        self.inputs[name]=dict(path=str(self.root/name),device=before.st_dev,inode=before.st_ino,
                               bytes=size,sha256=hashlib.sha256(raw).hexdigest())
        return raw

    def record(self,name,limit=META_BYTES):
        return json.loads(self.raw(name,limit),object_pairs_hook=unique_object,parse_constant=reject_constant)

    def verify_unchanged(self):
        """Recheck all previously read identities/metadata; not an atomic snapshot."""
        for name,before in self.snapshots.items():
            fd=self._open(name)
            try: require(same(before,os.fstat(fd)), 'evidence changed after read')
            finally: os.close(fd)


def read_sequence(evidence,*,jobs,work_deadline,sequence_dir='sequence'):
    """Read 74 files against supplied frozen jobs/deadline, with no workload launch.

    Returns exact raw report bytes for qualification plus parsed plan/result and
    the evidence inventory. The caller must separately run process/lifecycle/output
    validators on these same records and finish all frozen-file rechecks last.
    """
    parts(sequence_dir)
    require(isinstance(jobs,list) and len(jobs)==18 and finite(work_deadline)>0,
            'eighteen frozen jobs and original deadline required')
    expected=[]
    for i,(runtime,job) in enumerate(zip(ORDER,jobs),1):
        repeat=(i-1)//3+1; language='python' if runtime=='python' else 'rust'
        prefix=f'{sequence_dir}/{runtime}/codec-{language}-{repeat}'
        report=str(evidence.root/(prefix+'.json'))
        require(job['runtime']==runtime and number(job['repeat'],1)==repeat and job['workload']=='codec'
                and job['report']==report and job['argv'][-1]==report, 'frozen sequence report plan differs')
        expected.append(prefix)
    plan=evidence.record(sequence_dir+'/sequence-plan.json')
    result=evidence.record(sequence_dir+'/sequence-result.json')
    require(type(plan['schema_version']) is int and plan['schema_version']==1
            and type(result['schema_version']) is int and result['schema_version']==1
            and digest(plan['jobs'])==digest(jobs), 'persisted/frozen sequence plan differs')
    require(plan['diagnostic_only'] is True and plan['python_copies_deferred'] is True
            and plan['launch_readiness_verified'] is False
            and number(plan['capture_limit_bytes_per_job'])==CAPTURE_BYTES
            and number(plan['report_limit_bytes_per_job'])==REPORT_BYTES
            and number(plan['minimum_initial_free_bytes'])==MIN_FREE_BYTES
            and finite(plan['work_deadline_monotonic_seconds'])==finite(result['work_deadline_monotonic_seconds'])
            ==work_deadline, 'sequence limits/deadline differ')
    require(result['status']=='complete' and result['stop_reason'] is None and result['external_cancelled'] is False
            and result['diagnostic_only'] is True and result['python_copies_deferred'] is True
            and all(result[k] is False for k in ('launch_provenance_verified','benchmark_outputs_qualified',
                                                'whole_run_supervision_provided'))
            and isinstance(result['jobs'],list) and len(result['jobs'])==18, 'incomplete sequence files')
    reports,bindings,pids=[],[],set()
    for ordinal,(job,row,prefix) in enumerate(zip(jobs,result['jobs'],expected),1):
        require(number(row['ordinal'],1)==ordinal and row['runtime']==job['runtime']
                and number(row['repeat'],1)==job['repeat'], 'persisted sequence order differs')
        receipt=evidence.record(prefix+'.receipt.json',CAPTURE_BYTES)
        observation=evidence.record(prefix+'.process.json')
        require(digest(receipt)==digest(row['receipt']) and digest(observation)==digest(row['observation']),
                'individual/sequence evidence records differ')
        pid=receipt_complete(receipt)
        require(pid not in pids and receipt['argv']==job['argv'] and receipt['cwd']==job['cwd']
                and receipt['process_observer_enabled'] is True, 'duplicate process or command/observer mismatch')
        pids.add(pid)
        capture=evidence.raw(prefix+'.capture',CAPTURE_BYTES)
        capture_id=evidence.inputs[prefix+'.capture']
        require(number(receipt['output_limit_bytes'])==CAPTURE_BYTES
                and number(receipt['captured_bytes'])==number(receipt['observed_bytes'])==len(capture)
                and receipt['capture_sha256']==capture_id['sha256'], 'raw capture differs from receipt')
        raw=evidence.raw(prefix+'.json',REPORT_BYTES); identity=evidence.inputs[prefix+'.json']
        report=receipt['report_capture']
        require(number(receipt['report_limit_bytes'])==REPORT_BYTES and report['path']==job['report']
                and number(report['size_bytes'],1)==len(raw) and report['sha256']==identity['sha256']
                and key(report)==key(identity) and report['post_exit_hash_verified'] is True
                and report['producer_identity_verified'] is False, 'raw report identity differs from receipt')
        reports.append(raw)
        bindings.append({'ordinal':ordinal,'runtime':job['runtime'],'repeat':job['repeat'],'pid':pid,
                         'report':identity.copy(),'capture':capture_id.copy(),
                         'receipt_canonical_sha256':digest(receipt),'observation_canonical_sha256':digest(observation)})
    evidence.verify_unchanged()
    return {'plan':plan,'sequence':result,'raw_reports':reports,
            'binding':{'schema_version':1,'sequence_file_receipt_consistency_verified':True,'jobs':bindings,
                       'launch_provenance_verified':False,'benchmark_outputs_qualified':False,
                       'capture_prelaunch_inode_verified':False,'postflight_call_order_enforced':False,
                       'scope':'Current private bounded sequence files match supplied plan/deadline and embedded '
                               'receipts/observations. Reports retain receipt pathname identities; captures bind '
                               'hash/length only, not producer inode. Not live provenance, atomic snapshots, '
                               'full process/lifecycle/output qualification or finalization ordering.'}}
