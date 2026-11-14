"""CPU-only root-supervisor/ordinary-user fake transaction; no GPU or network.

Normal-user invocation is only a no-drop self-test. Via sudo, exact pinned
helpers are copied into a private root directory before use. This is not an entry
or a GPU-restoration proof; it uses a fake file as policy state.
"""
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat
import sys
import tempfile

if sys.flags.isolated != 1 or sys.flags.no_site != 1:
    raise SystemExit('Use python3 -I -S')
root_mode = os.geteuid() == 0
if root_mode:
    u, g = os.environ.get('SUDO_UID', ''), os.environ.get('SUDO_GID', '')
    if not u.isdecimal() or not g.isdecimal():
        raise SystemExit('Non-root sudo invoking identity required')
    uid, gid = int(u), int(g)
    account = pwd.getpwuid(uid)
    if uid <= 0 or gid <= 0 or account.pw_gid != gid:
        raise SystemExit('Unexpected invoking identity')
    groups = os.getgrouplist(account.pw_name, gid)
else:
    uid, gid, groups = os.getuid(), os.getgid(), os.getgroups()
    if uid <= 0 or gid <= 0:
        raise SystemExit('Non-root self-test identity required')

source = Path('/home/lexlapax/projects/lexlapax/ncm-rs/tools/bench')
pins = {
    'supervised_restoration.py':'125a82945473d7908944b90444bf40e44fb0cb418f5d6af0d849975d034a767d',
    'bounded_child.py':'2e39b2b7871cf29bf7dfa334bb4d041f5b6a452bdab4877fb979c7ef217ca076',
    'supervised_worker.py':'448f2897a39309be3f319e4518ccfdd660f17f4844876ac82515696f2f6cbcc1',
    'diagnostic_clock.py':'5906f0a6c20f9ac37b6ca3d1c8f4b131b0f26e7e2cba5f9510c8c5553997d789',
    'clock_drop_privileges.py':'cc52eb3a615aa5a55df8d54a49ffad622f41c3b5fc2c03d7342b43bf571d5cee',
    'clock_unprivileged_identity.py':'4b97d49d520949ea691e974b00476e3583546808c5a9b5c1ef0a33b0de0687e1',
}
os.umask(0o077)
root = Path(tempfile.mkdtemp(prefix='ncm-supervised-user-fake-', dir='/tmp'))
for name, expected in pins.items():
    fd = os.open(source/name, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        before=os.fstat(file.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>1024**2:
            raise SystemExit('Invalid pinned helper')
        raw=file.read(1024**2+1); after=os.fstat(file.fileno())
    if (len(raw)!=before.st_size or hashlib.sha256(raw).hexdigest()!=expected
            or any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))):
        raise SystemExit('Pinned helper mismatch: '+name)
    (root/name).write_bytes(raw)
sys.path.insert(0,str(root))
from supervised_restoration import supervise_restoration
from supervised_worker import direct_children, subreaper

policy=root/'fake-policy.py'
policy.write_text('''from pathlib import Path\nimport sys\np=Path(sys.argv[1]); kind=sys.argv[2]\nwith (p/'events').open('a') as f: f.write(kind+'\\n')\nif kind=='set': (p/'fake-lock').write_text('fake')\nelse: (p/'fake-lock').unlink()\n''')
worker='''import json,os,sys,time\nsys.path.insert(0,sys.argv[1])\nfrom clock_drop_privileges import drop_and_verify\nfrom clock_unprivileged_identity import verify\nuid=int(sys.argv[2]);gid=int(sys.argv[3]);groups=json.loads(sys.argv[4]);root_mode=sys.argv[5]=='root'\nif root_mode: drop_and_verify(uid,gid,groups)\nr=verify(uid,gid,groups)\ns=dict(line.split(':',1) for line in open('/proc/self/status') if ':' in line)\nresult={'uid_match':os.getuid()==uid,'gid_match':os.getgid()==gid,'groups_match':sorted(os.getgroups())==sorted(groups),'empty_capabilities':all(int(s[k].strip(),16)==0 for k in ('CapInh','CapPrm','CapEff','CapAmb')),'no_new_privs':s['NoNewPrivs'].strip()=='1','identity_sampled':r['sampled_unprivileged_identity_verified']}\nprint(json.dumps(result,sort_keys=True),flush=True)\ntime.sleep(float(sys.argv[6]))\n'''
python=str(Path('/usr/bin/python3').resolve(strict=True))
env={'PATH':os.defpath,'HOME':str(root),'LANG':'C','LC_ALL':'C','PYTHONNOUSERSITE':'1'}
results={}; failures=[]
for name,seconds in (('success',0.1),('deadline',10)):
    case=root/name; case.mkdir(mode=0o700)
    command=[python,'-I','-S',str(policy),str(case)]
    argv=[python,'-I','-S','-c',worker,str(root),str(uid),str(gid),json.dumps(groups),
          'root' if root_mode else 'self',str(seconds)]
    result=supervise_restoration(command+['set'],argv,command+['reset'],cwd='/tmp',env=env,
        output_dir=case/'restoration',parent_pid=os.getppid(),lock_seconds=4,
        command_seconds=1.5,cleanup_reserve_seconds=1)
    try:
        sample=json.loads((case/'restoration/worker.txt').read_bytes())
    except (OSError,ValueError): sample={}
    receipt=result.get('worker') or {}
    events=(case/'events').read_text().splitlines() if (case/'events').exists() else []
    checks={'expected_status':result['status']==('complete' if name=='success' else 'failed'),
            'fake_set_reset_once':events==['set','reset'],
            'fake_lock_absent':not (case/'fake-lock').exists(),
            'reset_succeeded':result['reset_command_succeeded'],
            'reset_request_within_deadline':result['reset_request_within_deadline'],
            'all_owned_cleanup_complete':all(row['result']['complete'] for row in result['cleanups']),
            'subreaper_restored':result['subreaper_restored'] and subreaper()==0 and direct_children()==[],
            'worker_reaped':receipt.get('direct_child_reaped') is True,
            'worker_capture_eof':receipt.get('capture_eof') is True,
            'worker_identity':all(sample.get(k) is True for k in
                ('uid_match','gid_match','groups_match','empty_capabilities','identity_sampled')),
            'no_new_privs_if_privileged':(not root_mode or sample.get('no_new_privs') is True),
            'deadline_termination_if_applicable':(name!='deadline' or (receipt.get('status')=='failed'
                 and 'sequence_deadline' in receipt.get('issues',[])))}
    results[name]={'transaction_status':result['status'],'worker_status':receipt.get('status'),
                   'worker_issues':receipt.get('issues'),'checks':checks}
    failures.extend(name+':'+key for key,value in checks.items() if value is not True)
report={'schema_version':1,'mode':'root' if root_mode else 'unprivileged-self-test',
        'supervisor_euid':os.geteuid(),'cases':results,'failures':failures,
        'gpu_or_real_policy_command_executed':False,'privileged_coordinator_wiring_verified':False,
        'hardware_restoration_verified':False,'launch_ready':False}
output=Path('/tmp/ncm-root-supervised-user-v2-'+str(os.geteuid())+'.json')
fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as file:
    json.dump(report,file,indent=2,sort_keys=True); file.write('\n'); file.flush(); os.fchmod(file.fileno(),0o644)
print(json.dumps(report,sort_keys=True),flush=True)
if failures: raise SystemExit(1)
