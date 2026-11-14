"""CPU-only fake real-handoff/drop/evidence suite; never accesses a GPU.

Uses hash-pinned local helper bytes in a private directory, with fake set/reset
files only. Does not authenticate a production entry or prove hardware policy.
"""
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys
import tempfile

if not sys.flags.isolated or not sys.flags.no_site:
    raise SystemExit('Use python3 -I -S')
root_mode=os.geteuid()==0
if root_mode:
    u,g=os.environ.get('SUDO_UID',''),os.environ.get('SUDO_GID','')
    if not u.isdecimal() or not g.isdecimal(): raise SystemExit('Invoking sudo identity required')
    uid,gid=int(u),int(g); account=pwd.getpwuid(uid)
    if uid<=0 or gid<=0 or account.pw_gid!=gid: raise SystemExit('Invalid invoking identity')
    groups=os.getgrouplist(account.pw_name,gid)
else:
    uid,gid,groups=os.getuid(),os.getgid(),os.getgroups()
    if uid<=0 or gid<=0: raise SystemExit('Unprivileged self-test required')
if 0 not in os.sched_getaffinity(0): raise SystemExit('CPU0 must be available for fake handoff')
source=Path('/home/lexlapax/projects/lexlapax/ncm-rs/tools/bench')
pins={'analyze_readonly_trace.py': '29cb7f6ce54fa498c7a6f2490bffa244965f961d8282d7ae3847122fb35735e0', 'bind_cupti_trials.py': '1b4026a7472df31da4196f0be5470ebaef38d2d3c27afa5a261ba27a5c999349', 'bounded_child.py': '2e39b2b7871cf29bf7dfa334bb4d041f5b6a452bdab4877fb979c7ef217ca076', 'clock_drop_privileges.py': 'cc52eb3a615aa5a55df8d54a49ffad622f41c3b5fc2c03d7342b43bf571d5cee', 'clock_unprivileged_identity.py': '4b97d49d520949ea691e974b00476e3583546808c5a9b5c1ef0a33b0de0687e1', 'clock_worker_handoff.py': '437dbd051cdd4e6f41af1f66992cad8d2876e0218f7c00e81627715b16349a41', 'clock_worker_identity_evidence.py': '6f62f38d02c56aebe137de3e7c23da5b82e7262137a2448579d7dc5db8e60219', 'cupti_kernel_windows.py': 'cf864ae5e25a6e444e35fee6434e8ed0714ce7e9852b8c55de155ea77c9b882f', 'cupti_trace_boundaries.py': '10e0da72eb62acd530ccf808b88c3d4c9a52a81f7267e9f6f3298e9ee02d4e88', 'diagnostic_clock.py': '5906f0a6c20f9ac37b6ca3d1c8f4b131b0f26e7e2cba5f9510c8c5553997d789', 'finalize_readonly_trace.py': 'e1c1cf0acc8db60b55d5c9f6ea1893c6909a38ffa13bc27af2c83c3e76743a7e', 'observe_readonly_process.py': '0dc960cf4dbe096a256e1e193e1029281ce9e086888bd9358bdb875a258eae5e', 'phase2_summary.py': '3f72dd17e3f40e42d147f6c68eb99dfc2f94f28c3cb663035ea293339b406b59', 'prepare_readonly_launch.py': '51a74b567cf0b7f3fb5d03e9e9743f404fe3c2cc12bf01776453ecc4e776192c', 'qualify_diagnostic_codec.py': '439733ef2f50855976ffbc868d1fdd3def97bbd5a6071431e75430c74efaf7ca', 'readonly_trace_preflight.py': '2f481f469fd39790d8dad6eb66c926e35adb38ea51ecbdae76e8a9f743dd1e86', 'readonly_trace_runtime.py': 'd067e7ce52f40a3dc4245904144bc4163301f802acc73b384784f14cb31dc321', 'readonly_trace_sequence.py': 'c5a7043b1b5d148ad3d2e4006c7c88a51d904bac5b2d2ea76069c8eb5d5b23c6', 'readonly_trace_telemetry.py': '79f5d94fa6055a2e14d49031d9d0ba9020b7851abbfa286e4c7d8faf03ff10c2', 'run_phase2_linux_comparison.py': '6b3a8f8bbbfbbc6a77e76bea8d1c8191b94644918b359ff5659aac16e046a375', 'sample_linux_cpu.py': '5cddef515c31008328c4a3315910b5d3da9a70f80568a9bc65f695d9d1b41569', 'supervised_restoration.py': '125a82945473d7908944b90444bf40e44fb0cb418f5d6af0d849975d034a767d', 'supervised_worker.py': '448f2897a39309be3f319e4518ccfdd660f17f4844876ac82515696f2f6cbcc1', 'validate_clock_worker_identity.py': 'abd8ab81b6564857e1969c12a9629da673a0648e04c18a3db242cf4ae6aa83d3', 'validate_readonly_lifecycle.py': 'e2e98745d42d65e9d337c06f3101c5cc870e62b0de729140415dfb3934aae5a5', 'validate_readonly_telemetry.py': '09553852dd0e4695c72bd0973013fb7fa3ed58d802958f4377515a67ba7e190a', 'verify_readonly_processes.py': 'dfebee7de700e6998550a341291ee92793b92806770ac9a8b9213dc89cb7d44c'}
os.umask(0o077)
bundle=Path(tempfile.mkdtemp(prefix='ncm-handoff-bundle-',dir='/tmp'))
for name,expected in pins.items():
    fd=os.open(source/name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>1024**2: raise SystemExit('Invalid helper')
        raw=stream.read(1024**2+1);after=os.fstat(stream.fileno())
    if (len(raw)!=before.st_size or hashlib.sha256(raw).hexdigest()!=expected
            or any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))):
        raise SystemExit('Pinned helper mismatch: '+name)
    (bundle/name).write_bytes(raw)
user_base=Path(tempfile.mkdtemp(prefix='ncm-handoff-user-',dir='/tmp'))
if root_mode: os.chown(user_base,uid,gid)
python=str(Path('/usr/bin/python3').resolve(strict=True)); taskset=str(Path('/usr/bin/taskset').resolve(strict=True))
env={'PATH':os.defpath,'HOME':str(user_base),'LANG':'C','LC_ALL':'C',
     'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'}
policy=bundle/'fake-policy.py'
policy.write_text('''from pathlib import Path\nimport sys\np=Path(sys.argv[1]); kind=sys.argv[2]\nwith (p/'events').open('a') as f: f.write(kind+'\\n')\nif kind=='set': (p/'fake-lock').write_text('fake')\nelse: (p/'fake-lock').unlink()\n''')
worker_code='''import json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from clock_worker_handoff import validate_handoff
from clock_drop_privileges import drop_and_verify
from clock_worker_identity_evidence import write
import diagnostic_clock
p=Path(sys.argv[2]); d=json.loads(p.read_bytes())
commands=d['commands']
if d['mode']=='reject': commands['set'].append('--unexpected')
try:
    deadline=validate_handoff(Path(d['obligation']),parent_pid=os.getppid(),parent_argv=d['parent_argv'],
        commands=commands,role_image=d['image'],taskset_path=d['taskset'],cwd='/tmp',env=d['env'])
except ValueError:
    if d['mode']=='reject':
        print(json.dumps({'rejected':True}),flush=True);raise SystemExit(1)
    raise
if d['mode']=='reject':raise SystemExit('tampered handoff accepted')
if d['root_mode']:drop_and_verify(d['uid'],d['gid'],d['groups'])
try: Path(d['obligation']).read_bytes(); denied=False
except PermissionError: denied=True
record=write(Path(d['user_dir']),d['uid'],d['gid'],d['groups'],require_no_new_privs=d['root_mode'])
life=diagnostic_clock.begin()
sequence={'schema_version':1,'lifetime':diagnostic_clock.finish(life)}
print(json.dumps({'deadline':deadline,'root_obligation_denied_if_privileged':not d['root_mode'] or denied,
    'sampled_identity':record['sampled_unprivileged_identity_verified'],'sequence':sequence},sort_keys=True),flush=True)
'''
supervisor_code='''import json,os,sys
from pathlib import Path
p=json.loads(sys.argv[1]);sys.path.insert(0,p['bundle'])
from supervised_restoration import supervise_restoration
from supervised_worker import direct_children,subreaper
from validate_clock_worker_identity import validate
parent_argv=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\\0') if v]
case=Path(p['case']); handoff=case/'handoff.json'; obligation=case/'restoration/obligation.json'
policy=[p['python'],'-I','-S',p['policy'],str(case)]
worker=[p['taskset'],'-c','0',p['python'],'-I','-S','-c',p['worker_code'],p['bundle'],str(handoff)]
commands={'set':policy+['set'],'worker':worker,'reset':policy+['reset']}
st=os.stat('/proc/self/exe')
data={'commands':commands,'parent_argv':parent_argv,'image':{'device':st.st_dev,'inode':st.st_ino},
      'obligation':str(obligation),'taskset':p['taskset'],'env':p['env'],'mode':p['mode'],
      'root_mode':p['root_mode'],'uid':p['uid'],'gid':p['gid'],'groups':p['groups'],'user_dir':p['user_dir']}
handoff.write_text(json.dumps(data))
r=supervise_restoration(commands['set'],commands['worker'],commands['reset'],cwd='/tmp',env=p['env'],
    output_dir=case/'restoration',parent_pid=os.getppid())
try: sample=json.loads((case/'restoration/worker.txt').read_bytes())
except (ValueError,OSError): sample={}
events=(case/'events').read_text().splitlines() if (case/'events').exists() else []
evidence=Path(p['user_dir'])/'worker-identity.json'
receipt=r.get('worker') or {}
checks={'expected_status':r['status']==('complete' if p['mode']=='success' else 'failed'),
        'fake_set_reset_once':events==['set','reset'],'fake_lock_absent':not (case/'fake-lock').exists(),
        'reset_succeeded':r['reset_command_succeeded'],'reset_within_deadline':r['reset_request_within_deadline'],
        'all_owned_cleanup_complete':all(item['result']['complete'] for item in r['cleanups']),
        'worker_reaped':receipt.get('direct_child_reaped') is True,
        'subreaper_restored':r['subreaper_restored'] and subreaper()==0 and direct_children()==[]}
if p['mode']=='success':
    try:
        rec=json.loads(evidence.read_bytes()); st=evidence.stat()
        expected={'uid':p['uid'],'gid':p['gid'],'groups':p['groups'],'require_no_new_privs':p['root_mode']}
        valid=validate(rec,expected,receipt,sample['sequence'])
        wrong=dict(expected,uid=p['uid']+1)
        try:validate(rec,wrong,receipt,sample['sequence']); reject=False
        except ValueError:reject=True
        checks.update(real_handoff_and_original_deadline=sample['deadline']==r['work_deadline_monotonic_seconds'],
            worker_identity_sampled=sample['sampled_identity'] is True,
            root_obligation_denied_if_privileged=sample['root_obligation_denied_if_privileged'] is True,
            user_owned_private_identity=st.st_uid==p['uid'] and st.st_mode&0o777==0o600,
            identity_receipt_consistent=valid['worker_identity_record_consistent'] is True,
            wrong_identity_rejected=reject)
    except (OSError,ValueError,KeyError,TypeError):checks['identity_validation_completed']=False
else:
    checks.update(tampered_handoff_rejected=sample.get('rejected') is True,
                  no_user_identity_record=not evidence.exists())
print(json.dumps({'mode':p['mode'],'status':r['status'],'checks':checks},sort_keys=True),flush=True)
'''
# Test-only subreaper owns only descendants of this fresh fake suite. It is not
# the production coordinator ancestry, and cleans adopted fake processes on error.
sys.path.insert(0,str(bundle))
from supervised_worker import subreaper,direct_children,cleanup_owned
if direct_children() or subreaper()!=0: raise SystemExit('Fresh fake-test process required')
results={}; failures=[]; outer_restored=False
subreaper(1)
try:
    for mode in ('success','reject'):
        case=bundle/mode;case.mkdir(mode=0o700)
        user_dir=user_base/mode;user_dir.mkdir(mode=0o700)
        if root_mode:os.chown(user_dir,uid,gid)
        payload=dict(bundle=str(bundle),case=str(case),python=python,taskset=taskset,policy=str(policy),
            worker_code=worker_code,env=env,mode=mode,root_mode=root_mode,uid=uid,gid=gid,groups=groups,user_dir=str(user_dir))
        try:
            run=subprocess.run([taskset,'-c','0',python,'-I','-S','-c',supervisor_code,json.dumps(payload)],
                               cwd='/tmp',env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
            try:result=json.loads(run.stdout)
            except ValueError:result={'mode':mode,'status':'error','checks':{'supervisor_output_parsed':False}}
            if run.returncode!=0:result['checks']['supervisor_exit_zero']=False
        except subprocess.TimeoutExpired:
            result={'mode':mode,'status':'error','checks':{'supervisor_within_timeout':False}}
        cleanup=cleanup_owned()
        result['checks']['outer_cleanup_complete']=cleanup['complete']
        result['checks']['no_outer_orphans']=not cleanup['signalled'] and not cleanup['reaped']
        results[mode]=result
        failures.extend(mode+':'+k for k,v in result['checks'].items() if v is not True)
        if not cleanup['complete']: break
finally:
    if direct_children(): cleanup_owned()
    if not direct_children():
        subreaper(0)
        outer_restored=subreaper()==0
if not outer_restored:failures.append('outer_test_subreaper_not_restored')
report={'schema_version':1,'mode':'root' if root_mode else 'unprivileged-self-test',
    'supervisor_euid':os.geteuid(),'outer_test_subreaper_restored':outer_restored,'cases':results,'failures':failures,
    'gpu_or_real_policy_command_executed':False,'full_entry_authenticated':False,
    'hardware_restoration_verified':False,'launch_ready':False}
output=Path('/tmp/ncm-root-handoff-suite-v2-'+str(os.geteuid())+'.json')
fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fchmod(stream.fileno(),0o644)
print(json.dumps(report,sort_keys=True),flush=True)
if failures:raise SystemExit(1)
