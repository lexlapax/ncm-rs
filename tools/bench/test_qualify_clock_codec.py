"""Archived outputs in a fresh child; synthetic receipts do not prove new processes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch

import qualify_clock_codec as qualification
from clock_process_observation import ORDER
from qualify_diagnostic_codec import EVIDENCE, ROOT, NATIVE_SOURCES, LAYOUT_SOURCES
from prepare_clock_python import FIELDS


def archived_inputs():
    spec=json.loads(EVIDENCE.read_bytes())['archive']; path=ROOT/spec['path']
    raw=path.read_bytes()
    assert len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256']
    members={row['path']:row for row in spec['members']}
    reports,jobs=[],[]
    with tarfile.open(path) as archive:
        for i,runtime in enumerate(ORDER):
            repeat=i//3+1; language='python' if runtime=='python' else 'rust'
            name=f'phase2-linux-diagnostic-v2/{runtime}/codec-{language}-{repeat}.json'
            raw=archive.extractfile(name).read()
            assert len(raw)==members[name]['bytes'] and hashlib.sha256(raw).hexdigest()==members[name]['sha256']
            reports.append(raw)
            capture=dict(path=f'/synthetic/{i+1}.json',size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                         device=1,inode=100+i,post_exit_hash_verified=True,producer_identity_verified=False)
            jobs.append(dict(ordinal=i+1,runtime=runtime,repeat=repeat,receipt=dict(status='complete',issues=[],
                returncode=0,direct_child_reaped=True,capture_eof=True,group_kill='sent',interrupt_signal=None,
                pid=1000+i,report_capture=capture)))
    decoded=[json.loads(raw) for raw in reports]
    sources={r:next(v['source_sha256'] for v,j in zip(decoded,jobs) if j['runtime']==r) for r in ('previous','candidate')}
    python=next(v for v,j in zip(decoded,jobs) if j['runtime']=='python')
    return reports,jobs,sources,{k:python[k] for k in FIELDS}


def archived_cases():
    raw,jobs,sources,metadata=archived_inputs()
    result=qualification.qualify(raw,jobs,sources=sources,python_metadata=metadata)
    assert result['reported_codec_outputs_qualified'] and len(result['jobs'])==18 and len(result['groups'])==4
    assert sum(group['includes_current_python'] for group in result['groups'])==2
    assert all(len(group['quality'])==12 for group in result['groups'])
    for group in result['groups']:
        assert all('runtimes' not in row for row in group['quality'])
        decode=next(row for row in group['quality'] if row['batch']==1 and row['scope']=='decode')
        assert len(decode['quality'])==(9 if group['includes_current_python'] else 6)
    for flag in ('performance_acceptance_evaluated','launch_provenance_verified','independent_processes_verified',
                 'capture_files_verified','operator_trace_parity_verified','postflight_call_order_enforced'):
        assert result[flag] is False
    originals=[json.loads(value) for value in raw]
    references,ref_binding=qualification.frozen_references()
    rejected=0
    def check_bad(reports,records,**changes):
        nonlocal rejected
        args=dict(sources=sources,python_metadata=metadata); args.update(changes)
        try: qualification.qualify(reports,records,**args)
        except ValueError: rejected+=1
        else: raise AssertionError('invalid clock report accepted')
    def encoded(values):
        reports=[json.dumps(value).encode() for value in values]; records=deepcopy(jobs)
        for report,row in zip(reports,records):
            row['receipt']['report_capture'].update(sha256=hashlib.sha256(report).hexdigest(),size_bytes=len(report))
        return reports,records
    # Archive/member binding already ran above. Cache only the offline golden
    # loader for mutation tests, never the actual codec/numerical validators.
    with patch.object(qualification,'frozen_references',return_value=(references,ref_binding)):
        for runtime in ('previous','candidate','python'):
            index=next(i for i,j in enumerate(jobs) if j['runtime']==runtime and j['repeat']==6)
            for mode in ('ids','logits','load','stage','trial','number','schema','profile'):
                values=deepcopy(originals); target=values[index]
                row=next(r for r in target['measurements'] if r['batch']==1 and r['scope']=='decode')
                if mode=='ids': row['output_ids_sha256']='0'*64
                if mode=='logits': row['first_logits'][0]=99.
                if mode=='load': target['load_trials_ms'].pop()
                if mode=='stage': target['instrumented_stages'].pop()
                if mode=='trial': row['cuda_stream_spans_ms'].pop()
                if mode=='number': row['trials_ms'][0]=True
                if mode=='schema': target['schema_version']=True
                if mode=='profile': target['diagnostic_profiling']=True
                check_bad(*encoded(values))
            # A repeated report in the other triple still is not an independent
            # report, even with different JSON whitespace and a unique fake PID.
            values=deepcopy(originals)
            first=next(i for i,j in enumerate(jobs) if j['runtime']==runtime)
            values[index]=deepcopy(values[first]); reports,records=encoded(values)
            reports[index]=json.dumps(values[index],indent=1).encode()
            assert reports[index] != reports[first]
            records[index]['receipt']['report_capture'].update(size_bytes=len(reports[index]),
                sha256=hashlib.sha256(reports[index]).hexdigest())
            check_bad(reports,records)
        for mode in ('order','repeat','pid','path','size','digest','exit','partial'):
            records=deepcopy(jobs); reports=list(raw)
            if mode=='order': records[-1]['runtime']='candidate'
            if mode=='repeat': records[-1]['repeat']=True
            if mode=='pid': records[-1]['receipt']['pid']=1000
            if mode=='path': records[-1]['receipt']['report_capture']['path']=records[0]['receipt']['report_capture']['path']
            if mode=='size': records[-1]['receipt']['report_capture']['size_bytes']+=1
            if mode=='digest': records[-1]['receipt']['report_capture']['sha256']='0'*64
            if mode=='exit': records[-1]['receipt']['returncode']=1
            if mode=='partial': reports.pop(); records.pop()
            check_bad(reports,records)
        for key in ('torch','upstream_commit','reference_versions','source_sha256'):
            values=deepcopy(originals); index=next(i for i,j in enumerate(jobs) if j['runtime']=='python' and j['repeat']==6)
            values[index][key]='changed'; check_bad(*encoded(values))
        wrong=deepcopy(metadata); wrong['torch']='changed'
        values=deepcopy(originals)
        for value,job in zip(values,jobs):
            if job['runtime']=='python': value['torch']='changed'
        check_bad(*encoded(values),python_metadata=wrong)
        # Direct-to-frozen comparison prevents a tolerance chain through Rust.
        baseline=next(r for r in references[0]['measurements'] if r['batch']==1 and r['scope']=='decode')['first_logits'][0]
        budget=5e-4+2e-4*abs(baseline); values=deepcopy(originals)
        for value,job in zip(values,jobs):
            if job['runtime'] in ('previous','python'):
                row=next(r for r in value['measurements'] if r['batch']==1 and r['scope']=='decode')
                row['first_logits'][0]=baseline+budget*(.75 if job['runtime']=='previous' else 1.5)
        check_bad(*encoded(values))
        for mode in ('duplicate_key','nonfinite'):
            reports=list(raw); records=deepcopy(jobs)
            reports[0]=(b'{"schema_version":1,'+raw[0].lstrip()[1:] if mode=='duplicate_key'
                        else b'{"unused":NaN,'+raw[0].lstrip()[1:])
            records[0]['receipt']['report_capture'].update(size_bytes=len(reports[0]),sha256=hashlib.sha256(reports[0]).hexdigest())
            check_bad(reports,records)
    return dict(reports=18,groups=4,scopes_per_group=12,rejected=rejected,acceptance=False)


class ClockCodecQualificationTests(unittest.TestCase):
    def test_all_eighteen_archived_outputs_and_rehashed_negatives_in_fresh_child(self):
        child=subprocess.run([sys.executable,'-c','import json; from test_qualify_clock_codec import archived_cases; '
                              'print(json.dumps(archived_cases()))'],cwd=Path(__file__).resolve().parent,
                             capture_output=True,text=True,timeout=120)
        self.assertEqual(child.returncode,0,child.stdout+child.stderr)
        self.assertEqual(json.loads(child.stdout),dict(reports=18,groups=4,scopes_per_group=12,rejected=43,acceptance=False))

    def test_missing_reports_rejected_before_reference_reads(self):
        with patch.object(qualification,'frozen_references',side_effect=AssertionError('no reference reads')):
            for raw,jobs in (([],[]),([b'{}']*17,[{}]*17),([b'{}']*18,[])):
                with self.subTest(count=len(raw)), self.assertRaises(ValueError):
                    qualification.qualify(raw,jobs,sources={},python_metadata={})

    def test_unreviewed_source_inventory_and_metadata_rejected_before_reference_reads(self):
        sources={p:LAYOUT_SOURCES.get(p,'f'*64) for p in NATIVE_SOURCES}
        for mode in ('missing_runtime','missing_source','digest','layout','python_metadata'):
            declared={'previous':dict(sources),'candidate':dict(sources)}
            if mode=='missing_runtime': declared.pop('candidate')
            if mode=='missing_source': declared['candidate'].pop('Cargo.lock')
            if mode=='digest': declared['candidate']['Cargo.lock']='bad'
            if mode=='layout': declared['candidate'][next(iter(LAYOUT_SOURCES))]='0'*64
            with self.subTest(mode=mode), patch.object(qualification,'frozen_references',side_effect=AssertionError('no reference reads')):
                with self.assertRaises(ValueError): qualification.qualify([b'{}']*18,[{}]*18,sources=declared,python_metadata={})


if __name__ == '__main__': unittest.main()
