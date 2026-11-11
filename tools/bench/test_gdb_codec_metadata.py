"""Offline tests for bookkeeping; native ABI/call attribution requires the GDB capture."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOL = Path(__file__).with_name('gdb_codec_metadata.py')


class MetadataTests(unittest.TestCase):
    def setup_capture(self, root, mismatch=False):
        executable = root/'executable'; executable.write_bytes(b'fixture')
        symbols = root/'symbols.json'
        symbols.write_text(json.dumps({'binary_sha256': '0'*64 if mismatch else hashlib.sha256(b'fixture').hexdigest(),
                                      'format':'format', 'encode':'encode', 'block':'block',
                                      'raw_to_demangled':{'caller':'ncm_model::fixture'}}))
        state = SimpleNamespace(tid=1, architecture='i386:x86-64', requested=64, callbacks=[], breakpoints=[], returns=[])
        class Breakpoint:
            def __init__(self, raw, internal=True):
                self.location=raw;state.breakpoints.append(self)
        class FinishBreakpoint:
            def __init__(self, frame, internal=True):state.returns.append(self)
        caller=SimpleNamespace(name=lambda:'caller')
        frame=SimpleNamespace(older=lambda:caller, architecture=lambda:SimpleNamespace(name=lambda:state.architecture))
        fake=SimpleNamespace(Breakpoint=Breakpoint,FinishBreakpoint=FinishBreakpoint,
                             current_progspace=lambda:SimpleNamespace(filename=str(executable)),
                             selected_thread=lambda:SimpleNamespace(global_num=state.tid),
                             newest_frame=lambda:frame,parse_and_eval=lambda expr:state.requested,
                             execute=lambda command:None,events=SimpleNamespace(exited=SimpleNamespace(connect=state.callbacks.append)))
        with patch.dict(sys.modules,{'gdb':fake}), patch.dict(os.environ,{'NCM_METADATA_SYMBOLS':str(symbols),'NCM_METADATA_REPORT':str(root/'report.json')}):
            namespace=runpy.run_path(str(TOOL))
        return namespace,state

    def test_formatter_scope_and_per_invocation_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,s=self.setup_capture(Path(tmp))
            by_name={b.location:b for b in s.breakpoints}
            by_name['encode'].stop();scope_return=s.returns[-1]
            by_name['format'].stop();format_return=s.returns[-1]
            by_name['malloc'].stop()
            s.requested=96;by_name['realloc'].stop()
            format_return.stop();scope_return.stop()
            self.assertEqual(n['formats']['ncm_model::fixture'],{'calls':1,'allocations':1,'reallocations':1,'requested_bytes':160})
            self.assertEqual(n['scope_instances']['encode'][0],{'allocations':1,'reallocations':1,'requested_bytes':160})
            self.assertFalse(any(n['active_formats'].values()))
            self.assertFalse(n['errors'])

    def test_threads_do_not_inherit_other_threads_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,s=self.setup_capture(Path(tmp));b={p.location:p for p in s.breakpoints}
            b['format'].stop();returned=s.returns[-1]
            s.tid=2;b['malloc'].stop();s.tid=1;returned.stop()
            self.assertEqual(n['all_calls']['allocations'],1)
            self.assertEqual(n['formats']['ncm_model::fixture']['allocations'],0)

    def test_failed_scope_and_wrong_architecture_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);n,s=self.setup_capture(root);b={p.location:p for p in s.breakpoints}
            s.architecture='aarch64'
            with self.assertRaises(RuntimeError):b['malloc'].stop()
            s.architecture='i386:x86-64';b['format'].stop();s.returns[-1].out_of_scope()
            s.callbacks[0](SimpleNamespace(exit_code=1))
            result=json.loads((root/'report.json').read_text())
            self.assertFalse(result['all_checks_passed']);self.assertTrue(result['errors'])

    def test_wrong_binary_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):self.setup_capture(Path(tmp),mismatch=True)
            self.assertFalse((Path(tmp)/'report.json').exists())


if __name__=='__main__':unittest.main()
