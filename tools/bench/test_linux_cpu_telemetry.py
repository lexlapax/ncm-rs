import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sample_linux_cpu import activity, cpu_list, parse_stat, run

STAT = ('cpu 50 0 10 30 10 0 0 0 40 0\n'
        'cpu2 25 0 5 15 5 0 0 0 20 0\n'
        'cpu3 0 0 0 0 0 0 0 0 0 0\n'
        'cpu12 25 0 5 15 5 0 0 0 20 0\n'
        'ctxt 123\nprocs_running 1\nprocs_blocked 0\n')


class CpuTelemetryTests(unittest.TestCase):
    def fixture(self, root):
        proc=root/'proc';proc.mkdir();(proc/'stat').write_text(STAT)
        sys=root/'sys';topology=sys/'cpu2/topology';topology.mkdir(parents=True)
        (topology/'thread_siblings_list').write_text('2,12\n')
        return proc,sys

    def test_cpu_ranges_and_invalid_lists(self):
        self.assertEqual(cpu_list('0-2,5,2\n'),[0,1,2,5])
        for text in ('','-1','2-1','1-3-5','a','65536'):
            with self.assertRaises(ValueError):cpu_list(text)

    def test_guest_is_not_counted_twice(self):
        ticks,counters=parse_stat(STAT)
        zero={cpu:{key:0 for key in row} for cpu,row in ticks.items()}
        result=activity(zero,ticks)
        self.assertEqual(result['cpu']['total_ticks'],100)
        self.assertEqual(result['cpu']['non_idle_non_iowait_percent'],60)
        self.assertFalse(result['cpu3']['available'])
        self.assertEqual(counters['ctxt'],123)
        self.assertFalse(activity(ticks,zero)['cpu']['available'])

    def test_incomplete_or_invalid_stat_is_rejected(self):
        for text in (STAT+'cpu 0 0 0 0 0 0 0 0\n',STAT.replace('ctxt 123\n',''),
                     STAT.replace('cpu2 25','cpu2 -25'),STAT.replace('cpu 50 0 10 30 10 0 0 0 40 0','cpu 1 2')):
            with self.assertRaises(ValueError):parse_stat(text)

    @patch('sample_linux_cpu.platform.system',return_value='Linux')
    @patch('sample_linux_cpu.os.sched_getaffinity',return_value={2,3,12})
    @patch('sample_linux_cpu.os.sched_setaffinity')
    def test_complete_capture_and_unavailable_optional_counters(self,set_affinity,*_):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proc,sys=self.fixture(root);out=root/'sample.jsonl'
            run(out,2,seconds=.1,interval=.1,observer_cpu=3,proc=proc,sys_cpu=sys)
            rows=[json.loads(s) for s in out.read_text().splitlines()]
            self.assertEqual(rows[0]['smt_siblings'],[2,12]);self.assertTrue(rows[0]['diagnostic_only'])
            self.assertEqual(rows[-1]['status'],'complete');self.assertEqual(rows[-1]['samples'],2)
            self.assertFalse(rows[1]['frequency_khz']['2']['available'])
            self.assertFalse(rows[1]['pressure']['cpu']['available'])
            self.assertGreaterEqual(rows[1]['monotonic_end_ns'],rows[1]['monotonic_start_ns'])
            set_affinity.assert_called_once_with(0,{3})
            with self.assertRaises(FileExistsError):run(out,2,seconds=.1,interval=.1,proc=proc,sys_cpu=sys)

    @patch('sample_linux_cpu.platform.system',return_value='Linux')
    @patch('sample_linux_cpu.os.sched_getaffinity',return_value={2,3,12})
    def test_rejects_wrong_cpu_sibling_and_invalid_duration(self,*_):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proc,sys=self.fixture(root)
            for args in ({'cpu':1},{'cpu':2,'observer_cpu':12},{'cpu':2,'seconds':float('nan')},
                         {'cpu':2,'seconds':float('inf')},{'cpu':2,'interval':0},{'cpu':2,'seconds':3601}):
                with self.assertRaises(ValueError):run(root/'unused.jsonl',proc=proc,sys_cpu=sys,**args)
            self.assertFalse((root/'unused.jsonl').exists())

    @patch('sample_linux_cpu.platform.system',return_value='Linux')
    @patch('sample_linux_cpu.os.sched_getaffinity',return_value={2,3,12})
    def test_failed_read_leaves_incomplete_capture(self,*_):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proc,sys=self.fixture(root);(proc/'stat').unlink();out=root/'failed.jsonl'
            with self.assertRaises(FileNotFoundError):run(out,2,seconds=.1,interval=.1,proc=proc,sys_cpu=sys)
            rows=[json.loads(s) for s in out.read_text().splitlines()]
            self.assertEqual([r['kind'] for r in rows],['metadata'])


if __name__=='__main__':unittest.main()
