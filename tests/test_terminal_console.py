import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from williams_diffraction.lab_server import Lab
from williams_diffraction.terminal_console import run_console, write_json


class TerminalConsoleTests(unittest.TestCase):
    def test_no_calculation_starts_if_console_cannot_open(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'configs').mkdir()
            (root/'configs/experiments.json').write_text(json.dumps(dict(tools=[
                dict(id='probe',title='Probe',type='detection',samples=[dict(id='a',label='A')])
            ])))
            lab=Lab(root)
            with patch('williams_diffraction.terminal_console.open_console',side_effect=RuntimeError('No terminal')),patch('williams_diffraction.lab_server.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'计算尚未启动'):
                    lab.start(dict(algorithm='probe',sample='a'))
                spawn.assert_not_called()
            self.assertEqual(lab.jobs,{})

    def test_console_reports_real_stage_and_html_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job=root/'work/lab_runs/probe';(job/'result').mkdir(parents=True)
            write_json(job/'console.json',dict(title='Actual example',steps=[dict(id='test.input',title='读取输入')],url='http://127.0.0.1:1234/#case/test/0',output='work/lab_runs/probe/result',log='worker.log'))
            write_json(job/'result/state.json',dict(status='complete',events=[dict(step='test.input')]))
            (job/'worker.log').write_text('[VAR] H=[[1,0,0],[0,1,0],[0,0,1]]; shape=(3,3)\n实际中间值=12.5\n',encoding='utf-8')
            write_json(job/'console_status.json',dict(status='complete'))
            output=io.StringIO()
            with patch('sys.stdout',output),patch('sys.stdin.isatty',return_value=False):
                run_console(root,job)
            self.assertTrue((job/'console_ready.json').exists())
            self.assertIn('H=[[1,0,0],[0,1,0],[0,0,1]]',output.getvalue())
            self.assertIn('实际中间值=12.5',output.getvalue())
            self.assertNotIn('本步完成',output.getvalue())
            self.assertIn('已完成。HTML 页面中已经有本次计算结果',output.getvalue())

    def test_cancelled_console_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job=root/'work/lab_runs/probe';job.mkdir(parents=True)
            write_json(job/'console.json',dict(title='Stopped example',steps=[],url='',output='',log='worker.log'))
            write_json(job/'console_status.json',dict(status='cancelled'))
            output=io.StringIO()
            with patch('sys.stdout',output),patch('sys.stdin.isatty',return_value=False):
                run_console(root,job)
            self.assertIn('本次计算已停止',output.getvalue())
            self.assertNotIn('已完成。HTML',output.getvalue())
