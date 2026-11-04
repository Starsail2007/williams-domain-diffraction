"""Batch selection must preserve input identity, real results and failures."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from williams_diffraction.batch_lab import run_batch,validate_tasks
from williams_diffraction.independent_lab import run_experiment
from williams_diffraction.lab_server import Lab,save
from williams_diffraction.delivery_io import read_json

ROOT=Path(__file__).resolve().parents[1]


class BatchSelectionTests(unittest.TestCase):
    def test_batch_matches_independent_numerical_results_and_samples(self):
        tasks=[dict(algorithm='texture.spectrum',sample='micro196_right'),dict(algorithm='texture.family',sample='micro164_left')]
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            base=Path(folder);state=run_batch(ROOT,base/'batch',tasks)
            self.assertEqual(state['status'],'complete')
            for i,task in enumerate(tasks):
                single=run_experiment(ROOT,base/f'single_{i}',task['algorithm'],task['sample'])
                entry=state['tasks'][i]
                self.assertEqual((entry['algorithm'],entry['sample']),(task['algorithm'],task['sample']))
                self.assertEqual(entry['status'],'complete')
                self.assertEqual(entry['events'][-1]['table'],single['events'][-1]['table'])
                self.assertEqual(entry['events'][-1]['metrics'],single['events'][-1]['metrics'])
                self.assertEqual(entry['events'][-1]['step'],task['algorithm'])

    def test_invalid_later_task_prevents_terminal_or_worker_start(self):
        lab=Lab(ROOT)
        with patch('williams_diffraction.terminal_console.open_console') as console,patch('williams_diffraction.lab_server.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'不适用'):
                lab.start(dict(tasks=[dict(algorithm='texture.spectrum',sample='micro196_right'),dict(algorithm='period.horizontal',sample='micro196_right')]))
            console.assert_not_called();spawn.assert_not_called()
        self.assertEqual(lab.jobs,{})

    def test_failure_is_retained_while_other_independent_tasks_complete(self):
        tasks=[dict(algorithm='texture.spectrum',sample='micro196_right'),dict(algorithm='texture.family',sample='micro164_left')]
        def execute(root,out,algorithm,sample,*args,**kwargs):
            if algorithm=='texture.spectrum':raise ValueError('deliberate algorithm failure')
            kwargs['on_state'](dict(status='complete',events=[dict(step=algorithm,status='computed')],version=1))
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            with patch('williams_diffraction.batch_lab.run_experiment',side_effect=execute):state=run_batch(ROOT,Path(folder)/'batch',tasks)
            self.assertEqual(state['status'],'failed')
            self.assertEqual([q['status'] for q in state['tasks']],['failed','complete'])
            self.assertIn('deliberate',state['tasks'][0]['message'])

    def test_cancel_wins_over_complete_exit_race_and_preserves_finished_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'job';(out/'result').mkdir(parents=True)
            save(out/'result/state.json',dict(status='complete',events=[],tasks=[dict(status='complete'),dict(status='running'),dict(status='pending')]))
            lab=object.__new__(Lab);lab.root=root;process=Mock();process.poll.return_value=0
            job=dict(id='test',case='batch',batch=True,tasks=[],folder=out,process=process,cancelled=True)
            state=lab.state(job)
            self.assertEqual(state['status'],'cancelled')
            self.assertEqual([q['status'] for q in state['tasks']],['complete','cancelled','cancelled'])

    def test_task_validation_copies_request_and_rejects_duplicate_algorithms(self):
        tasks=[dict(algorithm='texture.spectrum',sample='micro196_right')]
        clean=validate_tasks(ROOT,tasks);tasks[0]['sample']='micro164_left'
        self.assertEqual(clean[0]['sample'],'micro196_right')
        with self.assertRaisesRegex(ValueError,'只能选择一次'):validate_tasks(ROOT,clean+clean)

    def test_worker_crash_does_not_leave_tasks_appearing_to_run_forever(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'job';(out/'result').mkdir(parents=True)
            save(out/'result/state.json',dict(status='running',events=[],tasks=[dict(status='complete'),dict(status='running'),dict(status='pending')]))
            lab=object.__new__(Lab);lab.root=root;process=Mock();process.poll.return_value=-9
            job=dict(id='test',case='batch',batch=True,tasks=[],folder=out,process=process)
            state=lab.state(job)
            self.assertEqual(state['status'],'failed')
            self.assertEqual([q['status'] for q in state['tasks']],['complete','failed','cancelled'])


if __name__=='__main__':unittest.main()
