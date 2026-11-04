"""Contracts for independent algorithms and scientifically distinct samples."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from williams_diffraction.independent_lab import run_experiment,select_definition,one_d_observations,discussion_experiment,intensity_experiment
from williams_diffraction.delivery_io import read_json

ROOT=Path(__file__).resolve().parents[1]


class IndependentAlgorithmTests(unittest.TestCase):
    def test_every_algorithm_has_multiple_compatible_samples(self):
        config=read_json(ROOT/'configs/experiments.json')
        self.assertEqual(len({q['id'] for q in config['tools']}),len(config['tools']))
        for tool in config['tools']:
            with self.subTest(algorithm=tool['id']):
                self.assertGreaterEqual(len(tool['samples']),2)
                self.assertEqual(len({q['id'] for q in tool['samples']}),len(tool['samples']))
                self.assertTrue(tool['sources'])
                for sample in tool['samples']:
                    for image in sample['input']['images']:
                        self.assertTrue(image.startswith('data/'))
                        self.assertTrue((ROOT/image).is_file())

    def test_incompatible_sample_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'result'
            with self.assertRaisesRegex(ValueError,'不适用'):
                run_experiment(ROOT,out,'detection.guided','micro196_right')
            self.assertFalse(out.exists())

    def test_repeated_model_rows_are_one_observation_per_frame(self):
        for sample in ('cycle01/up','cycle01/down','cycle02/up','cycle02/down'):
            ids,U,y=one_d_observations(ROOT,dict(id=sample))
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(y.shape,(len(ids),2))
            self.assertTrue(np.isfinite(y).all())
            self.assertEqual(len(U),len(ids))

    def test_fft_runs_from_clean_state_without_prior_stage_or_downstream_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'spectrum'
            # Direct FFT may prepare detrending/windowing, but it must not need
            # a previous run or execute the later family/profile algorithms.
            with patch('williams_diffraction.delivery_algorithms.select_texture_family',side_effect=AssertionError('downstream family called')):
                result=run_experiment(ROOT,out,'texture.spectrum','micro196_right')
            self.assertEqual(result['status'],'PASS')
            self.assertEqual([q['step'] for q in result['events']],['texture.spectrum'])
            state=read_json(out/'state.json')
            self.assertEqual(state['status'],'complete')
            self.assertNotEqual(state['status'],'waiting')
            self.assertTrue(result['events'][0]['images'])

    def test_reference_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'result'
            with patch('williams_diffraction.delivery_algorithms.diffraction_case',return_value=dict(status='FAIL')):
                with self.assertRaisesRegex(ValueError,'验收参考不一致'):
                    run_experiment(ROOT,out,'detection.guided','200')
            self.assertEqual(read_json(out/'state.json')['status'],'failed')
            self.assertFalse((out/'run.json').exists())

    def test_rejected_basis_does_not_generate_supported_combination_lengths(self):
        from unittest.mock import Mock
        rejected=dict(reciprocal_basis_candidate=dict(first={},second={}),basis_passes_global_screen=False)
        recorder=Mock()
        with patch('williams_diffraction.delivery_algorithms.analyze_texture',return_value=(rejected,None,None,None,None)):
            discussion_experiment(ROOT,ROOT/'work/unused',dict(id='discussion.combination'),dict(id='micro164_left'),recorder)
        args=recorder.result.call_args.kwargs
        self.assertNotIn('table',args)
        self.assertIn('没有合格',str(args))

    def test_single_frame_does_not_identify_a_voltage_law_or_holdout_score(self):
        for algorithm in ('intensity.single_sine','intensity.double_harmonic','intensity.comparison'):
            with patch('williams_diffraction.diffraction_models.fit_one_d') as fit:
                with self.assertRaisesRegex(ValueError,'不足三个'):
                    intensity_experiment(ROOT,ROOT/'work/unused',dict(id=algorithm),dict(id='cycle01/down'),None)
                fit.assert_not_called()
            tool=next(q for q in read_json(ROOT/'configs/experiments.json')['tools'] if q['id']==algorithm)
            self.assertNotIn('cycle01/down',[q['id'] for q in tool['samples']])


if __name__=='__main__':unittest.main()
