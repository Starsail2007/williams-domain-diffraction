"""Scientific boundary tests for the paper-specific adapter."""
import hashlib
import re
import unittest
from pathlib import Path
from williams_diffraction.delivery_io import read_json, read_csv, new_output
from williams_diffraction.delivery_paper import period_summary

ROOT=Path(__file__).resolve().parents[1]


class PaperBoundaryTests(unittest.TestCase):
    def test_runtime_and_environment_configuration_are_protected_outputs(self):
        for directory in ('.runtime','runtime-bundles','configs'):
            with self.subTest(directory=directory):
                with self.assertRaises(ValueError):new_output(ROOT,ROOT/directory/'must_not_create_test_output')

    def test_aggregate_residual_before_group_median(self):
        rows=[]
        for fid,periods,d in [('A',[1.,2.,100.],1.),('B',[9.],3.)]:
            for p in periods:
                rows.append(dict(batch='test',cycle='test',video_id='micro_late',kind='1d',historical_frame_id=fid,
                                 image_period_um=p,current_optical_period_um=d,correspondence_factor=1.,current_residual=p/d-1))
        q=period_summary(rows)[0]
        self.assertEqual(q['n_historical_frames'],2)
        self.assertEqual(q['n_roi_matches'],4)
        self.assertEqual(q['residual_median'],1.5)
        self.assertNotEqual(q['p_median_um']/q['d_median_um']-1,q['residual_median'])

    def test_current_subset_is_one_video_with_frame_aggregation(self):
        rows=read_json(ROOT/'data/paper/microscopy_matches.json')
        self.assertEqual(len(rows),31)
        self.assertEqual({r['video_id'] for r in rows},{'micro_late'})
        q=period_summary(rows)
        self.assertEqual([(r['cycle'],r['kind'],r['n_historical_frames']) for r in q],
                         [('original','1d',4),('cycle01','1d',5),('cycle02','1d',7),('cycle02','2d',1)])

    def test_reference_figure_index_has_seventeen_entries(self):
        release=read_json(ROOT/'provenance/paper_release.json')
        self.assertEqual([q['number'] for q in release['figures']],list(range(1,18)))
        for figure in release['figures']:
            self.assertTrue((ROOT/figure['preview']).is_file())

    def test_experiment_settings_match_paper_table(self):
        rr=read_csv(ROOT/'data/experiments.csv')
        self.assertEqual([(r['up_frames'],r['down_frames'],r['screen_width_mm'],r['cell_to_screen_mm']) for r in rr],
                         [('46','34','100','120'),('49','38','120','300'),('76','41','120','300')])
        for q in rr:self.assertEqual(int(q['frames']),int(q['up_frames'])+int(q['down_frames']))


if __name__=='__main__':unittest.main()
