import unittest
from pathlib import Path
import numpy as np
from scipy.special import jv
from williams_diffraction.diffraction_models import phase_weights, metric, fit_geometry
from williams_diffraction.voltage_spectrum import phase_at, signal_at
from williams_diffraction.delivery_io import verify_bundle

ROOT=Path(__file__).resolve().parents[1]


class IndependentDeliveryTests(unittest.TestCase):
    def test_integrity_and_no_raw_or_history(self):
        manifest=verify_bundle(ROOT, quiet=True)
        for d in ('archive','data/raw','图像分析'):
            self.assertFalse((ROOT/d).exists())
        self.assertFalse(any((ROOT/r['path']).is_symlink() for r in manifest['files']))
        self.assertFalse(any(r['path'].startswith('.venv/') for r in manifest['files']))

    def test_fft_against_independent_bessel_formula(self):
        labels=np.array([[a,b] for a in range(-3,4) for b in range(-3,4)])
        expected=jv(labels[:,0],1.2)**2*jv(labels[:,1],1.8)**2
        np.testing.assert_allclose(phase_weights(labels,1.2,1.8,0,size=128),expected,atol=1e-15)

    def test_signed_loss_does_not_discard_negative_observation(self):
        q=np.array([-2.,1.,4.]);p=np.array([0.,1.,4.])
        self.assertAlmostEqual(metric(q,p)['NRMSE_L2'],2/np.sqrt(21))
        self.assertEqual(metric(q,p)['n_log_cells'],2)

    def test_rank_and_nonpositive_scale_fail(self):
        with self.assertRaises(ValueError):fit_geometry([[0,0],[1,0],[2,0]],[[0,0],[1,0],[2,0]],'rectangular')
        with self.assertRaises(ValueError):signal_at([-1,0],6)
        with self.assertRaises(ValueError):phase_at([1,2,1,2],6,[7,5])


if __name__=='__main__':unittest.main()
