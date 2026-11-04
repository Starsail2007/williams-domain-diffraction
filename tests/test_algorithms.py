"""Real-image challenges: harmonic identity and spatial-mixture false positives."""
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from williams_diffraction.delivery_io import read_json
from williams_diffraction.delivery_algorithms import analyze_texture
from williams_diffraction.algorithm_graphics import fft_view

ROOT=Path(__file__).resolve().parents[1]


class TextureChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core=ROOT/'data/core';index=read_json(core/'index.json');cls.results={}
        for item in index['microscopy']:
            if item['id'] not in ('micro164_left','micro164_center','micro196_right','micro254_right','micro436_center'):continue
            with Image.open(core/item['image']) as image:array=np.asarray(image.convert('RGB'))
            x,y,w,h=item['roi_xywh']
            cls.results[item['id']]=analyze_texture(array[y:y+h,x:x+w],*item['pixel_scale_um'])[0]

    def test_strongest_peak_is_harmonic_not_family_base(self):
        q=self.results['micro196_right']
        self.assertAlmostEqual(q['candidates']['peaks'][0]['period_um'],6.158261,places=4)
        self.assertEqual(q['primary_family']['base']['rank'],2)
        self.assertAlmostEqual(q['primary_family']['base']['period_um'],12.292432,places=4)

    def test_global_two_mode_spectrum_does_not_prove_local_coexistence(self):
        mixed=self.results['micro164_center'];grid=self.results['micro164_left']
        self.assertTrue(mixed['basis_passes_global_screen'])
        self.assertGreater(mixed['reciprocal_basis_candidate']['coverage'],.7)
        self.assertEqual(mixed['jointly_supported_tiles'],0)
        self.assertFalse(mixed['spatial_pair_supported'])
        self.assertEqual(grid['jointly_supported_tiles'],3)
        self.assertTrue(grid['spatial_pair_supported'])

    def test_weak_family_identity_stays_ambiguous(self):
        q=self.results['micro254_right']
        self.assertIsNone(q['primary_family'])
        self.assertTrue(q['family_ambiguous'])
        self.assertIsNotNone(q['lower_support_family'])

    def test_no_qualified_pair_is_not_a_zero_of_four_failure(self):
        q=self.results['micro436_center']
        self.assertIsNone(q['primary_family'])
        self.assertFalse(q['spatial_pair_applicable'])
        self.assertIsNone(q['jointly_supported_tiles'])
        self.assertIsNone(q['spatial_pair_supported'])

    def test_unresolved_tiny_field_of_view_has_explicit_display(self):
        q,power,fx,fy,targets=analyze_texture(np.zeros((64,64,3),np.uint8),.0001,.0001)
        self.assertFalse(q['family_passes_quality_gate'])
        self.assertFalse(targets)
        image=fft_view(power,fx,fy,q['candidates']['peaks'],set())
        self.assertGreater(image.width,0)


if __name__=='__main__':unittest.main()
