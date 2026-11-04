import contextlib
import io
import unittest
import numpy as np
from williams_diffraction.runtime_trace import console_values, log_values
from williams_diffraction.guide_events import emit, observe
from williams_diffraction.diffraction_models import fit_one_d, one_d_prediction


class NumericTraceTests(unittest.TestCase):
    def test_trace_uses_actual_values_and_cannot_mutate_array(self):
        values=np.arange(120.,dtype=float).reshape(10,12)
        before=values.copy();out=io.StringIO()
        with contextlib.redirect_stdout(out),console_values():
            log_values('measurement',raw=values,threshold=17.25)
        np.testing.assert_array_equal(values,before)
        self.assertIn('shape=(10, 12)',out.getvalue())
        self.assertIn('min=0, max=119',out.getvalue())
        self.assertIn('threshold=17.25',out.getvalue())

    def test_trace_is_not_filtered_by_html_selection(self):
        out=io.StringIO();received=[]
        with contextlib.redirect_stdout(out),console_values(),observe(lambda *x:received.append(x),lambda *x:False):
            emit('photometry.channel',peak='other_peak',channel='G2',array=np.array([1.,2.,3.]))
        self.assertEqual(received,[])
        self.assertIn('other_peak',out.getvalue())
        self.assertIn('channel=G2',out.getvalue())
        self.assertIn('values=[1. 2. 3.]',out.getvalue())

    def test_optimization_parameters_are_identical_with_numeric_prints(self):
        z=np.linspace(0,1,7)
        observed=one_d_prediction(np.array([1.8,2.5,.3]),z)
        expected=fit_one_d(z,observed)
        output=io.StringIO()
        with contextlib.redirect_stdout(output),console_values():
            actual=fit_one_d(z,observed)
        np.testing.assert_array_equal(actual[0],expected[0])
        np.testing.assert_array_equal(actual[1],expected[1])
        self.assertIn('one_d_residual_evaluation',output.getvalue())
        self.assertIn('Cost',output.getvalue())
        self.assertIn('one_d_best',output.getvalue())
