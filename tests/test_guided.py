"""The presentation observer must not change science or mix calculation passes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import numpy as np
from williams_diffraction.guide_events import observe, emit
from williams_diffraction.microscopy import spectrum_peaks
from williams_diffraction import evidence_detection
from williams_diffraction.guided_lab import Recorder
from williams_diffraction.lab_server import Lab, save


class GuidedObservationTests(unittest.TestCase):
    def test_callback_cannot_mutate_fft_input(self):
        y,x=np.mgrid[:128,:192];image=30+7*np.cos(2*np.pi*x/16)+3*np.sin(2*np.pi*y/32)
        expected=spectrum_peaks(image,.75,.75)
        stages=[]
        def malicious(name,data):
            stages.append(name)
            for value in data.values():
                if isinstance(value,np.ndarray):value[:]=0
            if 'peaks' in data:data['peaks'].clear()
        with observe(malicious):actual=spectrum_peaks(image,.75,.75)
        self.assertEqual(expected[0],actual[0])
        for a,b in zip(expected[1:],actual[1:]):np.testing.assert_array_equal(a,b)
        self.assertEqual(stages,['texture.detrend','texture.window','texture.spectrum'])

    def test_filter_never_receives_mutable_scientific_values(self):
        source=np.arange(5.);received=[]
        def filter_stage(name,metadata):
            self.assertNotIn('array',metadata);metadata['peak']='changed';return True
        with observe(lambda name,data:received.append(data),filter_stage):emit('sample',array=source,peak='original')
        self.assertEqual(received[0]['peak'],'original');np.testing.assert_array_equal(source,np.arange(5.))

    def test_observer_context_restored_after_failure(self):
        calls=[]
        with self.assertRaisesRegex(ValueError,'deliberate'):
            with observe(lambda name,data:(_ for _ in ()).throw(ValueError('deliberate'))):emit('sample')
        with observe(lambda name,data:calls.append(name)):emit('after')
        self.assertEqual(calls,['after'])

    def test_real_closure_wrapper_records_both_passes(self):
        calls=[];events=[]
        def detector(red,corners,H,cfg,exclusions,green,mask,quarantine):
            calls.append(len(quarantine))
            peak=dict(u_raw_px=1.,v_raw_px=2.,review_flags=['UNRESOLVED_SHALLOW_SADDLE'] if not quarantine else [])
            audit=[dict(status='MODEL_AND_TEMPLATE_TRAINING_INPUTS',eligible_raw_xy=[[1.,2.]],template_raw_xy=[])]
            return [peak],[],audit
        with patch.object(evidence_detection,'_detect_evidence',detector),observe(lambda name,data:events.append((name,data))):
            _,_,audit=evidence_detection.detect_evidence(None,None,None,dict(close_training_evidence=True))
        self.assertEqual(calls,[0,1])
        self.assertEqual([d['pass_index'] for n,d in events if n=='detection.pass'],[1,2])
        self.assertEqual([d['stable'] for n,d in events if n=='detection.closure'],[False,True])
        self.assertEqual(audit[-1]['status'],'TRAINING_EVIDENCE_CLOSURE_STABLE')

    def test_new_detection_pass_supersedes_old_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';out.mkdir()
            recorder=Recorder(root,out,dict(id='probe',type='detection',steps=[]))
            recorder.events=[dict(step=s,status='computed',pass_index=1) for s in ('detection.input','detection.geometry','detection.strong','detection.models','detection.guided')]
            recorder.record('detection.pass',dict(pass_index=2))
            self.assertEqual([q['status'] for q in recorder.events],['computed','computed','superseded','superseded','superseded'])

    def test_ui_completion_waits_for_worker_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'job';(folder/'result').mkdir(parents=True)
            save(folder/'result/state.json',dict(case='probe',status='complete',events=[]))
            lab=object.__new__(Lab);lab.root=root
            process=Mock();process.poll.return_value=None
            job=dict(id='probe-job',case='probe',folder=folder,process=process)
            state=lab.state(job)
            self.assertEqual(state['status'],'running')
            self.assertEqual(state['case'],'probe')
            process.poll.return_value=0
            self.assertEqual(lab.state(job)['status'],'complete')
            process.poll.return_value=7
            self.assertEqual(lab.state(job)['status'],'failed')


if __name__=='__main__':unittest.main()
