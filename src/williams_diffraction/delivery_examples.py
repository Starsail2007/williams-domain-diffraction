"""Deterministic synthetic array examples plus two real decoded microscopy crops.

No ARW/video decoder, historical workspace, camera power calibration or system
font is used. Synthetic truth checks validate algorithms, not the experiment.
"""
import json
import time
import numpy as np
from PIL import Image, ImageDraw
from .delivery_io import read_json, write_json, verify_bundle, new_output, parity_svg


def run_examples(root, output):
    verify_bundle(root, quiet=True)
    out = new_output(root, output)
    start = time.perf_counter()
    from .geometry import homography, transform
    from .screen_audit import refine_screen
    from .evidence_detection import detect_evidence
    from .raw_photometry import measure_frame
    from .microscopy import pixel_scale, spectrum_peaks
    cfg = read_json(root/'examples/config/photometry.json')['photometry']
    det = read_json(root/'examples/config/detector_v16.json')['detector']
    result={'identity':'Synthetic array tests; real microscopy crops are separately identified','checks':{}}
    corners=np.array([[120.,80.],[1500.,220.],[1080.,1500.],[250.,1200.]])
    H=homography(corners,120,120);target=np.array([[0,0],[120,0],[120,120],[0,120]])
    error=float(np.abs(transform(corners,H)-target).max())
    assert error<1e-5
    interior=np.array([[400.,500.],[700.,700.]])
    np.testing.assert_allclose(transform(transform(interior,H),np.linalg.inv(H)),interior,atol=1e-9)
    try:homography([[0,0],[1,0],[2,0],[3,0]],120,120)
    except ValueError:pass
    else:raise AssertionError('Degenerate corners must fail')
    result['checks']['perspective']={'max_corner_error_mm':error,'independent_interior_roundtrip':True,'degenerate_rejected':True}
    gray=np.zeros((320,340));gray[40:280,40:300]=100
    seed=np.array([[40.,40.],[300.,40.],[300.,280.],[40.,280.]])
    reviews=[dict(side=i,physical_edge_identity_visible=True,endpoint_visibility={'start':True,'end':True}) for i in range(4)]
    reviews[1]['endpoint_visibility']['end']='uncertain'
    _,edges=refine_screen(gray,seed,method='multipeak',edge_reviews=reviews)
    assert not edges[1]['supported']
    result['checks']['screen_prior']={'uncertain_endpoint_status':edges[1]['status'],'independent_metric_validation':False}

    y,x=np.mgrid[:400,:400]
    red=200*np.exp(-((x-130.)**2+(y-180.)**2)/72.)+8*np.exp(-((x-250.)**2+(y-180.)**2)/72.)
    board=np.array([[20.,20.],[780.,20.],[780.,780.],[20.,780.]])
    peaks,models,audit=detect_evidence(red,board,homography(board,120,120),det)
    truth=np.array([[260.,360.],[500.,360.]])
    for q in truth:
        assert any(np.linalg.norm(np.array([p['u_raw_px'],p['v_raw_px']])-q)<10 for p in peaks)
    flat,flat_models,_=detect_evidence(np.ones((400,400))*10,board,homography(board,120,120),det)
    assert not flat and not flat_models
    assert all(p['physical_order'] is None and p['zero_order_confirmed'] is False for p in peaks)
    result['checks']['detector']={'candidate_count':len(peaks),'weak_peak_amplitude_DN':8,'both_truth_positions_recovered':True,'flat_candidates':len(flat),'physical_orders_unconfirmed':True}
    write_json(out/'detector_evidence.json',{'peaks':peaks,'models':models,'audit':audit})
    view=np.uint8(255*np.arcsinh(red/2)/np.arcsinh(red.max()/2))
    im=Image.fromarray(view).convert('RGB');draw=ImageDraw.Draw(im)
    for p in peaks:
        px,py=p['u_raw_px']/2,p['v_raw_px']/2;draw.ellipse((px-8,py-8,px+8,py+8),outline='cyan',width=1)
    im.save(out/'synthetic_detection.png')

    channels={'R':(0,0,0),'G1':(0,1,1),'G2':(1,0,3),'B':(1,1,2)}
    def photometry(amplitude=20.,center=(400.,400.),refs=()):
        a=np.empty((800,800),float)
        for name,(dy,dx,ci) in channels.items():
            yy,xx=np.mgrid[dy:800:2,dx:800:2]
            amp={'R':amplitude*5,'G1':amplitude,'G2':amplitude,'B':amplitude*.25}[name]
            a[dy::2,dx::2]=512+8+.001*xx+.002*yy+amp*np.exp(-((xx-center[0])**2+(yy-center[1])**2)/(2*30**2))
        peak=[{'peak_uid':'synthetic_p001','frame_id':'synthetic','u_raw_px':center[0],'v_raw_px':center[1],'detection_contrast_DN':100,'model_groups':''}]
        b=np.array([[0,0],[800,0],[800,800],[0,800]],float)
        return measure_frame(a,[512]*4,16383,[15360]*4,peak,b,list(refs),cfg=cfg,green_to_red_ratio=None,green_linearity='SYNTHETIC_NOT_RADIOMETRIC')
    measured,ch=photometry();q=measured[0]
    expected=20*2*np.pi*30**2*(1-np.exp(-125**2/(2*30**2)))
    relative=abs(q['Q_G_r125']/expected-1)
    assert relative<.005
    assert q['Q_G_r125']==(q['Q_G1_r125']+q['Q_G2_r125'])/2
    assert q['red_equivalent_from_green_assumed'] is None
    negative,nch=photometry(-10.);assert negative[0]['Q_G_r125']<0
    edge,ech=photometry(center=(40.,400.));assert edge[0]['aperture_coverage_min']<.75
    reflection,rch=photometry(refs=[dict(u_raw_px=550.,v_raw_px=400.)])
    assert reflection[0]['reflection_overlap_r125'] and not reflection[0]['reflection_overlap_r50']
    result['checks']['cfa']={'finite_circle_analytic_relative_error_r125':relative,'green_is_mean_of_two_planes':True,'negative_green_integral':negative[0]['Q_G_r125'],'screen_edge_coverage':edge[0]['aperture_coverage_min'],'reflection_overlap_r125':True,'reflection_overlap_r50':False,'reflection_flux_subtracted':False}
    write_json(out/'cfa_measurements.json',{'positive':{'peaks':measured,'channels':ch},'negative':{'peaks':negative,'channels':nch},'edge':{'peaks':edge,'channels':ech},'reflection_diagnostic':{'peaks':reflection,'channels':rch}})

    # Independent synthetic texture spacing, including blank image rejection.
    yy,xx=np.mgrid[:512,:512];sx,sy=pixel_scale(384,512)
    texture=100+20*np.cos(2*np.pi*xx*sx/12.)
    synthetic,_,_,_=spectrum_peaks(texture,sx,sy)
    assert abs(synthetic['peaks'][0]['period_um']/12-1)<.01
    blank,_,_,_=spectrum_peaks(np.ones((128,128)),sx,sy)
    assert blank['status']=='NO_RESOLVED_TEXTURE'
    real={}
    for name in ('stripe','grid'):
        provenance=read_json(root/f'data/examples/microscopy/{name}_provenance.json')
        with Image.open(root/f'data/examples/microscopy/{name}.png') as image:a=np.asarray(image)
        found,_,_,_=spectrum_peaks(a,provenance['sx_um'],provenance['sy_um'])
        if not found['peaks']:raise ValueError('No candidate in documented real crop')
        real[name]={'kind':'real decoded-video crop, not sensor CFA','provenance':provenance,'measurement':found}
    write_json(out/'real_microscopy.json',real)
    result['checks']['microscopy']={'synthetic_period_um':synthetic['peaks'][0]['period_um'],'truth_period_um':12.,'blank_rejected':True,'real_crops':2,'real_peak_identity':'Candidates retain harmonics; not a certified director fundamental'}
    result.update(status='PASS',elapsed_s=time.perf_counter()-start)
    write_json(out/'examples.json',result)
    (out/'README.md').write_text('# 算法示例结果\n\n所有合成断言通过，见 examples.json。\n\n[合成检测](synthetic_detection.png)、[逐点证据](detector_evidence.json)、[CFA正/负/边界积分](cfa_measurements.json)、[两张真实显微纹理的候选频率](real_microscopy.json)。\n\n合成恢复验证计算；真实显微PNG是解码视频的纹理，不是CFA测光输入。\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
