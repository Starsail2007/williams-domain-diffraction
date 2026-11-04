"""Held-out screen-outline consistency and reference-zero registration.

Only existing sample displays/positions are inspected. Neither held-out edge
pixels nor user-confirmed outer dimensions constitute independent mm checks.
Physical diffraction orders remain null; 001 alone has direct user confirmation.
"""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .green_channel_study import BASE, read_csv, save_csv, save_json, digest
from .geometry import display_to_raw, homography, transform, raw_to_display


def fit_line(points):
    if len(points)<8:raise ValueError('Insufficient edge points')
    keep=np.ones(len(points),bool)
    for _ in range(5):
        center=points[keep].mean(axis=0)
        direction=np.linalg.svd(points[keep]-center,full_matrices=False)[2][0]
        normal=np.array([-direction[1],direction[0]])
        distances=(points-center)@normal
        sigma=max(1.4826*np.median(np.abs(distances[keep]-np.median(distances[keep]))),.25)
        keep=np.abs(distances-np.median(distances[keep]))<=3*sigma
        if keep.sum()<8:raise ValueError('Unstable screen edge')
    return center,normal,keep


def find_edge(image,a,b,center,edge_index):
    tangent=(b-a)/np.linalg.norm(b-a);normal=np.array([-tangent[1],tangent[0]])
    if normal@(center-(a+b)/2)<0:normal=-normal
    offsets=np.arange(-12,12.01,.25);rows=[]
    for n,t in enumerate(np.linspace(.10,.90,49)):
        if edge_index==2 and .32<t<.68:continue # known central clamp occlusion
        p=(1-t)*a+t*b
        loc=p+offsets[:,None]*normal
        values=ndi.map_coordinates(image,[loc[:,1],loc[:,0]],order=1,mode='nearest')
        grad=np.gradient(ndi.gaussian_filter1d(values,2),offsets)
        # A consistent dark-board transition has a negative inward gradient.
        strength=-grad;k=int(np.argmax(strength))
        score=float(strength[k]);keep=np.abs(offsets-offsets[k])>=3
        second=float(strength[keep].max()) if keep.any() else 0
        accepted=(2<k<len(offsets)-3 and score>=1.0 and score>=1.20*max(second,.1))
        q=p+offsets[k]*normal
        rows.append({'sample_index':n,'t':float(t),'x_display_px':float(q[0]),'y_display_px':float(q[1]),
                     'offset_display_px':float(offsets[k]),'gradient_DN_display_per_px':score,
                     'accepted':accepted,'reason':'OK' if accepted else 'AMBIGUOUS_OR_WEAK_OR_SEARCH_EDGE',
                     'split':'train' if n//6%2==0 else 'held_out'})
    return rows


def intersection(line1,line2):
    c1,n1=line1;c2,n2=line2;M=np.vstack([n1,n2])
    if abs(np.linalg.det(M))<.01:raise ValueError('Near parallel boundary lines')
    return np.linalg.solve(M,[n1@c1,n2@c2])


def run(config_path,out):
    from .paths import require_writable_run
    out=require_writable_run(out)
    cfg=json.loads(config_path.read_text());samples=json.loads((BASE/'annotations/sample_config.json').read_text())
    confirmations=json.loads((BASE/'annotations/experiment_confirmations_20261010.json').read_text())
    out.mkdir(parents=True,exist_ok=False);os.environ['MPLCONFIGDIR']=str(out/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':10,'axes.unicode_minus':False,'savefig.dpi':150})
    fids=[cfg['reference_frame_id']]+cfg['sample_ids'];rows=[];edges=[];calibrations=[];registrations=[]
    old=read_csv(BASE/'outputs/stage1_20261010_corrected/spots.csv')
    ref=next(r for r in old if r['frame_id']=='001' and r['spot_id']=='spot_002')
    refmm=None
    names=['上边','右边','下边','左边']
    for fid in fids:
        sample=next(s for s in samples['samples'] if s['frame_id']==fid)
        image=Image.open(BASE/'outputs/rawcheck_20261010/raw_displays'/f'{fid}.png').convert('L')
        gray=ndi.gaussian_filter(np.array(image,dtype=float),1.0)
        corners=np.array(sample['corners_display'],float);lines=[];this=[]
        fig,ax=plt.subplots(figsize=(10,7),layout='constrained');ax.imshow(gray,cmap='gray',vmin=0,vmax=255)
        ax.plot(*np.vstack([corners,corners[0]]).T,'--',c='#e0b04a',lw=1.4,label='旧人工边界')
        for i,(a,b) in enumerate(zip(corners,np.roll(corners,-1,axis=0))):
            pp=find_edge(gray,a,b,corners.mean(axis=0),i)
            for p in pp:p.update({'frame_id':fid,'edge':names[i]})
            train=np.array([[p['x_display_px'],p['y_display_px']] for p in pp if p['accepted'] and p['split']=='train']).reshape(-1,2)
            held=np.array([[p['x_display_px'],p['y_display_px']] for p in pp if p['accepted'] and p['split']=='held_out']).reshape(-1,2)
            status='INSUFFICIENT_SUPPORT';p95=None;line=None
            if len(train)>=8 and len(held)>=6:
                try:
                    c,n,keep=fit_line(train);residual=np.abs((held-c)@n)
                    p95=float(np.percentile(residual,95));line=(c,n)
                    status='OUTLINE_CONSISTENT' if p95<=cfg['geometry_heldout_strip_max_p95_display_px'] else 'OUTLINE_REVIEW_REQUIRED'
                    direction=np.array([-n[1],n[0]]);q=c+np.array([-350,350])[:,None]*direction
                    ax.plot(q[:,0],q[:,1],c='#69dbbd',lw=1)
                except ValueError:pass
            if len(train):ax.scatter(train[:,0],train[:,1],s=10,c='#68d4df')
            if len(held):ax.scatter(held[:,0],held[:,1],s=12,c='#f59b72',marker='x')
            for p in pp:rows.append(p)
            record={'frame_id':fid,'edge':names[i],'accepted_train':len(train),'accepted_held_out':len(held),
                    'held_out_p95_display_px':p95,'status':status,
                    'independent_metric_validation':False,'physical_boundary_identity_confirmed':True}
            edges.append(record);this.append(record);lines.append(line)
        # Refinement is accepted only when every held-out edge meets the rule.
        refined=None
        if all(r['status']=='OUTLINE_CONSISTENT' for r in this):
            refined=np.array([intersection(lines[3],lines[0]),intersection(lines[0],lines[1]),intersection(lines[1],lines[2]),intersection(lines[2],lines[3])])
            if not cv2.isContourConvex(refined.astype(np.float32)):refined=None
        used=corners if refined is None else refined
        H=homography(display_to_raw(used,(2344,3514)),100,120)
        calibrations.append({'frame_id':fid,'corners_display_used':used.tolist(),'H_raw_to_candidate_mm':H.tolist(),
                             'refinement_accepted':refined is not None,'boundary_identity_confirmed':True,
                             'independent_metric_validation':False,'lens_distortion_corrected':False})
        candidates=[r for r in old if r['frame_id']==fid]
        q=np.array([[float(r['u_raw_px']),float(r['v_raw_px'])] for r in candidates]);mm=transform(q,H)
        if fid==cfg['reference_frame_id']:
            k=next(i for i,r in enumerate(candidates) if r['spot_id']==ref['spot_id'])
            refmm=mm[k].copy()
        distances=np.linalg.norm(mm-refmm,axis=1)
        k=int(np.argmin(distances));dist=float(distances[k])
        second=float(np.sort(distances)[1]) if len(distances)>1 else None
        chosen=candidates[k]
        registrations.append({'frame_id':fid,'baseline_spot_id':chosen['spot_id'],'X_candidate_mm':float(mm[k,0]),'Y_candidate_mm':float(mm[k,1]),
                              'distance_to_001_registered_candidate_mm':dist,
                              'second_candidate_distance_mm':second,
                              'candidate_distance_margin_mm':None if second is None else second-dist,
                              'zero_order_status':'USER_CONFIRMED_REFERENCE' if fid=='001' else 'REGISTERED_REFERENCE_CANDIDATE_REVIEW',
                              'physical_order':0 if fid=='001' else None,
                              'zero_order_confirmed':fid=='001','brightest_rule_used_for_assignment':False})
        pos=raw_to_display(q[k:k+1],(2344,3514))[0];ax.plot(*pos,'o',mfc='none',mec='#ed72d8',ms=12)
        ax.text(pos[0]+12,pos[1], '001已确认零级' if fid=='001' else '注册参考候选',color='#efa1ee')
        ax.set(xlim=(0,1406),ylim=(938,0),title=f'{fid} 屏边留出检查：青点训练，橙叉留出\n'+('四边通过本轮轮廓检查' if refined is not None else '自动边线不足，保留人工候选变换')+'；不等于独立毫米标定');ax.axis('off')
        fig.savefig(out/f'{fid}_outline.png');plt.close(fig)
    save_csv(out/'edge_samples.csv',rows);save_csv(out/'edge_validation.csv',edges)
    save_csv(out/'zero_reference_registration.csv',registrations);save_json(out/'candidate_calibrations.json',calibrations)
    save_json(out/'summary.json',{'sample_ids':fids,'refined_geometry_count':sum(r['refinement_accepted'] for r in calibrations),
                                'metric_independent_validation':False,'user_boundary_identity_confirmed':True,
                                'user_zero_order_reference_confirmed':True,'physical_orders_assigned_to_other_spots':False,
                                'note':'Held-out points test image contour consistency only. Old images lack extra coplanar known-length references.',
                                'experiment_confirmations':confirmations})
    save_json(out/'source_hashes.json',{str(source_label(p)):digest(p) for p in [Path(__file__),config_path,PACKAGE_DIR/'geometry.py',BASE/'annotations/sample_config.json',BASE/'outputs/stage1_20261010_corrected/spots.csv',BASE/'annotations/experiment_confirmations_20261010.json']})
    print(json.dumps({'frames':len(fids),'accepted_refined_H':sum(r['refinement_accepted'] for r in calibrations)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=BASE/'annotations/photometry_trial_green_v1.json');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.out)
