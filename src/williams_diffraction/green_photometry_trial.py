"""Six-frame green photometry trial. RAW is read only; no physical orders.

The old r50 model is held fixed and used only as an independent diagnostic.
Measured G is never replaced by predicted G. Background is estimated from
off-peak pixels. Adaptive regions preserve signed residuals and record losses.
"""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from .green_channel_study import (BASE, RAW, CHANNELS, read_csv, save_csv,
                                 save_json, digest, background, aperture_weights,
                                 robust_sigma)
from .green_expanded import measure as old_measure
from .geometry import display_to_raw, homography, transform, raw_to_display

TRIAL = BASE / 'outputs/algorithm_trial_v2_r4'
OLD = BASE / 'outputs/stage1_20261010_corrected'


def ellipse_coordinates(xx, yy, angle):
    c, s = np.cos(angle), np.sin(angle)
    return c*xx+s*yy, -s*xx+c*yy


def ellipse_weights(xx, yy, a, b, angle):
    if min(a, b) <= 0:
        raise ValueError('Nonpositive integration axes')
    w = np.zeros_like(xx, float)
    for dy in (-.75, -.25, .25, .75):
        for dx in (-.75, -.25, .25, .75):
            u, v = ellipse_coordinates(xx+dx, yy+dy, angle)
            w += (u/a)**2+(v/b)**2 <= 1
    return w/16


def channel_patch(mosaic, black, center, half, name):
    dy, dx, ci = CHANNELS[name]
    cx, cy = center
    j0=max(0,int(np.floor((cx-half-dx)/2)))
    j1=min(mosaic.shape[1]//2,int(np.ceil((cx+half-dx)/2)))
    i0=max(0,int(np.floor((cy-half-dy)/2)))
    i1=min(mosaic.shape[0]//2,int(np.ceil((cy+half-dy)/2)))
    xx, yy = np.meshgrid(np.arange(j0,j1)*2+dx-cx,
                         np.arange(i0,i1)*2+dy-cy)
    z=mosaic[dy::2,dx::2][i0:i1,j0:j1].astype(float)-black[ci]
    return z,xx,yy,ci


def estimate_shape(g, center, kind, row_angle, cfg):
    """G averages only assist region geometry; final CFA integration is separate."""
    cx,cy=center;half=65
    j=int(round((cx-.5)/2));i=int(round((cy-.5)/2))
    y0=max(0,i-half);y1=min(g.shape[0],i+half+1)
    x0=max(0,j-half);x1=min(g.shape[1],j+half+1)
    xx,yy=np.meshgrid(np.arange(x0,x1)*2+.5-cx,np.arange(y0,y1)*2+.5-cy)
    z=g[y0:y1,x0:x1];rr=np.hypot(xx,yy)
    ring=(rr>=100)&(rr<=128)
    med=float(np.median(z[ring]));noise=max(robust_sigma(z[ring]),.3)
    smooth=ndi.gaussian_filter(z,cfg['shape_smoothing_green_samples'])-med
    peak=float(np.max(smooth[rr<35]))
    mask=(rr<=cfg['shape_initial_radius_raw_px']) & (smooth>max(cfg['shape_threshold_peak_fraction']*peak, cfg['shape_min_signal_to_background_MAD']*noise))
    if mask.sum()<12 or peak<=3*noise:
        return {'sigma_a':22.,'sigma_b':22.,'angle':row_angle if kind=='row_broad' else 0.,'shape_supported':False,'peak_DN':peak,'shape_noise_DN':noise}
    w=np.where(mask,np.maximum(smooth,0),0)
    mx=float(np.sum(w*xx)/w.sum());my=float(np.sum(w*yy)/w.sum())
    # The existing position is retained; no unverified snapping to model sites.
    dx,dy=xx-mx,yy-my
    covariance=np.array([[np.sum(w*dx*dx),np.sum(w*dx*dy)],[np.sum(w*dx*dy),np.sum(w*dy*dy)]])/w.sum()
    if kind=='row_broad':
        c,s=np.cos(row_angle),np.sin(row_angle)
        a=np.array([c,s]);b=np.array([-s,c])
        widths=np.sqrt(np.maximum([a@covariance@a,b@covariance@b],0))
        angle=row_angle
    else:
        values,vectors=np.linalg.eigh(covariance)
        widths=np.sqrt(np.maximum(values[::-1],0));axis=vectors[:,1]
        angle=float(np.arctan2(axis[1],axis[0]))
    # Moment truncation slightly narrows widths. Growth diagnostics, not a
    # Gaussian fit, decide whether an aperture is complete.
    widths=np.clip(widths*1.12,*cfg['shape_sigma_bounds_raw_px'])
    return {'sigma_a':float(widths[0]),'sigma_b':float(widths[1]),'angle':float(angle),'shape_supported':True,'peak_DN':peak,'shape_noise_DN':noise}


def design(x,y,degree):
    x=x/300.;y=y/300.
    a=[np.ones_like(x),x,y]
    if degree==2:a += [x*x,x*y,y*y]
    return np.column_stack(a)


def robust_coeff(M,z):
    keep=np.ones(len(z),bool)
    for _ in range(6):
        if keep.sum()<max(10,M.shape[1]+3):raise ValueError('Too few robust background tiles')
        coef=np.linalg.lstsq(M[keep],z[keep],rcond=None)[0]
        residual=z-M@coef
        scale=max(robust_sigma(residual[keep]),.15)
        keep=np.abs(residual-np.median(residual[keep]))<3*scale
    return coef,keep


def fit_tiled_background(z,xx,yy,mask,degree,cfg):
    coords=np.c_[xx[mask],yy[mask]];target=z[mask]
    if len(target)<1000:raise ValueError('Insufficient unmasked background')
    cells=np.floor((coords+cfg['background_patch_half_raw_px'])/cfg['background_tile_raw_px']).astype(int)
    unique,inverse=np.unique(cells,axis=0,return_inverse=True)
    tx=[];ty=[];tz=[];ids=[]
    for k,cell in enumerate(unique):
        use=inverse==k
        if use.sum()>=cfg['background_min_valid_tile_samples']:
            tx.append(float(np.mean(coords[use,0])));ty.append(float(np.mean(coords[use,1])))
            tz.append(float(np.median(target[use])));ids.append(cell)
    x,y,v=np.array(tx),np.array(ty),np.array(tz)
    if len(v)<cfg['background_min_tiles']:raise ValueError('Insufficient spatial background tiles')
    sectors=np.minimum(7,((np.arctan2(y,x)+np.pi)/(2*np.pi)*8).astype(int))
    if len(set(sectors))<cfg['background_min_sectors']:raise ValueError('Background support is one-sided')
    M=design(x,y,degree);coef,keep=robust_coeff(M,v)
    parity=(np.array(ids).sum(axis=1)%2)==0
    errors=[]
    for train in (parity,~parity):
        if train.sum()<10 or (~train).sum()<5:raise ValueError('Insufficient held-out tile support')
        c,_=robust_coeff(M[train],v[train]);errors.extend((v[~train]-M[~train]@c).tolist())
    field=(design(xx.ravel(),yy.ravel(),degree)@coef).reshape(z.shape)
    residual=z[mask]-field[mask];sigma=max(robust_sigma(residual),.3)
    return {'field':field,'coef':coef,'degree':degree,'sigma':sigma,
            'cv_MAD_DN':max(robust_sigma(np.array(errors)),.05),
            'tile_x':x,'tile_y':y,'tile_values':v,'tile_keep':keep,
            'n_tiles':len(v),'n_pixels':int(mask.sum()),'sectors':len(set(sectors))}


def support_mask(xx,yy,center,all_shapes,reflections,corners,cfg):
    cx,cy=center;mask=np.ones_like(xx,bool)
    for p in all_shapes:
        nx,ny=p['center'];shape=p['shape']
        u,v=ellipse_coordinates(xx+cx-nx,yy+cy-ny,shape['angle'])
        mask &= (u/(cfg['background_peak_exclusion_sigma']*shape['sigma_a']))**2+(v/(cfg['background_peak_exclusion_sigma']*shape['sigma_b']))**2>1
    for nx,ny in reflections:
        mask &= (xx+cx-nx)**2+(yy+cy-ny)**2>cfg['background_reflection_exclusion_raw_px']**2
    # Actual board mask protects a background fit against off-screen machinery.
    pts=np.c_[xx.ravel()+cx,yy.ravel()+cy].astype(np.float32)
    a=corners.astype(float);inside=np.ones(len(pts),bool)
    for p,q in zip(a,np.roll(a,-1,axis=0)):
        cross=(q[0]-p[0])*(pts[:,1]-p[1])-(q[1]-p[1])*(pts[:,0]-p[0])
        inside &= cross>=0
    mask &= inside.reshape(xx.shape)
    rr=np.hypot(xx,yy)
    return mask&(rr>=90)&(rr<=340),inside.reshape(xx.shape)


def measure_target(mosaic,black,center,shape,all_shapes,reflections,corners,cfg):
    result={};diagnostic={};growth={}
    for name in ('R','G1','G2'):
        z,xx,yy,ci=channel_patch(mosaic,black,center,cfg['background_patch_half_raw_px'],name)
        mask,inside=support_mask(xx,yy,center,all_shapes,reflections,corners,cfg)
        primary=fit_tiled_background(z,xx,yy,mask,1,cfg)
        quad=fit_tiled_background(z,xx,yy,mask,2,cfg)
        improvement=1-quad['cv_MAD_DN']/primary['cv_MAD_DN']
        if (improvement>=cfg['quadratic_holdout_improvement_fraction'] and abs(quad['coef'][0]-primary['coef'][0])<=cfg['quadratic_max_center_difference_background_MAD']*primary['sigma']):
            primary=quad
        alternatives=[fit_tiled_background(z,xx,yy,mask&(np.hypot(xx,yy)<=270),primary['degree'],cfg),
                      fit_tiled_background(z,xx,yy,mask&(np.hypot(xx,yy)>=130),primary['degree'],cfg)]
        w50=aperture_weights(xx,yy,50)
        result[name+'_new_r50_Q']=float(4*np.sum(w50*(z-primary['field'])))
        result[name+'_r50_alt_Q']=[float(4*np.sum(w50*(z-a['field']))) for a in alternatives]
        result[name+'_bg_degree']=primary['degree'];result[name+'_bg_cv_MAD_DN']=primary['cv_MAD_DN']
        result[name+'_bg_tiles']=primary['n_tiles'];result[name+'_bg_center_DN']=float(primary['coef'][0])
        gs=[]
        for alpha in cfg['ellipse_growth_multipliers']:
            w=ellipse_weights(xx,yy,alpha*shape['sigma_a'],alpha*shape['sigma_b'],shape['angle'])
            # Separating cells prevent counting the same pixels as two peaks.
            cell=inside.copy()
            for p in all_shapes:
                nx,ny=p['center']
                if np.hypot(nx-center[0],ny-center[1])>3:
                    cell &= xx*xx+yy*yy <= (xx+center[0]-nx)**2+(yy+center[1]-ny)**2
            uncut=float(w.sum());w=w*cell
            q=float(4*np.sum(w*(z-primary['field'])))
            qa=[float(4*np.sum(w*(z-a['field']))) for a in alternatives]
            raw=z+black[ci]
            gs.append({'alpha':alpha,'Q':q,'alt_Q':qa,'area_rawpixel2':float(4*w.sum()),
                       'cell_fraction':float(w.sum()/max(uncut,1e-12)),
                       'sigma_sampling_indicator_Q':float(4*primary['sigma']*np.sqrt(np.sum(w*w))),
                       'max_raw_DN':float(raw[w>0].max()),'near_white':bool(np.any(raw[w>0]>=black[ci]+cfg['green_safety_fraction_of_camera_white']*(15360-black[ci])))})
        growth[name]=gs
        diagnostic[name]=(z,xx,yy,primary,alternatives,mask)
    merged=[]
    for i,alpha in enumerate(cfg['ellipse_growth_multipliers']):
        g1,g2=growth['G1'][i],growth['G2'][i]
        q=(g1['Q']+g2['Q'])/2
        alt=np.mean([g1['alt_Q'],g2['alt_Q']],axis=0)
        error=max(abs(alt-q));sampling=np.hypot(g1['sigma_sampling_indicator_Q'],g2['sigma_sampling_indicator_Q'])/2
        merged.append({'alpha':alpha,'Q_G':q,'Q_R':growth['R'][i]['Q'],
                       'bg_change_fraction':float(error/max(abs(q),1e-12)),
                       'error_indicator_Q':float(max(error,sampling)),
                       'area_rawpixel2':(g1['area_rawpixel2']+g2['area_rawpixel2'])/2,
                       'cell_fraction':min(g1['cell_fraction'],g2['cell_fraction']),
                       'near_white_G':g1['near_white'] or g2['near_white'],
                       'near_white_R':growth['R'][i]['near_white'],
                       'G1_G2_difference_fraction':float((g2['Q']-g1['Q'])/max(abs(q),1e-12)),
                       'increment_fraction': None if i==0 else float(abs(q-merged[i-1]['Q_G'])/max(abs(q),1e-12))})
    selected=None
    # A plateau must contain two genuinely small consecutive increments and
    # an uncut region. Flat clipped-cell sums are not accepted as completeness.
    consecutive=int(cfg['plateau_required_consecutive_steps'])
    if not 1<=consecutive<len(merged):raise ValueError('Invalid plateau step count')
    for i in range(consecutive,len(merged)):
        if all(merged[j]['increment_fraction']<=cfg['plateau_increment_limit_fraction'] and merged[j]['cell_fraction']>=cfg['minimum_cell_fraction'] for j in range(i-consecutive+1,i+1)):
            selected=i;break
    plateau=selected is not None
    if selected is None:
        safe=[i for i,v in enumerate(merged) if v['cell_fraction']>=cfg['minimum_cell_fraction']]
        selected=safe[-1] if safe else 0
    chosen=merged[selected]
    result['new_r50_Q_G']=(result['G1_new_r50_Q']+result['G2_new_r50_Q'])/2
    result['new_r50_Q_R']=result['R_new_r50_Q']
    alternatives50=np.mean([result['G1_r50_alt_Q'],result['G2_r50_alt_Q']],axis=0)
    result['new_r50_background_change_fraction']=float(np.max(np.abs(alternatives50-result['new_r50_Q_G']))/max(abs(result['new_r50_Q_G']),1e-12))
    result.update({'adaptive_Q_G':chosen['Q_G'],'adaptive_Q_R':chosen['Q_R'],
                   'adaptive_area_rawpixel2':chosen['area_rawpixel2'],
                   'chosen_alpha':chosen['alpha'],'plateau_found':plateau,
                   'adaptive_background_change_fraction':chosen['bg_change_fraction'],
                   'adaptive_error_indicator_Q':chosen['error_indicator_Q'],
                   'adaptive_cell_fraction':chosen['cell_fraction'],
                   'G1_G2_difference_fraction':chosen['G1_G2_difference_fraction'],
                   'near_white_G':chosen['near_white_G'],'near_white_R':chosen['near_white_R']})
    return result,merged,diagnostic


def classify(r,fid):
    group=r.get('model_groups','')
    if group=='lattice_2d':return 'lattice_ellipse'
    if 'row' in group or fid in ('026','064','070','077'):return 'row_broad'
    return 'reference_compact'


def matched_sector_sensitivity(diagnostic, center, neighbours, reflections):
    """Same r50 aperture and eight leave-one-angular-sector perturbations.

    Old estimator uses individual annulus samples; new estimator uses its
    off-peak tile medians. This measures spatial support sensitivity, not
    calibration error, a confidence interval, or independent replication.
    """
    base={}; alternatives={}
    for name in ('G1','G2'):
        z,xx,yy,new,_,_=diagnostic[name]
        rr=np.hypot(xx,yy);valid=(rr>=85)&(rr<=120)
        for nx,ny in neighbours:
            if np.hypot(nx-center[0],ny-center[1])>3:
                valid &= (xx+center[0]-nx)**2+(yy+center[1]-ny)**2>70**2
        for nx,ny in reflections:
            valid &= (xx+center[0]-nx)**2+(yy+center[1]-ny)**2>90**2
        w=aperture_weights(xx,yy,50)
        oldfield,*_=background(z,xx,yy,valid)
        base[name]=[float(4*np.sum(w*(z-oldfield))),float(4*np.sum(w*(z-new['field'])))]
        sector=np.minimum(7,((np.arctan2(yy,xx)+np.pi)/(2*np.pi)*8).astype(int))
        tile_sector=np.minimum(7,((np.arctan2(new['tile_y'],new['tile_x'])+np.pi)/(2*np.pi)*8).astype(int))
        M=design(new['tile_x'],new['tile_y'],new['degree'])
        # Weighted design integral avoids materializing eight full fields.
        aperture_M=4*np.sum(w.ravel()[:,None]*design(xx.ravel(),yy.ravel(),new['degree']),axis=0)
        raw_sum=float(4*np.sum(w*z));values=[]
        for omitted in range(8):
            field,*_=background(z,xx,yy,valid&(sector!=omitted))
            keep=tile_sector!=omitted
            coef,_=robust_coeff(M[keep],new['tile_values'][keep])
            values.append([float(4*np.sum(w*(z-field))),float(raw_sum-aperture_M@coef)])
        alternatives[name]=values
    reference=np.mean(list(base.values()),axis=0)
    values=np.mean(list(alternatives.values()),axis=0)
    change=np.max(np.abs(values-reference),axis=0)/np.maximum(np.abs(reference),1e-12)
    return {'matched_sector_before_fraction':float(change[0]),
            'matched_sector_after_fraction':float(change[1]),
            'matched_sector_count':8}


def run(config_path,out):
    import rawpy
    from .paths import require_writable_run
    out=require_writable_run(out)
    cfg=json.loads(config_path.read_text());out.mkdir(parents=True,exist_ok=False)
    (out/'comparisons').mkdir();(out/'diagnostics').mkdir()
    os.environ['MPLCONFIGDIR']=str(out/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Ellipse, Circle
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':11,'savefig.dpi':160,'axes.unicode_minus':False})
    spots=read_csv(TRIAL/'spots.csv');old=read_csv(OLD/'spots.csv')
    # 070 already has candidate positions but was not in the eight-frame trial.
    spots += [{**r,'previous_spot_id':r['spot_id'],'model_groups':'row','detection_origin':'existing_geometry_control'} for r in old if r['frame_id']=='070']
    rejected=read_csv(TRIAL/'rejected_candidates.csv')
    frame_map={r['frame_id']:r for r in read_csv(OLD/'frames.csv')}
    samples={s['frame_id']:s for s in json.loads((BASE/'annotations/sample_config.json').read_text())['samples']}
    model=json.loads((BASE/cfg['model_source']).read_text())['fits']['G_R_50']
    training_frames={r['frame_id'] for r in read_csv(BASE/'outputs/green40_20261010/selected40.csv')}
    table=[];curves=[];hashes=[];frame_diagnostics=[]
    for fid in cfg['sample_ids']:
        path=RAW/f'{fid}.ARW';before=digest(path)
        assert before==frame_map[fid]['sha256']
        with rawpy.imread(str(path)) as raw:
            assert raw.raw_pattern.tolist()==[[0,1],[3,2]]
            mosaic=raw.raw_image_visible.copy();black=raw.black_level_per_channel
        g=(mosaic[0::2,1::2].astype(float)-black[1]+mosaic[1::2,0::2].astype(float)-black[3])/2
        rows=[r for r in spots if r['frame_id']==fid]
        centers=np.array([[float(r['u_raw_px']),float(r['v_raw_px'])] for r in rows])
        anchor=int(np.argmax([float(r.get('contrast_DN',r.get('detection_contrast_DN',0))) for r in rows]))
        near_row=centers[np.abs(centers[:,1]-centers[anchor,1])<80]
        if len(near_row)>=3:
            direction=np.linalg.svd(near_row-near_row.mean(axis=0),full_matrices=False)[2][0]
            angle=float(np.arctan2(direction[1],direction[0]));angle=angle%np.pi
        else:angle=0.
        shapes=[{'center':c,'shape':estimate_shape(g,c,classify(r,fid),angle,cfg)} for r,c in zip(rows,centers)]
        reflections=np.array([[float(r['u_raw_px']),float(r['v_raw_px'])] for r in rejected if r['frame_id']==fid]).reshape(-1,2)
        corners=display_to_raw(samples[fid]['corners_display'],g.shape)
        records=[]
        for r,p in zip(rows,shapes):
            if r['spot_id'] not in cfg['targets'][fid]:continue
            center=p['center'];shape=p['shape'];key=f"{fid}_{r['spot_id']}"
            rec={'frame_id':fid,'spot_id':r['spot_id'],'previous_spot_id':r['previous_spot_id'],
                 'voltage_RMS_V':frame_map[fid]['voltage_recorded_V'],'scan_direction':frame_map[fid]['scan_direction'],
                 'u_raw_px':float(center[0]),'v_raw_px':float(center[1]),'branch':classify(r,fid),
                 **shape,'physical_order':None,'formal_photometry_valid':False}
            flags=[]
            if not shape['shape_supported']:flags.append('LOW_GREEN_SHAPE_EVIDENCE')
            if any(np.linalg.norm(center-rc)<120 for rc in reflections) or 'POSSIBLE_REFLECTION_OVERLAP' in r['quality_flags']:
                flags.append('REFLECTION_OVERLAP_REVIEW')
            try:
                baseline=old_measure(mosaic,black,center,centers,reflections)
                new,growth,diag=measure_target(mosaic,black,center,shape,shapes,reflections,corners,cfg)
                rec.update({k:v for k,v in new.items() if not isinstance(v,list)})
                rec.update(matched_sector_sensitivity(diag,center,centers,reflections))
                rec['baseline_Q_G']=baseline['Q_G_r50'];rec['baseline_Q_R']=baseline['Q_R_r50']
                rec['baseline_background_change_fraction']=baseline['G_background_change_fraction']
                rec['same_r50_signal_change_fraction']=(rec['new_r50_Q_G']-rec['baseline_Q_G'])/max(abs(rec['baseline_Q_G']),1e-12)
                rec['adaptive_vs_baseline_signal_change_fraction']=(rec['adaptive_Q_G']-rec['baseline_Q_G'])/max(abs(rec['baseline_Q_G']),1e-12)
                # The fixed-area intercept is valid only for the original r50 diagnostic.
                pred=model['slope']*rec['new_r50_Q_R']+model['intercept_DN']
                rec['frozen_model_predicted_r50_G']=pred
                pred_before=model['slope']*rec['baseline_Q_R']+model['intercept_DN']
                rec['frozen_model_before_residual_fraction']=(rec['baseline_Q_G']-pred_before)/pred_before if pred_before>0 else None
                rec['model_r50_applicable']=baseline['R_max_raw_within81_DN']<12390.4 and model['x_range_DN'][0]<=rec['new_r50_Q_R']<=model['x_range_DN'][1]
                rec['model_diagnostic_frame_held_out']=fid not in training_frames
                rec['frozen_model_r50_residual_fraction']=(rec['new_r50_Q_G']-pred)/pred if pred>0 else None
                # Variable-area comparison uses the previously fitted zero-offset slope;
                # never carry the fixed r50 intercept into an adaptive integral.
                proportional=model['zero_intercept_slope']*rec['adaptive_Q_R']
                rec['adaptive_proportional_predicted_G']=proportional
                rec['adaptive_ratio_residual_fraction']=(rec['adaptive_Q_G']-proportional)/proportional if proportional>0 else None
                if not rec['plateau_found']:flags.append('NO_ISOLATED_PLATEAU')
                if rec['adaptive_cell_fraction']<cfg['minimum_cell_fraction']:flags.append('REGION_TRUNCATED_BY_CELL_OR_SCREEN')
                if rec['adaptive_background_change_fraction']>cfg['background_sensitivity_limit_fraction']:flags.append('BACKGROUND_SENSITIVE')
                if rec['adaptive_Q_G']<=cfg['signal_to_background_indicator_min']*rec['adaptive_error_indicator_Q']:flags.append('LOW_GREEN_BACKGROUND_SEPARATION')
                if rec['near_white_G']:flags.append('GREEN_NEAR_SATURATION')
                if rec['near_white_R']:flags.append('R_OUTSIDE_VALIDATED_MODEL_RANGE')
                if 'EDGE_RISK' in r['quality_flags']:flags.append('SCREEN_EDGE_REVIEW')
                if fid=='077':flags.append('DIFFUSE_STATE_DISCRETE_FLUX_UNCERTAIN')
                rec['green_integration_quality_pass_trial']=not [f for f in flags if f!='R_OUTSIDE_VALIDATED_MODEL_RANGE']
                if rec['near_white_R']:flags.append('GREEN_NEAR_RED_SATURATION_UNVALIDATED')
                rec['isolated_signal_usable_trial']=not flags
                rec['fixed_aperture_relative_signal_usable_trial']=(
                    rec['new_r50_Q_G']>0
                    and rec['new_r50_background_change_fraction']<=cfg['background_sensitivity_limit_fraction']
                    and rec['model_r50_applicable']
                    and not any(f in flags for f in ('REFLECTION_OVERLAP_REVIEW','GREEN_NEAR_SATURATION','SCREEN_EDGE_REVIEW','LOW_GREEN_SHAPE_EVIDENCE','DIFFUSE_STATE_DISCRETE_FLUX_UNCERTAIN')))
                for v in growth:curves.append({'frame_id':fid,'spot_id':r['spot_id'],**v})
                records.append((rec,growth,diag))
                fig,axes=plt.subplots(2,2,figsize=(12.8,10),layout='constrained')
                half=max(160,shape['sigma_a']*5.1,shape['sigma_b']*5.1)
                zz,xx,yy,_=channel_patch(mosaic,black,center,int(half),'G1')
                lo=float(np.percentile(zz,5));hi=float(np.percentile(zz,99.5));show=np.arcsinh((zz-lo)/5)
                bounds=[float(xx.min()),float(xx.max()),float(yy.max()),float(yy.min())]
                for ax in axes[0]:
                    ax.imshow(show,cmap='gray',extent=bounds,vmin=0,vmax=np.arcsinh(max(hi-lo,1)/5));ax.set(xlabel='相对中心 / RAW像素',ylabel='相对中心 / RAW像素')
                axes[0,0].add_patch(Circle((0,0),50,fill=False,color='#5bd6ef',lw=2))
                for radius in (85,120):axes[0,0].add_patch(Circle((0,0),radius,fill=False,color='#7aca90',lw=1.5))
                axes[0,0].set_title('优化前：50像素圆孔径与固定背景环')
                a=rec['chosen_alpha']*shape['sigma_a'];b=rec['chosen_alpha']*shape['sigma_b']
                axes[0,1].add_patch(Ellipse((0,0),2*a,2*b,angle=np.degrees(shape['angle']),fill=False,color='#ffb351',lw=2))
                dd=diag['G1'][3];axes[0,1].scatter(dd['tile_x'],dd['tile_y'],s=12,c='#65dba8',alpha=.7)
                axes[0,1].set(xlim=(-half,half),ylim=(half,-half),title='优化后：分支区域与避峰背景块')
                alpha=[v['alpha'] for v in growth];q=np.array([v['Q_G'] for v in growth])/1e6
                axes[1,0].plot(alpha,q,'o-',color='#b96921',label='绿色净积分')
                axes[1,0].axhline(rec['baseline_Q_G']/1e6,ls='--',c='#1687a0',label='旧圆孔径积分')
                axes[1,0].axvline(rec['chosen_alpha'],ls=':',c='black',label='本轮选取区域')
                axes[1,0].set(xlabel='区域轴长 / 实测形态尺度',ylabel='QG / 10⁶ DN·RAW pixel²',title='区域增长：没有平台时不称完整斑功率');axes[1,0].legend(fontsize=9)
                axes[1,1].plot(alpha,[v['bg_change_fraction']*100 for v in growth],'o-',label='背景选择变化')
                axes[1,1].plot(alpha,[100*(1-v['cell_fraction']) for v in growth],'s--',label='邻峰/屏边造成的面积损失')
                axes[1,1].axhline(5,c='gray',ls=':');axes[1,1].set(xlabel='区域轴长 / 实测形态尺度',ylabel='%',title='质量诊断：敏感或重叠保留标记');axes[1,1].legend(fontsize=9)
                fig.suptitle(f"{fid} / {r['spot_id']} | {rec['branch']} | 同法扇区扰动 {rec['matched_sector_before_fraction']*100:.1f}% → {rec['matched_sector_after_fraction']*100:.1f}%（同圆孔径）\n"+('；'.join(flags) if flags else '通过本轮小样本质量条件；非绝对功率标定'),fontsize=12)
                fig.savefig(out/'diagnostics'/f'{key}.png');plt.close(fig)
            except ValueError as e:
                rec['measurement_error']=str(e);flags.append('BACKGROUND_UNRELIABLE');rec['isolated_signal_usable_trial']=False
            rec['quality_flags']=';'.join(flags);table.append(rec)
        # Compare masks on the exact same display data; original stray light is intact.
        x0=max(0,int(min(centers[:,0])-180)//2);x1=min(g.shape[1],int(max(centers[:,0])+180)//2)
        y0=max(0,int(min(centers[:,1])-230)//2);y1=min(g.shape[0],int(max(centers[:,1])+230)//2)
        field=g[y0:y1,x0:x1]
        vmax=np.arcsinh(1500/5);fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
        for ax in axes:
            ax.imshow(np.arcsinh(np.maximum(field,0)/5),cmap='gray',vmin=0,vmax=vmax,
                      extent=[x0*2+.5,x1*2+.5,y1*2+.5,y0*2+.5]);ax.axis('off')
        for rec,growth,diag in records:
            cx,cy=rec['u_raw_px'],rec['v_raw_px'];axes[0].add_patch(Circle((cx,cy),50,fill=False,color='#56d7f2',lw=1.5))
            alpha=rec['chosen_alpha'];axes[1].add_patch(Ellipse((cx,cy),2*alpha*rec['sigma_a'],2*alpha*rec['sigma_b'],angle=np.degrees(rec['angle']),fill=False,color='#ffb14e' if rec['isolated_signal_usable_trial'] else '#ed8487',lw=1.5))
            for ax in axes:ax.text(cx+16,cy-20,rec['spot_id'][5:],color='#8ee3ee',fontsize=10)
        for ax in axes:
            for cx,cy in reflections:ax.plot(cx,cy,'x',c='#ffb553',ms=9)
        meta=frame_map[fid];axes[0].set_title(f'{fid} {meta["scan_direction"]} {meta["voltage_recorded_V"]} V RMS\n优化前：相同圆孔径')
        axes[1].set_title('优化后：实测分支区域；红圈表示需复核\n同一绿色数据、统一显示范围，杂光原像素保留')
        fig.savefig(out/'comparisons'/f'{fid}.png');plt.close(fig)
        after=digest(path);assert before==after
        hashes.append({'frame_id':fid,'sha256_before':before,'sha256_after':after})
        frame_diagnostics.append({'frame_id':fid,'targets':len(cfg['targets'][fid]),'measured':len(records),'usable_trial':sum(v[0]['isolated_signal_usable_trial'] for v in records)})
        print(fid,frame_diagnostics[-1],flush=True)
    save_csv(out/'measurements.csv',table);save_csv(out/'growth_curves.csv',curves)
    save_json(out/'raw_integrity.json',hashes);save_json(out/'run_config.json',cfg)
    save_json(out/'frozen_prediction_model.json',model)
    valid=[r for r in table if 'new_r50_background_change_fraction' in r]
    clean=[r for r in valid if 'REFLECTION_OVERLAP' not in r['quality_flags'] and r['baseline_Q_G']>0]
    summary={'sample_frames':cfg['sample_ids'],'target_count':len(table),'measured_count':len(valid),
             'usable_trial_count':sum(r['isolated_signal_usable_trial'] for r in table),
             'fixed_aperture_relative_signal_usable_trial_count':sum(r.get('fixed_aperture_relative_signal_usable_trial',False) for r in table),
             'background_comparison_count':len(clean),
             'before_r50_bg_sensitivity_p50_p95_pct':np.percentile([r['baseline_background_change_fraction']*100 for r in clean],[50,95]).tolist() if clean else None,
             'after_r50_bg_sensitivity_p50_p95_pct':np.percentile([r['new_r50_background_change_fraction']*100 for r in clean],[50,95]).tolist() if clean else None,
             'matched_sector_before_p50_p95_pct':np.percentile([r['matched_sector_before_fraction']*100 for r in clean],[50,95]).tolist() if clean else None,
             'matched_sector_after_p50_p95_pct':np.percentile([r['matched_sector_after_fraction']*100 for r in clean],[50,95]).tolist() if clean else None,
             'historical_sensitivity_comparison_warning':'Old and new historical metrics perturb different supports/model families. Do not interpret their ratio as an accuracy gain. Use matched_sector fields for the common angular omission check.',
             'plateau_count':sum(r.get('plateau_found',False) for r in valid),
             'green_integration_quality_pass_trial_count':sum(r.get('green_integration_quality_pass_trial',False) for r in valid),
             'model_role':'Held fixed independent diagnostic; no residual-based background tuning.',
             'frame_results':frame_diagnostics,'full80_processed':False,
             'formal_photometry_valid':False,'unit':'DN times RAW pixel area (pixel^2), not optical watts'}
    save_json(out/'summary.json',summary)
    save_json(out/'source_hashes.json',{str(source_label(p)):digest(p) for p in [Path(__file__),config_path,PACKAGE_DIR/'green_channel_study.py',PACKAGE_DIR/'green_expanded.py',BASE/cfg['model_source']]})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=BASE/'annotations/photometry_trial_green_v1.json');parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.out)
