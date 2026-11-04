"""RAW-only integrals. Rendering/homographies are never inputs to the flux sum."""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import json
import numpy as np
from .common import STAGE1
from .green_photometry_trial import channel_patch,support_mask,estimate_shape,fit_tiled_background
from .green_channel_study import aperture_weights,CHANNELS
from .weak_tail_trial import region_cell
from .guide_events import emit
from .runtime_trace import log_values


def aperture_reflection_overlap(xx,yy,center,weights,reflections,mask_radius_raw_px=90.):
    """Intersect actual aperture/partition/screen weights with confirmed masks.

    This is a contamination diagnostic, not a correction to the signed integral.
    Each CFA sample represents four RAW pixels; overlapping masks count once.
    """
    area=float(4*np.sum(weights));refs=np.asarray(reflections,float).reshape(-1,2)
    result=dict(area_raw_pixel2=0.,fraction=0.,mask_radius_raw_px=float(mask_radius_raw_px),
                contributing_reflections=[],integral_corrected=False)
    use=weights>0
    if not len(refs) or not use.any():return result
    extent=float(np.hypot(xx[use],yy[use]).max());center=np.asarray(center,float)
    union=np.zeros(weights.shape,bool)
    for ref in refs:
        distance=float(np.linalg.norm(center-ref))
        if distance>extent+mask_radius_raw_px:continue
        mask=(xx+center[0]-ref[0])**2+(yy+center[1]-ref[1])**2<=mask_radius_raw_px**2
        overlap=float(4*np.sum(weights*mask))
        if overlap>0:result['contributing_reflections'].append(dict(raw_xy=ref.tolist(),distance_raw_px=distance,area_raw_pixel2=overlap))
        union |= mask
    overlap=float(4*np.sum(weights*union))
    result.update(area_raw_pixel2=overlap,fraction=overlap/area if area else 0.)
    return result


def measure_frame(mosaic,black,white,camera_white,peaks,corners,reflections,cfg=None,
                  green_to_red_ratio=.11937135801286244,
                  green_linearity='USER_AUTHORIZED_ASSUMPTION'):
    # Defaults retain the first experiment; new batches supply their own profile.
    if cfg is None:cfg=json.loads((FROZEN_INPUTS/'photometry_trial_green_v1.json').read_text())
    centers=np.array([[p['u_raw_px'],p['v_raw_px']] for p in peaks])
    g=(mosaic[0::2,1::2].astype(float)-black[1]+mosaic[1::2,0::2].astype(float)-black[3])/2
    anchor=int(np.argmax([p['detection_contrast_DN'] for p in peaks]));near=centers[abs(centers[:,1]-centers[anchor,1])<80]
    direction=np.linalg.svd(near-near.mean(axis=0),full_matrices=False)[2][0] if len(near)>2 else [1,0]
    angle=float(np.arctan2(direction[1],direction[0]))%np.pi
    shapes=[{'center':c,'shape':estimate_shape(g,c,'row_broad' if 'row' in p['model_groups'] else 'lattice_ellipse',angle,cfg)} for c,p in zip(centers,peaks)]
    refs=np.array([[r['u_raw_px'],r['v_raw_px']] for r in reflections]).reshape(-1,2)
    measurements=[];channels=[]
    for p,shape in zip(peaks,shapes):
        center=shape['center'];rec={'peak_uid':p['peak_uid'],'frame_id':p['frame_id'],'aperture_radius_raw_px':125.,'background_protected_radius_raw_px':180.,'measurement_unit':'DN_RAW_pixel2','green_linearity':green_linearity}
        rec.update({f'background_exclusion_shape_{k}':v for k,v in shape['shape'].items()})
        flags=[];stats=[]
        if not shape['shape']['shape_supported']:flags.append('WEAK_SHAPE_EVIDENCE')
        if any(np.linalg.norm(center-r)<150 for r in refs):flags.append('NEAR_REFLECTION')
        for name in ('R','G1','G2','B'):
            row={'peak_uid':p['peak_uid'],'frame_id':p['frame_id'],'channel':name}
            bg=None;error=None
            # The same plane is retained. Edge cases explicitly broaden support or relax
            # spatial diagnostics; neither brightness ratios nor fitted models select it.
            for half,minimum_sectors,minimum_tiles,label in [(360,6,18,'LOCAL_PLANE_180_340'),(360,2,18,'LOCAL_PLANE_RELAXED_EDGE_SUPPORT'),(600,2,18,'EXPANDED_PLANE_180_580')]:
                local_cfg={**cfg,'background_patch_half_raw_px':half,'background_min_sectors':minimum_sectors,'background_min_tiles':minimum_tiles}
                z,xx,yy,ci=channel_patch(mosaic,black,center,half,name)
                emit('photometry.channel', peak=p['peak_uid'], channel=name, array=z, black=float(black[ci]))
                mask,inside=support_mask(xx,yy,center,shapes,refs,corners,local_cfg)
                rr=np.hypot(xx,yy)
                if half==600:
                    # Rebuild the radial bound only; preserve all off-peak/screen exclusions.
                    extra_cfg={**local_cfg};mask,inside=support_mask(xx,yy,center,shapes,refs,corners,extra_cfg)
                    for target in shapes:
                        from .green_photometry_trial import ellipse_coordinates
                        u,v=ellipse_coordinates(xx+center[0]-target['center'][0],yy+center[1]-target['center'][1],target['shape']['angle'])
                        inside_peak=(u/(4*target['shape']['sigma_a']))**2+(v/(4*target['shape']['sigma_b']))**2<=1
                        inside=inside & ~inside_peak
                    for rx,ry in refs:inside &= (xx+center[0]-rx)**2+(yy+center[1]-ry)**2>cfg['background_reflection_exclusion_raw_px']**2
                    mask=inside&(rr>=180)&(rr<=580)
                    # Integration uses the physical screen mask, not the background peak mask.
                    _,inside=support_mask(xx,yy,center,shapes,refs,corners,extra_cfg)
                try:
                    bg=fit_tiled_background(z,xx,yy,mask&(rr>=180),1,local_cfg);error=None;break
                except ValueError as e:error=str(e)
            if bg is None:
                row.update({'measurement_status':'BACKGROUND_FAILED','measurement_error':error});channels.append(row);stats.append(row);flags.append(name+'_BACKGROUND_FAILED');continue
            emit('photometry.background', peak=p['peak_uid'], channel=name, array=bg['field'], mask=mask&(rr>=180), coefficients=bg['coef'])
            cell=region_cell(xx,yy,center,centers,inside)
            for rad in (50,125):
                whole=aperture_weights(xx,yy,rad);w=whole*cell;area=float(4*w.sum())
                dy,dx,_=CHANNELS[name]
                ideal_x=np.arange(int(np.floor((center[0]-rad-2-dx)/2)),int(np.ceil((center[0]+rad+2-dx)/2))+1)*2+dx-center[0]
                ideal_y=np.arange(int(np.floor((center[1]-rad-2-dy)/2)),int(np.ceil((center[1]+rad+2-dy)/2))+1)*2+dy-center[1]
                ix,iy=np.meshgrid(ideal_x,ideal_y);ideal_area=float(4*aperture_weights(ix,iy,rad).sum())
                row[f'Q_r{rad}']=float(4*np.sum(w*(z-bg['field'])))
                log_values('CFA_aperture',peak=p['peak_uid'],channel=name,radius_raw_px=rad,integral=row[f'Q_r{rad}'],area=area,weights=w,residual=z-bg['field'])
                if rad==125:emit('photometry.integral', peak=p['peak_uid'], channel=name, weights=w, residual=z-bg['field'], integral=row[f'Q_r{rad}'], area=area)
                row[f'area_raw_pixel2_r{rad}']=area;row[f'coverage_r{rad}']=area/ideal_area
                overlap=aperture_reflection_overlap(xx,yy,center,w,refs,cfg['background_reflection_exclusion_raw_px'])
                row[f'reflection_overlap_area_raw_pixel2_r{rad}']=overlap['area_raw_pixel2']
                row[f'reflection_overlap_fraction_r{rad}']=overlap['fraction']
                row[f'reflection_overlap_evidence_r{rad}']=overlap
                if overlap['area_raw_pixel2']>0:flags.append(f'REFLECTION_APERTURE_OVERLAP_R{rad}')
                row[f'raw_sensor_coverage_r{rad}']=float(4*whole.sum()/ideal_area)
                row[f'mean_net_DN_r{rad}']=row[f'Q_r{rad}']/area if area else None
            raw=z+black[ci];use=w>0
            cap=camera_white[ci] if camera_white and camera_white[ci] else white
            row.update({'measurement_status':'OK','background_method':label,'background_degree':1,
                        'background_c0_DN':float(bg['coef'][0]),'background_cx_per300_DN':float(bg['coef'][1]),'background_cy_per300_DN':float(bg['coef'][2]),
                        'background_sigma_DN':bg['sigma'],'background_cv_MAD_DN':bg['cv_MAD_DN'],'background_tiles':bg['n_tiles'],'background_pixels':bg['n_pixels'],'background_sectors':bg['sectors'],
                        'black_level_DN':int(black[ci]),'white_level_DN':int(white),'camera_white_level_DN':int(cap),
                        'max_raw_DN':float(raw[use].max()) if use.any() else None,'saturated_sample_count':int(np.sum(raw[use]>=white)),
                        'near_camera_white_sample_count':int(np.sum(raw[use]>=black[ci]+.8*(cap-black[ci]))),
                        'sample_count_r125':int(use.sum())})
            if label!='LOCAL_PLANE_180_340':flags.append(name+'_BACKGROUND_SUPPORT_FALLBACK')
            if row['coverage_r125']<.995:flags.append('REGION_PARTITIONED_OR_SCREEN_CLIPPED')
            if row['saturated_sample_count']:flags.append(name+'_SATURATED')
            elif row['near_camera_white_sample_count']:flags.append(name+'_NEAR_CAMERA_WHITE')
            channels.append(row);stats.append(row)
        by={s['channel']:s for s in stats}
        for rad in (50,125):
            for name in ('R','G1','G2','B'):rec[f'Q_{name}_r{rad}']=by[name].get(f'Q_r{rad}')
            g1,g2=rec[f'Q_G1_r{rad}'],rec[f'Q_G2_r{rad}']
            rec[f'Q_G_r{rad}']=(g1+g2)/2 if g1 is not None and g2 is not None else None
            values=[rec[f'Q_{name}_r{rad}'] for name in ('R','G','B')]
            rec[f'Q_RGB_r{rad}']=sum(values) if all(v is not None for v in values) else None
            rec[f'reflection_overlap_area_max_raw_pixel2_r{rad}']=max(s.get(f'reflection_overlap_area_raw_pixel2_r{rad}',0.) for s in stats)
            rec[f'reflection_overlap_fraction_max_r{rad}']=max(s.get(f'reflection_overlap_fraction_r{rad}',0.) for s in stats)
            rec[f'reflection_overlap_r{rad}']=rec[f'reflection_overlap_area_max_raw_pixel2_r{rad}']>0
        rec['Q_G']=rec['Q_G_r125'];rec['Q_R']=rec['Q_R_r125'];rec['Q_B']=rec['Q_B_r125'];rec['Q_RGB']=rec['Q_RGB_r125']
        if green_to_red_ratio is not None and (not np.isfinite(green_to_red_ratio) or green_to_red_ratio<=0):
            raise ValueError('green_to_red_ratio must be positive or None')
        rec['red_equivalent_from_green_assumed']=rec['Q_G']/green_to_red_ratio if rec['Q_G'] is not None and green_to_red_ratio is not None else None
        rec['aperture_coverage_min']=min([s.get('coverage_r125',0) for s in stats])
        rec['measurement_status']='OK' if all(s['measurement_status']=='OK' for s in stats) else 'PARTIAL_CHANNEL_FAILURE'
        rec['quality_flags']=';'.join(sorted(set(flags)));measurements.append(rec)
    return measurements,channels
