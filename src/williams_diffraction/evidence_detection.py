"""Versioned image-evidence detector; predictions propose, never populate, peaks.

Inputs are black-subtracted R CFA samples and a screen homography in visible RAW
pixel centres. Returned positions are measured in RAW; screen fitting is in mm.
This is a candidate catalogue, not physical-order identification or photometry.
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .geometry import centroid, transform
from .guide_events import emit
from .runtime_trace import log_values
from .candidate_evidence import (resolved_pair, cfa_background_evidence,
                                  contextual_brightness, connected_saddle,boundary_ridge_evidence,
                                  single_cfa_impulse_evidence)
from .guided_detection import (fit_lattice, fit_row, prediction_sites,
                                local_evidence, normalized, patch)


def maxima_candidates(field, threshold, valid=None):
    mask = (field == ndi.maximum_filter(field, size=11)) & (field >= threshold)
    if valid is not None:
        mask &= np.asarray(valid, bool)
    yy, xx = np.where(mask)
    order = np.argsort(-field[yy, xx], kind='stable')
    return [(float(field[yy[i], xx[i]]), np.array([xx[i], yy[i]], float)) for i in order]


def separated(q, others, minimum_raw_px):
    return all(np.linalg.norm(q - p) * 2 >= minimum_raw_px for p in others)


def localize(red, q, radius=32.5, maximum_shift_raw_px=12.):
    center, bg, major, ratio = centroid(red, q, radius)
    shift = float(np.linalg.norm(center-q)*2)
    # A neighbouring tail must not pull the reported centre away from its maximum.
    return (center if shift <= maximum_shift_raw_px else q), dict(
        centroid_shift_raw_px=shift, localization='centroid' if shift <= maximum_shift_raw_px else 'local_maximum',
        background_DN=bg, major_rms_raw_px=major, axis_ratio=ratio)


def near_white_localize(red, green, q, coarse_seed, radius=32.5):
    """Conditional location from aligned G samples when the R maximum is flat.

    The caller must establish native near-white samples in this candidate's core.
    No radiometric correction or claim of an unsaturated green core is made.
    """
    fallback, shape = localize(red, q, radius)
    rc, rbg, rmajor, rratio = centroid(red, coarse_seed, radius)
    gc, gbg, gmajor, gratio = centroid(green, coarse_seed, radius)
    separation = float(np.linalg.norm(gc-rc)*2)
    shift = float(np.linalg.norm(gc-q)*2)
    coarse_shift = float(np.linalg.norm(gc-coarse_seed)*2)
    x,y=np.clip(np.rint(gc).astype(int),[0,0],np.array(green.shape[::-1])-1)
    accepted=(np.isfinite(gc).all() and green[y,x]>gbg and gmajor>0
              and np.linalg.norm(coarse_seed-q)*2<=40 and coarse_shift<=12 and shift<=40)
    evidence=dict(method='NATIVE_NEAR_WHITE_ALIGNED_GREEN_CENTROID',
        status='CONDITIONAL_GREEN_LOCATION' if accepted else 'RED_LOCATION_RETAINED',
        red_centroid_sample_xy=rc.tolist(),green_centroid_sample_xy=gc.tolist(),
        coarse_seed_sample_xy=np.asarray(coarse_seed).tolist(),
        red_green_centroid_separation_raw_px=separation,
        green_shift_from_fine_raw_px=shift,green_shift_from_coarse_raw_px=coarse_shift,
        interpretation='position diagnostic only; native clipping and physical reference uncertainty remain')
    shape['near_white_localization']=evidence
    if accepted:
        shape.update(localization='NEAR_WHITE_GREEN_CENTROID_CONDITIONAL',
            localization_channel='G_MEAN_ALIGNED_AT_R_CENTRES',centroid_shift_raw_px=shift,
            red_background_DN=rbg,red_major_rms_raw_px=rmajor,red_axis_ratio=rratio,
            background_DN=gbg,major_rms_raw_px=gmajor,axis_ratio=gratio,
            localization_channel_disagreement=separation>4.)
        return gc,shape
    return fallback,shape


def analytic_templates():
    """Explicit shape prior for frames without an unsaturated side-spot template."""
    y,x=np.mgrid[-20:21,-20:21];bank=[]
    for sigma in (5.,9.,14.):
        for ratio in (1.,2.):
            for angle in (0.,np.pi/4,np.pi/2,3*np.pi/4):
                u=x*np.cos(angle)+y*np.sin(angle);v=-x*np.sin(angle)+y*np.cos(angle)
                bank.append(normalized(np.exp(-.5*((u/sigma)**2+(v/(sigma*ratio))**2))))
    return bank


def search_evidence(field,noise,predicted,templates,existing,cfg,maxima,
                    separation_check=None,brightness_prediction=None,candidate_check=None):
    """Try every measured local maximum in the prediction window, not just one."""
    radius=cfg['search_radius_raw_px']/2
    lo=np.maximum(np.floor(predicted-radius).astype(int),0)
    hi=np.minimum(np.ceil(predicted+radius).astype(int)+1,field.shape[::-1])
    yy,xx=np.where(maxima[lo[1]:hi[1],lo[0]:hi[0]])
    candidates=[np.array([x+lo[0],y+lo[1]],float) for x,y in zip(xx,yy)
                if np.hypot(x+lo[0]-predicted[0],y+lo[1]-predicted[1])<=radius]
    candidates.sort(key=lambda q:-field[int(q[1]),int(q[0])])
    attempts=[];accepted=[]
    for q in candidates:
        found,e=local_evidence(field,noise,q,templates,existing if separation_check is None and candidate_check is None else [],[],{**cfg,'search_radius_raw_px':0.},maxima)
        e.update(local_maximum_sample_xy=q.tolist(),prediction_error_raw_px=float(np.linalg.norm(q-predicted)*2))
        if found is not None and e['prediction_error_raw_px']>cfg['maximum_error_raw_px']:
            found=None;e['status']='POSITION_MISMATCH'
        if found is not None and candidate_check is None and separation_check is not None and not separation_check(found):
            found=None;e['status']='ALREADY_DETECTED_OR_UNRESOLVED'
        if found is not None and candidate_check is not None:
            accepted_candidate,status=candidate_check(found)
            if not accepted_candidate:found=None;e['status']=status
        score=(e['prediction_error_raw_px']/max(cfg['maximum_error_raw_px'],1e-9))**2
        score+=max(0.,1-e.get('diffraction_shape_NCC',0.))
        if brightness_prediction and brightness_prediction.get('predicted_contrast_DN',0)>0 and e.get('contrast_DN',0)>0:
            expected=brightness_prediction['predicted_contrast_DN']
            width=max(np.log(8.),np.log(brightness_prediction['upper_contrast_DN']/expected))
            score+=(np.log(e['contrast_DN']/expected)/width)**2
            e['brightness_prediction']=brightness_prediction
        e['joint_position_shape_brightness_score']=float(score)
        attempts.append(e)
        if found is not None:accepted.append((score,found,e.copy()))
    if accepted:
        _,found,e=min(accepted,key=lambda item:item[0])
        return found,{**e,'attempts':attempts,'selection':'MINIMUM_JOINT_EVIDENCE_SCORE'}
    return None,dict(status='NO_ACCEPTED_LOCAL_MAXIMUM',attempts=attempts)


def geometry_models(rawpoints, amplitudes, H, cfg):
    points = transform(rawpoints, H)
    anchor = int(np.argmax(amplitudes))
    models = []
    lattice = fit_lattice(points, anchor, cfg['lattice'])
    row = fit_row(points, anchor, cfg['row'])
    for kind, model in [('lattice_2d', lattice), ('row', row)]:
        if model is None:
            models.append(dict(kind=kind, coordinate_unit='screen_mm', status='INSUFFICIENT_OR_INCONSISTENT_GEOMETRY'))
            continue
        members = np.flatnonzero(model['inliers'])
        # Fixed observed labels: conditional leave-one-position check, not blind indexing.
        ix = np.asarray(model['indices'])[model['inliers']] if kind == 'lattice_2d' else np.asarray(model['indices'])
        if kind == 'lattice_2d':
            M = np.c_[np.ones(len(ix)), ix]
        else:
            M = np.c_[np.ones(len(ix)), ix, ix*abs(ix), abs(ix)] if model['curve_term_supported'] else np.c_[np.ones(len(ix)), ix, abs(ix)]
        errors = []
        for i in range(len(members)):
            train = np.arange(len(members)) != i
            if np.linalg.matrix_rank(M[train]) == M.shape[1]:
                pred = M[i] @ np.linalg.lstsq(M[train], points[members[train]], rcond=None)[0]
                errors.append(float(np.linalg.norm(pred-points[members[i]])))
        fraction=max(errors)/model['spacing'] if len(errors)==len(members) else None
        enabled=fraction is not None and fraction<=cfg.get('maximum_conditional_loo_fraction',.2)
        models.append(dict(kind=kind, coordinate_unit='screen_mm', status='OBSERVED_GEOMETRY', model=model,
                           conditional_loo_error_mm=errors, maximum_conditional_loo_fraction=fraction,
                           prediction_enabled=enabled,
                           prediction_validation='CONDITIONAL_POSITION_CHECK' if enabled else 'UNSTABLE_CONDITIONAL_PREDICTION',
                           input_raw_points=np.asarray(rawpoints).tolist()))
    return models


def candidate_status(flags):
    """Image-evidence state, deliberately separate from physical peak identity."""
    if any(f in flags for f in ('BOUNDARY_OR_OCCLUSION_REVIEW', 'UNRESOLVED_SHALLOW_SADDLE', 'OFF_MODEL_REVIEW', 'LOW_LOCAL_SIGNIFICANCE','REGIONAL_BRIGHTNESS_OUTLIER','TRAINING_QUARANTINE','LOCALIZATION_CHANNEL_DISAGREEMENT','EDGE_PROFILE_UNRESOLVED')):
        return 'REVIEW_REQUIRED'
    return 'WEAK_SIGNAL_CANDIDATE' if 'WEAK_SIGNAL' in flags else 'SIGNAL_CANDIDATE'


def screen_contrast_fields(sub, valid, scales, mask_edges=False):
    """Exclude the external background before smoothing near the screen edge.

    Normalized convolution avoids turning the screen/background step into a
    row of apparent weak spots. Values and detection thresholds remain in DN.
    """
    cache={}
    weight=np.asarray(valid,np.float32)
    def smooth(sigma):
        if sigma not in cache:
            if mask_edges:
                cache[sigma]=ndi.gaussian_filter(sub*weight,sigma)/np.maximum(
                    ndi.gaussian_filter(weight,sigma),1e-6)
            else:cache[sigma]=ndi.gaussian_filter(sub,sigma)
        return cache[sigma]
    return {sigma:(smooth(sigma)-smooth(max(35.,4*sigma)),smooth(sigma)-smooth(2*sigma))
            for sigma in scales}


def brightness_diagnostics(peaks, H):
    """Leave-one-candidate-out empirical brightness prediction, not deletion.

    Nearby radial peers allow a broad, anisotropic uncertainty interval. The
    central saturated reference is never used to infer a linear brightness.
    An isolated bright point is a review proposal; real diffraction orders can
    have nonmonotonic intensities and must not be rejected from this alone.
    """
    if len(peaks)<6:return
    raw=np.array([[p['u_raw_px'],p['v_raw_px']] for p in peaks])
    xy=transform(raw,H);amps=np.array([max(p['detection_contrast_DN'],1e-6) for p in peaks])
    anchor=int(np.argmax(amps));radial=np.linalg.norm(xy-xy[anchor],axis=1)
    pair=np.linalg.norm(xy[:,None]-xy,axis=2);np.fill_diagonal(pair,np.inf)
    spacing=float(np.median(pair.min(axis=1)))
    for i,p in enumerate(peaks):
        eligible=np.array([j!=i and j!=anchor and q['edge_distance_raw_px']>=180
            and 'LOW_LOCAL_SIGNIFICANCE' not in q['review_flags'] for j,q in enumerate(peaks)])
        same_shell=eligible & (abs(radial-radial[i])<=max(1.5,.4*spacing))
        j=np.flatnonzero(same_shell)
        result=dict(method='LEAVE_ONE_OUT_RADIAL_PEERS',status='INSUFFICIENT_OBSERVED_PEERS',
            radius_mm=float(radial[i]),peer_count=len(j),observed_contrast_DN=float(amps[i]))
        if i!=anchor and len(j)>=4:
            expected=float(np.median(amps[j]));upper=max(8*expected,3*float(np.percentile(amps[j],90)))
            result.update(status='BRIGHTNESS_OUTLIER_REVIEW' if amps[i]>upper else 'WITHIN_BROAD_EMPIRICAL_RANGE',
                predicted_contrast_DN=expected,upper_contrast_DN=upper,
                observed_to_predicted_ratio=float(amps[i]/expected),
                peer_raw_xy=raw[j].tolist(),interpretation='image-contrast prediction only; not physical power or a deletion rule')
        p['brightness_prediction']=result


def _detect_evidence(red, corners, H, cfg, exclusions=(), green=None,nonlinear_mask=None,training_quarantine=()):
    """Local detection at two scales with Euclidean suppression and complete audit."""
    corners, H = np.asarray(corners, float), np.asarray(H, float)
    lo = np.maximum(np.floor(corners.min(0)/2).astype(int)-100, 0)
    hi = np.minimum(np.ceil(corners.max(0)/2).astype(int)+100, red.shape[::-1])
    sub = red[lo[1]:hi[1], lo[0]:hi[0]]
    local_corners = corners/2-lo
    valid = np.zeros(sub.shape, np.uint8)
    cv2.fillConvexPoly(valid, np.rint(local_corners).astype(np.int32), 1)
    fields=screen_contrast_fields(sub,valid,cfg['scales_red_samples'],cfg.get('mask_external_screen_background',False))
    log_values('detector_configuration',cfg=cfg,corners=corners,H=H,exclusions=exclusions,lo=lo,hi=hi)
    emit('detection.contrast', fields=fields, valid=valid, lo=lo)
    green_fields=(screen_contrast_fields(green[lo[1]:hi[1],lo[0]:hi[0]],valid,
        cfg['scales_red_samples'],cfg.get('mask_external_screen_background',False)) if green is not None else None)
    color_protected=(ndi.maximum_filter(np.asarray(nonlinear_mask[lo[1]:hi[1],lo[0]:hi[0]],bool),size=61)
                     if nonlinear_mask is not None else None)
    contrast, noise = fields[4]
    yy,xx=np.mgrid[-70:71,-70:71]
    noise_annulus=(np.hypot(xx,yy)>=40)&(np.hypot(xx,yy)<=70)
    points, audit = [], []
    maximum_masks={s:f[0]==ndi.maximum_filter(f[0],size=11) for s,f in fields.items()}
    coarse=fields[max(fields)][0]
    cy,cx=np.where(maximum_masks[max(fields)] & (valid.astype(bool) if cfg.get('mask_external_screen_background',False) else True))
    coarse_positions=np.c_[cx,cy]
    coarse_tree=cKDTree(coarse_positions)
    green_sub=green[lo[1]:hi[1],lo[0]:hi[0]] if green is not None else None
    core_near_white=(ndi.maximum_filter(np.asarray(nonlinear_mask[lo[1]:hi[1],lo[0]:hi[0]],bool),size=7)
                     if nonlinear_mask is not None else None)
    def measured_center(q,radius=32.5):
        x,y=np.clip(np.rint(q).astype(int),[0,0],np.array(sub.shape[::-1])-1)
        if (cfg.get('saturation_aware_localization',False) and green_sub is not None
                and core_near_white is not None and core_near_white[y,x]):
            distance,index=coarse_tree.query(q)
            if distance*2<=40:
                return near_white_localize(sub,green_sub,q,coarse_positions[index].astype(float),radius)
        return localize(sub,q,radius)
    mode_positions={s:np.c_[np.where(mask)[1],np.where(mask)[0]] for s,mask in maximum_masks.items()}
    mode_trees={s:cKDTree(v) for s,v in mode_positions.items()}
    modern=cfg.get('resolution_aware_separation',False)
    polygon = corners.astype(np.float32)
    def edge_distance(q):
        return float(cv2.pointPolygonTest(polygon, tuple(map(float, (q+lo)*2)), True))
    def excluded(q):
        raw=(q+lo)*2
        for e in exclusions:
            end=np.array([e['u_raw_px'],e['v_raw_px']])
            start=np.asarray(e.get('segment_start_raw_xy',end))
            delta=end-start
            t=np.clip((raw-start)@delta/max(delta@delta,1e-12),0,1)
            if np.linalg.norm(raw-(start+t*delta)) <= e.get('radius_raw_px',45):
                audit.append(dict(status='CONFIRMED_NON_OPTICAL_REVIEW',raw_xy=raw.tolist(),review=e))
                return True
        return False
    def top_bottom_distance(q):
        raw=(q+lo)*2
        return min(abs((b-a)[0]*(raw-a)[1]-(b-a)[1]*(raw-a)[0])/np.linalg.norm(b-a)
                   for a,b in [(corners[0],corners[1]),(corners[3],corners[2])])
    def distinct_position(q,phase):
        for p in points:
            if np.linalg.norm(q-p['q'])*2>=cfg['minimum_separation_raw_px']:continue
            accepted,evidence=(resolved_pair(q,p['q'],fields,mode_trees,mode_positions,cfg,valid)
                               if modern else (False,{}))
            if modern:audit.append(dict(origin=phase,raw_xy=((q+lo)*2).tolist(),
                other_raw_xy=((p['q']+lo)*2).tolist(),resolution_evidence=evidence,
                status='RESOLVED_CLOSE_PAIR' if accepted else 'SAME_OR_UNRESOLVED_MODE'))
            if not accepted:return False
        return True
    def background_evidence(q):
        if green_fields is None or not cfg.get('cfa_background_prediction',False):return {}
        x,y=np.clip(np.rint(q).astype(int),[0,0],np.array(sub.shape[::-1])-1)
        if color_protected is not None and color_protected[y,x]:
            return dict(method='NATIVE_CFA_NEAR_WHITE_GUARD',status='RADIOMETRIC_NONLINEARITY_NO_COLOR_REJECTION',
                interpretation='near-white/saturated native samples cannot establish a linear R/G background')
        return cfa_background_evidence(q,contrast,green_fields[4][0],green_fields[4][1],valid,cfg.get('cfa_background',{}))
    def is_broadband(q,phase):
        evidence=background_evidence(q)
        if evidence.get('status')=='BROADBAND_BACKGROUND_EXPLAINS_CANDIDATE':
            audit.append(dict(status='MEASURED_BROADBAND_BACKGROUND',origin=phase,
                raw_xy=((q+lo)*2).tolist(),cfa_background=evidence))
            return True,evidence
        return False,evidence
    def is_single_cfa_impulse(q,phase):
        if not cfg.get('single_cfa_impulse_rejection',False):return False
        evidence=single_cfa_impulse_evidence(q,sub,fields,valid,cfg['guided']['minimum_contrast_DN'],cfg.get('mask_external_screen_background',False))
        if evidence['status']=='SINGLE_CFA_IMPULSE_WITHOUT_SPATIAL_SPOT_SUPPORT':
            evidence['impulse_pixel_raw_xy']=((np.asarray(evidence['impulse_pixel_sample_xy'])+lo)*2).tolist()
            audit.append(dict(status=evidence['status'],origin=phase,raw_xy=((q+lo)*2).tolist(),native_impulse=evidence))
            return True
        return False
    def is_edge_ridge(q,phase):
        if not cfg.get('edge_ridge_rejection',False):return False
        value=float(ndi.map_coordinates(contrast,[[q[1]],[q[0]]],order=1)[0])
        if value>=cfg['strong_minimum_DN']:return False
        evidence=boundary_ridge_evidence(q,coarse,local_corners,
            cfg.get('weak_edge_ridge_distance_raw_px',80.),cfg.get('minimum_edge_tangential_prominence',.15))
        if evidence['status']=='SCREEN_EDGE_RIDGE_WITHOUT_RESOLVED_TANGENTIAL_PEAK':
            audit.append(dict(status=evidence['status'],origin=phase,raw_xy=((q+lo)*2).tolist(),edge_profile=evidence))
            return True
        return False
    def mark_resolution(phase):
        ordered=sorted(points,key=lambda p:-float(ndi.map_coordinates(coarse,[[p['q'][1]],[p['q'][0]]],order=1)[0]))
        distinct=[]
        for p in ordered:
            q=p['q'];v=float(ndi.map_coordinates(coarse,[[q[1]],[q[0]]],order=1)[0])
            duplicate=None
            for other in distinct:
                if np.linalg.norm(q-other['q'])*2>cfg.get('saddle_check_radius_raw_px',300):continue
                if modern:
                    saddle=connected_saddle(coarse,q,other['q'],max(fields),valid)
                    prominence=saddle['saddle_fraction']
                else:
                    line=q+(other['q']-q)*np.linspace(0,1,41)[:,None]
                    values=ndi.map_coordinates(coarse,[line[:,1],line[:,0]],order=1)
                    prominence=float((v-values.min())/max(abs(v),1e-8));saddle={}
                if prominence<cfg.get('minimum_saddle_fraction',.15):
                    duplicate=dict(status='UNRESOLVED_SHALLOW_SADDLE',origin=p['origin'],phase=phase,
                        raw_xy=((q+lo)*2).tolist(),stronger_raw_xy=((other['q']+lo)*2).tolist(),
                        saddle_fraction=prominence,connected_saddle=saddle)
                    break
            if duplicate:
                if 'UNRESOLVED_SHALLOW_SADDLE' not in p['review_flags']:p['review_flags'].append('UNRESOLVED_SHALLOW_SADDLE')
                p['evidence'].update(unresolved_stronger_raw_xy=duplicate['stronger_raw_xy'],saddle_fraction=duplicate['saddle_fraction'])
                audit.append(duplicate)
            else:distinct.append(p)
    for amp, q in maxima_candidates(contrast, cfg['strong_minimum_DN'], valid):
        original_seed=q.copy()
        coarse_distance,coarse_index=coarse_tree.query(q)
        if coarse_distance*2>cfg.get('strong_maximum_scale_shift_raw_px',80):
            audit.append(dict(origin='independent_strong',status='FINE_TEXTURE_WITHOUT_COARSE_PEAK',raw_xy=((q+lo)*2).tolist(),amplitude_DN=amp))
            continue
        # One broad optical envelope can contain several fine speckle maxima.
        # Consolidate them at their common coarser maximum before Euclidean NMS.
        if coarse_distance*2>cfg['maximum_scale_shift_raw_px']:
            q=coarse_positions[coarse_index].astype(float)
        center, shape = measured_center(q)
        shape.update(seed_raw_xy=((original_seed+lo)*2).tolist(),seed_contrast_DN=amp,
                     coarse_maximum_distance_raw_px=float(coarse_distance*2),
                     localization_seed_raw_xy=((q+lo)*2).tolist(),
                     localized_contrast_DN=float(ndi.map_coordinates(contrast,[[center[1]],[center[0]]],order=1)[0]))
        if edge_distance(center) < cfg.get('strong_edge_margin_raw_px',80) or top_bottom_distance(center) < cfg['strong_top_bottom_margin_raw_px']:
            audit.append(dict(origin='independent_strong', status='SCREEN_EDGE', raw_xy=((center+lo)*2).tolist(), amplitude_DN=amp))
            continue
        if excluded(center) or not distinct_position(center,'independent_strong'):
            continue
        broadband,color_evidence=is_broadband(center,'independent_strong')
        if broadband:continue
        if is_single_cfa_impulse(center,'independent_strong') or is_edge_ridge(center,'independent_strong'):continue
        if color_evidence:shape['cfa_background']=color_evidence
        n=patch(noise,center,70)[noise_annulus]
        local_noise=max(.25,1.4826*float(np.median(abs(n-np.median(n)))))
        significance=shape['localized_contrast_DN']/local_noise
        shape.update(local_noise_DN=local_noise,contrast_to_local_MAD_ratio=significance)
        flags=[]
        if shape.get('localization_channel_disagreement'):flags.append('LOCALIZATION_CHANNEL_DISAGREEMENT')
        if significance<cfg['guided']['minimum_significance']:flags.append('LOW_LOCAL_SIGNIFICANCE')
        if shape['localized_contrast_DN']<cfg['strong_minimum_DN']:flags.append('WEAK_SIGNAL')
        points.append(dict(q=center, amplitude_DN=shape['localized_contrast_DN'], origin='INDEPENDENT_STRONG', scale=4,
                           evidence=shape, groups=[], review_flags=flags))
    emit('detection.strong', points=points, lo=lo)
    def independent_weak_search(templates,phase):
        for sigma,(field,nfield) in fields.items():
            candidates=maxima_candidates(field,cfg['guided']['minimum_contrast_DN'],valid)
            empirical=bool(templates.get(sigma,[]))
            generic=analytic_templates()
            for amp,q0 in candidates:
                if not modern and (edge_distance(q0)<cfg['weak_edge_margin_raw_px'] or not distinct_position(q0,phase)):continue
                record=dict(origin=phase,scale_red_samples=sigma,raw_xy=((q0+lo)*2).tolist())
                persistence=float(coarse_tree.query(q0)[0]*2)
                if persistence>cfg['maximum_scale_shift_raw_px']:
                    record.update(status='SCALE_UNSTABLE_REVIEW',coarse_maximum_distance_raw_px=persistence)
                    audit.append(record);continue
                # A different observed spot shape must not disable the generic
                # independently declared bank. Each bank keeps its own gate.
                banks=[]
                if empirical:banks.append(('INDEPENDENT_OBSERVED',templates[sigma],cfg['guided']))
                banks.append(('ANALYTIC_GAUSSIAN_PRIOR',generic,{**cfg['guided'],
                    'minimum_significance':cfg.get('generic_template_minimum_significance',12),
                    'minimum_diffraction_NCC':cfg.get('generic_template_minimum_NCC',.8)}))
                trials=[];q=None;evidence={}
                for basis,bank,settings in banks:
                    found,e=local_evidence(field,nfield,q0,bank,[] if modern else [p['q'] for p in points],[],
                        {**settings,'search_radius_raw_px':0.},maximum_masks[sigma])
                    trials.append({**e,'template_basis':basis})
                    if found is not None:q=found;evidence=e;record['template_basis']=basis;break
                record.update(evidence if evidence else trials[-1]);record['template_trials']=trials
                if q is not None:
                    center,shape=measured_center(q,22.5)
                    if excluded(center):record['status']='CONFIRMED_NON_OPTICAL_REVIEW'
                    elif not distinct_position(center,phase):record['status']='ALREADY_DETECTED_AFTER_LOCALIZATION'
                    elif edge_distance(center)<cfg['weak_edge_margin_raw_px']:record['status']='SCREEN_EDGE'
                    else:
                        broadband,color_evidence=is_broadband(center,phase)
                        if broadband:record['status']='MEASURED_BROADBAND_BACKGROUND'
                        elif is_single_cfa_impulse(center,phase):record['status']='SINGLE_CFA_IMPULSE_WITHOUT_SPATIAL_SPOT_SUPPORT'
                        elif is_edge_ridge(center,phase):record['status']='SCREEN_EDGE_RIDGE_WITHOUT_RESOLVED_TANGENTIAL_PEAK'
                        else:
                            points.append(dict(q=center,amplitude_DN=evidence['contrast_DN'],origin='INDEPENDENT_WEAK',scale=sigma,
                                evidence={**evidence,**shape,'coarse_maximum_distance_raw_px':persistence,
                                    'template_basis':record['template_basis'],'cfa_background':color_evidence},
                                groups=[],review_flags=['WEAK_SIGNAL','OFF_MODEL_REVIEW']))
                            record['measured_raw_xy']=((center+lo)*2).tolist()
                log_values('candidate_audit',record=record)
                audit.append(record)
    if cfg.get('weak_context_before_training',False):independent_weak_search({},'independent_weak_before_training')
    emit('detection.weak', points=points, lo=lo)
    for p in points:
        raw=(p['q']+lo)*2
        inherited=[e for e in training_quarantine if np.linalg.norm(raw-np.asarray(e['raw_xy']))<1e-5]
        if inherited:
            p['review_flags'].append('TRAINING_QUARANTINE')
            p['review_flags'].extend(f for e in inherited for f in e['reasons'] if f not in p['review_flags'])
            p['evidence']['training_quarantine']=inherited
    if modern:mark_resolution('BEFORE_MODEL_AND_TEMPLATE_TRAINING')
    if cfg.get('contextual_brightness_prediction',False):
        raw=np.array([(p['q']+lo)*2 for p in points])
        predictions=contextual_brightness(raw,[p['amplitude_DN'] for p in points],H,
            [edge_distance(p['q'])>=50 and not any(f in p['review_flags'] for f in
                ('LOW_LOCAL_SIGNIFICANCE','UNRESOLVED_SHALLOW_SADDLE','REGIONAL_BRIGHTNESS_OUTLIER')) for p in points],
            bilateral_support=cfg.get('brightness_bilateral_support',False))
        for p,prediction in zip(points,predictions):
            p['brightness_prediction']=prediction
            if prediction['status']=='REGIONAL_BRIGHTNESS_OUTLIER':p['review_flags'].append('REGIONAL_BRIGHTNESS_OUTLIER')
    # Boundary/occlusion transitions may be very bright. They remain visible as
    # candidates, but cannot teach either the lattice or the optical template.
    eligible=[i for i,p in enumerate(points) if not p['review_flags'] and edge_distance(p['q'])>=cfg.get('boundary_review_distance_raw_px',180)]
    rawpoints=np.array([(points[i]['q']+lo)*2 for i in eligible])
    models = geometry_models(rawpoints, [points[i]['amplitude_DN'] for i in eligible], H, cfg) if eligible else []
    anchor = int(np.argmax([p['amplitude_DN'] for p in points])) if points else None
    def shape_eligible(i):
        p=points[i];x,y=np.rint(p['q']).astype(int)
        return (i in eligible and i!=anchor and p['amplitude_DN']>=cfg['template_minimum_DN']
                and not (color_protected is not None and color_protected[y,x]))
    shape_inputs=[i for i in eligible if shape_eligible(i)]
    templates = {s:[normalized(patch(f[0], p['q'])) for i,p in enumerate(points)
                   if i in shape_inputs]
                 for s,f in fields.items()}
    audit.append(dict(status='MODEL_AND_TEMPLATE_TRAINING_INPUTS',
        eligible_raw_xy=rawpoints.tolist(),template_raw_xy=[((points[i]['q']+lo)*2).tolist()
            for i in shape_inputs],near_white_candidates_are_shape_templates=False))
    emit('detection.models', models=models, training=audit[-1], points=points, lo=lo)
    invH=np.linalg.inv(H)
    if cfg.get('weak_context_before_training',False):independent_weak_search(templates,'independent_weak_with_observed_templates')
    def brightness_at_site(q):
        if not cfg.get('contextual_brightness_prediction',False) or not points:return None
        raw=[(p['q']+lo)*2 for p in points]+[(q+lo)*2]
        use=[p['origin'].startswith('INDEPENDENT') and edge_distance(p['q'])>=50
            and np.linalg.norm(p['q']-q)*2>cfg['guided']['search_radius_raw_px']
            and 'LOW_LOCAL_SIGNIFICANCE' not in p['review_flags']
            and 'UNRESOLVED_SHALLOW_SADDLE' not in p['review_flags']
            and 'REGIONAL_BRIGHTNESS_OUTLIER' not in p['review_flags'] for p in points]+[False]
        prediction=contextual_brightness(raw,[p['amplitude_DN'] for p in points]+[1.],H,use,
            bilateral_support=cfg.get('brightness_bilateral_support',False))[-1]
        if 'predicted_contrast_DN' not in prediction:return None
        for key in ('observed_contrast_DN','observed_to_predicted_ratio'):prediction.pop(key,None)
        prediction['status']='SEARCH_PRIOR_FROM_OTHER_MEASURED_LOCATIONS'
        return prediction
    def guided_candidate_check(q):
        center,_=measured_center(q,22.5)
        if excluded(center):return False,'CONFIRMED_NON_OPTICAL_REVIEW'
        if edge_distance(center)<cfg['weak_edge_margin_raw_px']:return False,'SCREEN_EDGE_AFTER_LOCALIZATION'
        if not distinct_position(center,'guided_search_localized'):return False,'ALREADY_DETECTED_AFTER_LOCALIZATION'
        if is_broadband(center,'guided_search_localized')[0]:return False,'MEASURED_BROADBAND_BACKGROUND'
        if is_single_cfa_impulse(center,'guided_search_localized'):return False,'SINGLE_CFA_IMPULSE_WITHOUT_SPATIAL_SPOT_SUPPORT'
        if is_edge_ridge(center,'guided_search_localized'):return False,'SCREEN_EDGE_RIDGE_WITHOUT_RESOLVED_TANGENTIAL_PEAK'
        return True,'DETECTED'
    for entry in models:
        if entry['status'] != 'OBSERVED_GEOMETRY' or not entry['prediction_enabled']:
            continue
        model,kind=entry['model'],entry['kind']
        members=np.asarray(eligible)[np.flatnonzero(model['inliers'])]
        for i in members:
            points[i]['groups'].append(kind)
        model_templates=[normalized(patch(contrast, points[i]['q'])) for i in members if i in shape_inputs]
        for m,n,site in prediction_sites(model,kind):
            qpred=transform([site],invH)[0]/2-lo
            record=dict(origin='guided_'+kind, index_m=m, index_n=n,
                        predicted_screen_mm=site.tolist(), predicted_raw_xy=((qpred+lo)*2).tolist())
            if not (30<=qpred[0]<sub.shape[1]-30 and 30<=qpred[1]<sub.shape[0]-30) or edge_distance(qpred)<cfg['weak_edge_margin_raw_px']:
                record['status']='SCREEN_EDGE_OR_OUTSIDE'; audit.append(record); continue
            if not model_templates:
                record['status']='NO_INDEPENDENT_SHAPE_TEMPLATE'; audit.append(record); continue
            q, evidence=search_evidence(contrast, noise, qpred, model_templates,
                                       [p['q'] for p in points], cfg['guided'], maximum_masks[4],
                                       separation_check=(lambda q:distinct_position(q,'guided_search')) if modern else None,
                                       brightness_prediction=brightness_at_site(qpred),candidate_check=guided_candidate_check)
            record.update(evidence)
            if q is not None:
                center,shape=measured_center(q,22.5)
                if excluded(center):
                    record['status']='CONFIRMED_NON_OPTICAL_REVIEW'
                elif edge_distance(center)<cfg['weak_edge_margin_raw_px']:
                    record['status']='SCREEN_EDGE'
                elif not distinct_position(center,'guided_localization'):
                    record['status']='ALREADY_DETECTED_AFTER_LOCALIZATION'
                elif is_broadband(center,'guided')[0]:record['status']='MEASURED_BROADBAND_BACKGROUND'
                else:
                    points.append(dict(q=center,amplitude_DN=evidence['contrast_DN'],origin='GUIDED_SCREEN_'+kind.upper(),
                                       scale=4,evidence={**evidence,**shape},groups=[kind],review_flags=['WEAK_SIGNAL']))
                    record['measured_raw_xy']=((center+lo)*2).tolist()
            log_values('candidate_audit',record=record)
            audit.append(record)
    emit('detection.guided', points=points, lo=lo, audit=audit, models=models)
    # Model-free weak search also runs when no row/lattice was identifiable.
    for sigma,(field,nfield) in ([] if cfg.get('weak_context_before_training',False) else fields.items()):
        candidates=maxima_candidates(field,cfg['guided']['minimum_contrast_DN'],valid)
        empirical=bool(templates[sigma])
        template_bank=templates[sigma] if empirical else analytic_templates()
        for amp,q0 in candidates:
            if edge_distance(q0)<cfg['weak_edge_margin_raw_px'] or not distinct_position(q0,'independent_weak'):
                continue
            record=dict(origin='independent_weak',scale_red_samples=sigma,raw_xy=((q0+lo)*2).tolist())
            persistence=float(coarse_tree.query(q0)[0]*2)
            if persistence>cfg['maximum_scale_shift_raw_px']:
                record.update(status='SCALE_UNSTABLE_REVIEW',coarse_maximum_distance_raw_px=persistence)
                audit.append(record);continue
            weak_cfg=cfg['guided'] if empirical else {**cfg['guided'],
                'minimum_significance':cfg.get('generic_template_minimum_significance',12),
                'minimum_diffraction_NCC':cfg.get('generic_template_minimum_NCC',.8)}
            record['template_basis']='INDEPENDENT_OBSERVED' if empirical else 'ANALYTIC_GAUSSIAN_PRIOR'
            q,evidence=local_evidence(field,nfield,q0,template_bank,
                                      [p['q'] for p in points],[],{**weak_cfg,'search_radius_raw_px':0.},maximum_masks[sigma])
            record.update(evidence)
            if q is not None:
                persistence=float(coarse_tree.query(q)[0]*2)
                record['coarse_maximum_distance_raw_px']=persistence
                if persistence>cfg['maximum_scale_shift_raw_px']:
                    record['status']='SCALE_UNSTABLE_REVIEW'
                else:
                    center,shape=measured_center(q,22.5)
                    if excluded(center):
                        record['status']='CONFIRMED_NON_OPTICAL_REVIEW'
                    elif not distinct_position(center,'independent_weak_localization'):
                        record['status']='ALREADY_DETECTED_AFTER_LOCALIZATION'
                    elif edge_distance(center)<cfg['weak_edge_margin_raw_px']:
                        record['status']='SCREEN_EDGE'
                    else:
                        points.append(dict(q=center,amplitude_DN=evidence['contrast_DN'],origin='INDEPENDENT_WEAK',scale=sigma,
                                           evidence={**evidence,**shape, 'coarse_maximum_distance_raw_px':persistence,'template_basis':record['template_basis']},
                                           groups=[],review_flags=['WEAK_SIGNAL','OFF_MODEL_REVIEW']))
                        record['measured_raw_xy']=((center+lo)*2).tolist()
            log_values('candidate_audit',record=record)
            audit.append(record)
    # A shallow saddle cannot resolve whether this is one spot or an overlap.
    # Preserve both measured maxima visibly; do not turn a resolution limit into
    # a claim of physical identity, or silently delete the weaker maximum.
    mark_resolution('FINAL_MEASURED_CANDIDATES')
    peaks=[]
    for p in points:
        raw=(p['q']+lo)*2
        if edge_distance(p['q'])<cfg.get('boundary_review_distance_raw_px',180):
            p['review_flags'].append('BOUNDARY_OR_OCCLUSION_REVIEW')
        if cfg.get('edge_ridge_diagnostic',False) and float(ndi.map_coordinates(contrast,[[p['q'][1]],[p['q'][0]]],order=1)[0])<cfg['strong_minimum_DN']:
            edge_profile=boundary_ridge_evidence(p['q'],coarse,local_corners,
                cfg.get('weak_edge_ridge_distance_raw_px',80.),cfg.get('minimum_edge_tangential_prominence',.15))
            if edge_profile['status']!='NOT_NEAR_BOUNDARY':
                p['evidence']['boundary_tangential_profile']=edge_profile
                if edge_profile['status']=='SCREEN_EDGE_RIDGE_WITHOUT_RESOLVED_TANGENTIAL_PEAK':
                    p['review_flags'].append('EDGE_PROFILE_UNRESOLVED')
                    audit.append(dict(status='UNRESOLVED_EDGE_PROFILE_CANDIDATE_RETAINED',raw_xy=raw.tolist(),
                        edge_profile=edge_profile,interpretation='a shallow along-edge profile cannot distinguish a real overlapping optical wing from a boundary background'))
        peaks.append(dict(u_raw_px=float(raw[0]),v_raw_px=float(raw[1]),detection_contrast_DN=p['amplitude_DN'],
                          detection_origin=p['origin'],scale_red_samples=p['scale'],evidence=p['evidence'],
                          model_groups=p['groups'],review_flags=p['review_flags'],physical_order=None,
                          candidate_status=candidate_status(p['review_flags']),
                          zero_order_confirmed=False,edge_distance_raw_px=edge_distance(p['q'])))
        if 'brightness_prediction' in p:peaks[-1]['brightness_prediction']=p['brightness_prediction']
    peaks.sort(key=lambda p:(round(p['v_raw_px']/140),p['u_raw_px']))
    if cfg.get('contextual_brightness_prediction',False):
        predictions=contextual_brightness([[p['u_raw_px'],p['v_raw_px']] for p in peaks],
            [p['detection_contrast_DN'] for p in peaks],H,
            [p['detection_origin'].startswith('INDEPENDENT') and p['edge_distance_raw_px']>=50 and not any(f in p['review_flags'] for f in
                ('LOW_LOCAL_SIGNIFICANCE','UNRESOLVED_SHALLOW_SADDLE','REGIONAL_BRIGHTNESS_OUTLIER')) for p in peaks],
            bilateral_support=cfg.get('brightness_bilateral_support',False))
        for p,prediction in zip(peaks,predictions):p['final_regional_brightness_prediction']=prediction
    elif cfg.get('brightness_prediction',False):brightness_diagnostics(peaks,H)
    return peaks,models,audit


def detect_evidence(red, corners, H, cfg, exclusions=(), green=None,nonlinear_mask=None):
    """Rebuild teachers if final independent evidence invalidates an earlier input.

    Quarantine affects prediction/template training only. Measured candidates stay
    in the catalogue; every pass recomputes dependent searches from the RAW array.
    """
    quarantine=[];history=[]
    for iteration in range(cfg.get('maximum_training_closure_passes',5)):
        emit('detection.pass', pass_index=iteration+1)
        peaks,models,audit=_detect_evidence(red,corners,H,cfg,exclusions,green,nonlinear_mask,quarantine)
        if not cfg.get('close_training_evidence',False):return peaks,models,audit
        teacher_records=[e for e in audit if e.get('status')=='MODEL_AND_TEMPLATE_TRAINING_INPUTS']
        teachers=[q for e in teacher_records for q in e['eligible_raw_xy']+e['template_raw_xy']]
        additions=[]
        for p in peaks:
            xy=[p['u_raw_px'],p['v_raw_px']]
            if not any(np.linalg.norm(np.asarray(q)-xy)<1e-5 for q in teachers):continue
            reasons=[f for f in p['review_flags'] if f=='UNRESOLVED_SHALLOW_SADDLE']
            prediction=p.get('final_regional_brightness_prediction',{})
            if prediction.get('status')=='REGIONAL_BRIGHTNESS_OUTLIER':reasons.append('REGIONAL_BRIGHTNESS_OUTLIER')
            if reasons:additions.append(dict(raw_xy=xy,reasons=reasons,final_prediction=prediction))
        history.append(dict(pass_index=iteration+1,training_inputs=teacher_records,new_quarantine=additions))
        emit('detection.closure', passes=history, additions=additions, stable=not additions)
        if not additions:
            audit.append(dict(status='TRAINING_EVIDENCE_CLOSURE_STABLE',passes=history,quarantine=quarantine))
            return peaks,models,audit
        quarantine.extend(additions)
    raise ValueError('Training evidence did not reach a stable quarantine; no complete catalogue may be accepted')
