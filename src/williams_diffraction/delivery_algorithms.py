"""Real-image algorithm experiments, using the full accepted array algorithms.

Diffraction inputs are lossless native CFA-derived arrays. Screen fitting uses
the separately reviewed display image. Saved detections are evaluation-only.
Microscopy experiments recompute frequency families, basis candidates, crops,
tiles and profiles from native decoded pixels. Importing this module does no IO.
"""
from collections import Counter
import csv
import html
from pathlib import Path
import time
import numpy as np
import cv2
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
from .delivery_io import read_json,write_json,new_output,verify_bundle,digest
from .geometry import display_to_raw,raw_to_display,homography,transform
from .screen_audit import refine_screen
from .evidence_detection import detect_evidence,screen_contrast_fields
from .microscopy import spectrum_peaks,normal_profile,axial_frequency_distance
from .microscopy_lattice import select_texture_family,select_texture_pair
from .microscopy_series import texture_measurement
from . import algorithm_graphics as views
from .guide_events import emit


def save_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        if not rows:return
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def final_prediction_diagnostics(peaks,audit,maximum_error):
    """Same final-centroid check as the accepted audit_images.py caller."""
    for p in peaks:
        if not p['detection_origin'].startswith('GUIDED_SCREEN'):continue
        xy=np.array([p['u_raw_px'],p['v_raw_px']])
        matches=[a for a in audit if a.get('status')=='DETECTED' and 'measured_raw_xy' in a
                 and np.linalg.norm(xy-np.array(a['measured_raw_xy']))<1e-8]
        if len(matches)!=1:raise ValueError('Guided peak must have one measured evidence record')
        a=matches[0];error=float(np.linalg.norm(xy-np.array(a['predicted_raw_xy'])))
        a['final_prediction_error_raw_px']=error
        p['evidence'].update(predicted_raw_xy=a['predicted_raw_xy'],final_prediction_error_raw_px=error)
        if error>maximum_error:
            if 'PREDICTION_OFFSET_REVIEW' not in p['review_flags']:p['review_flags'].append('PREDICTION_OFFSET_REVIEW')
            p['candidate_status']='REVIEW_REQUIRED'


def compare_diffraction(core,case,peaks,models,corners,H,out):
    """Only called after the entire detector has returned; never fits to references."""
    ref=read_json(core/case['references'])
    xy=np.array([[p['u_raw_px'],p['v_raw_px']] for p in peaks]);target=np.array([[p['u_raw_px'],p['v_raw_px']] for p in ref['peaks']])
    if not len(xy) or not len(target):raise ValueError('Selected real case unexpectedly has no candidates')
    distance=np.linalg.norm(xy[:,None]-target[None,:],axis=2)
    ri,ci=linear_sum_assignment(distance)
    matched=[dict(run_candidate=peaks[i]['candidate_uid'],reference_uid=ref['peaks'][j]['peak_uid'],distance_raw_px=float(distance[i,j]),same_origin=peaks[i]['detection_origin']==ref['peaks'][j]['detection_origin']) for i,j in zip(ri,ci)]
    save_csv(out/'reference_matches.csv',matched)
    save_csv(out/'historical_human_review.csv',[dict(reference_uid=q['peak_uid'],raw_x=q['raw_xy'][0],raw_y=q['raw_xy'][1],historical_class=q['class'],reason=q['reason']) for q in ref['human_candidate_reviews']])
    maximum=max(q['distance_raw_px'] for q in matched)
    geometry_error=float(np.max(abs(corners-np.asarray(ref['geometry']['corners_display']))))
    # Insufficient models omit the gate in the original algorithm; the compact
    # reference snapshot spells that absent field as null. Both disable search.
    gates={q['kind']:bool(q.get('prediction_enabled',False)) for q in models}
    expected_gates={q['kind']:bool(q.get('prediction_enabled',False)) for q in ref['models']}
    model_statuses={q['kind']:q['status'] for q in models}
    expected_statuses={q['kind']:q['status'] for q in ref['models']}
    success=len(peaks)==ref['n_candidates'] and len(matched)==len(target) and maximum<.05 and all(q['same_origin'] for q in matched) and gates==expected_gates and model_statuses==expected_statuses and geometry_error<1e-5
    result=dict(status='PASS' if success else 'FAIL',scope='regression against accepted v9/v16 coordinates, not detection accuracy against independent human truth',
        reference_loaded_after_detection=True,candidates_found=len(peaks),reference_candidates=ref['n_candidates'],maximum_coordinate_difference_raw_px=maximum,
        maximum_screen_corner_difference_display_px=geometry_error,prediction_gates=gates,reference_prediction_gates=expected_gates,model_statuses=model_statuses,reference_model_statuses=expected_statuses,
        historical_human_review_summary=ref['human_review_summary'],historical_class_counts=dict(Counter(q['class'] for q in ref['human_candidate_reviews'])))
    write_json(out/'reference_check.json',result)
    return result


def diffraction_case(core,item,out,check_reference=True):
    started=time.perf_counter();out.mkdir()
    case=read_json(core/item['case']);folder=core/Path(item['case']).parent
    if digest(folder/case['raw_display'])!=case['physical_visibility_review']['raw_display_sha256']:raise ValueError('Visibility review bound to a different display image')
    with Image.open(folder/case['raw_display']) as source:display=source.convert('RGB')
    seed=np.asarray(case['corners_prior_display'],float)
    for name,xy in case['corner_overrides_display'].items():
        if name in ('TL','TR','BR','BL'):seed[('TL','TR','BR','BL').index(name)]=xy
    print(item['id']+': recomputing held-out screen edges and native-coordinate homography',flush=True)
    review=case['physical_visibility_review']
    corners,edges=refine_screen(np.asarray(display.convert('L')),seed,case['edge_half_widths_display'],method=case['edge_method'],edge_reviews=review['edges'],corner_constraints=review.get('accepted_visible_corner_constraints',{}))
    raw_corners=display_to_raw(corners,case['red_shape']);H=homography(raw_corners,*case['screen_size_mm'])
    corner_mapping_error=float(np.max(abs(transform(raw_corners,H)-np.array([[0,0],[120,0],[120,120],[0,120]]))))
    geometry=dict(seed_display=seed.tolist(),corners_display=corners.tolist(),corners_raw_px=raw_corners.tolist(),H_raw_to_screen_mm=H.tolist(),
                  edge_diagnostics=edges,conditional_edges=[i for i,e in enumerate(edges) if not e['supported']],corner_mapping_error_mm=corner_mapping_error,
                  independent_metric_validation=False,pixel_centers='display resize centers -> R CFA centers -> visible RAW centers; no camera JPEG crop substitution')
    write_json(out/'geometry.json',geometry)
    views.screen_overlay(display,seed,corners,edges,H,case['red_shape']).save(out/'01_screen_geometry.png')
    emit('detection.geometry', geometry=geometry, image=out/'01_screen_geometry.png', red_shape=case['red_shape'], H=H, folder=folder, case=case)
    panels=[]
    for i,e in enumerate(edges):
        train=e.get('selected_train',[]);held=e.get('selected_held_out',[])
        if train or held:
            panels.append(dict(title=f'Edge {i}: {e["status"]}',xlabel='fraction along edge',ylabel='residual (display px)',series=[('train',[q['t'] for q in train],[q['residual_px'] for q in train]),('held out',[q['t'] for q in held],[q['residual_px'] for q in held])]))
        else:
            samples=e.get('samples',[])
            panels.append(dict(title=f'Edge {i}: no supported edge line; gradient candidate counts',xlabel='fraction along edge',ylabel='candidates/profile',series=[('observed gradient candidates',[q['t'] for q in samples],[len(q['candidates']) for q in samples])]))
    views.line_svg(out/'02_edge_validation.svg',panels)
    with np.load(folder/case['input_arrays'],allow_pickle=False) as arrays:
        red=arrays['red'];green=arrays['green_aligned'];mask=arrays['near_white_mask']
    if list(red.shape)!=case['red_shape'] or green.shape!=red.shape or mask.shape!=red.shape or not np.isfinite(red).all() or not np.isfinite(green).all():raise ValueError('Invalid native input arrays')
    cfg=read_json(core/'config/detector_v16.json')['detector']
    print(item['id']+': independent strong/weak search, CFA background, free geometry, guided evidence and teacher closure',flush=True)
    peaks,models,audit=detect_evidence(red,raw_corners,H,cfg,case['reviewed_non_optical_exclusions'],green=green,nonlinear_mask=mask)
    final_prediction_diagnostics(peaks,audit,cfg['guided']['maximum_error_raw_px'])
    xy=np.array([[p['u_raw_px'],p['v_raw_px']] for p in peaks]);mm=transform(xy,H) if len(xy) else np.empty((0,2))
    for i,(p,q) in enumerate(zip(peaks,mm)):
        p.update(candidate_uid=f"run_{item['id']}_p{i+1:03d}",label=f'{i+1:03d}',X_mm=float(q[0]),Y_mm=float(q[1]))
    for model in models:
        if 'input_raw_points' in model:
            model['input_run_candidate_uids']=[peaks[int(np.linalg.norm(xy-point,axis=1).argmin())]['candidate_uid'] if np.linalg.norm(xy-point,axis=1).min()<1e-6 else None for point in np.asarray(model['input_raw_points'])]
            if any(q is None for q in model['input_run_candidate_uids']):raise ValueError('A geometry-training candidate is absent from final catalogue')
    write_json(out/'detected_candidates.json',peaks);write_json(out/'geometry_models.json',models);write_json(out/'all_search_evidence.json',audit)
    save_csv(out/'coordinates.csv',[dict(candidate_uid=p['candidate_uid'],u_raw_px=p['u_raw_px'],v_raw_px=p['v_raw_px'],X_mm=p['X_mm'],Y_mm=p['Y_mm'],origin=p['detection_origin'],contrast_DN=p['detection_contrast_DN'],status=p['candidate_status'],flags=';'.join(p['review_flags']),physical_order='',zero_order_confirmed=False) for p in peaks])
    # These fields are visualized after detection, not substituted for its inputs.
    lo=np.maximum(np.floor(raw_corners.min(0)/2).astype(int)-100,0);hi=np.minimum(np.ceil(raw_corners.max(0)/2).astype(int)+100,red.shape[::-1])
    sub=red[lo[1]:hi[1],lo[0]:hi[0]];valid=np.zeros(sub.shape,np.uint8)
    cv2.fillConvexPoly(valid,np.rint(raw_corners/2-lo).astype(np.int32),1)
    fields=screen_contrast_fields(sub,valid,cfg['scales_red_samples'],cfg.get('mask_external_screen_background',False))
    for sigma,(field,noise) in fields.items():views.contrast_view(field,valid,f'Native R local contrast, sigma={sigma} R samples').save(out/f'03_contrast_sigma{sigma}.png')
    scale=10.;S=np.array([[scale,0,-.5],[0,scale,-.5],[0,0,1.]])
    rect=Image.fromarray(cv2.warpPerspective(views.tone(red),S@H@np.diag([2.,2.,1.]),(1200,1200)))
    views.marked_image(rect,peaks,S@H,case['reviewed_non_optical_exclusions'],only_independent=True).save(out/'04_independent_detection.png')
    proposals=[q for q in audit if 'predicted_raw_xy' in q]
    views.marked_image(rect,peaks,S@H,case['reviewed_non_optical_exclusions'],proposals=proposals).save(out/'05_guided_detection.png')
    views.patches(out/'06_local_evidence.png',red,audit,fields[4][0],lo)
    origin=dict(Counter(p['detection_origin'] for p in peaks));statuses=dict(Counter(q.get('status','UNKNOWN') for q in audit))
    summary=dict(status='COMPUTED',id=item['id'],case_title=case['title'],origin_counts=origin,candidates=len(peaks),with_quality_flags=sum(bool(p['review_flags']) for p in peaks),review_required=sum(p['candidate_status']=='REVIEW_REQUIRED' for p in peaks),candidate_status_counts=dict(Counter(p['candidate_status'] for p in peaks)),
        supplied_human_exclusions=len(case['reviewed_non_optical_exclusions']),conditional_edges=geometry['conditional_edges'],audit_status_counts=statuses,
        model_gates=[{k:q.get(k) for k in ('kind','status','prediction_enabled','maximum_conditional_loo_fraction')} for q in models],
        elapsed_s=time.perf_counter()-started,raw_decoder_used_at_runtime=False,reference_evaluation=None)
    if check_reference:
        summary['reference_evaluation']=compare_diffraction(core,case,peaks,models,corners,H,out)
        summary['status']=summary['reference_evaluation']['status']
    write_json(out/'summary.json',summary)
    emit('detection.final', summary=summary, out=out, audit=audit, peaks=peaks)
    body=f'<p>{html.escape(case["title"])}</p><p>真实原生CFA平面重新计算。{case["voltage_RMS_V"]} V，{case["direction"]}；完整RAW {case["raw_shape"][1]}×{case["raw_shape"][0]}，R平面 {red.shape[1]}×{red.shape[0]}。</p>'
    body+=views.table(['强候选','独立弱候选','几何补检','算法待复核','有任意质量标志','输入的人审排除'],[[origin.get('INDEPENDENT_STRONG',0),origin.get('INDEPENDENT_WEAK',0),sum(v for k,v in origin.items() if k.startswith('GUIDED')),summary['review_required'],summary['with_quality_flags'],summary['supplied_human_exclusions']]])
    if summary['reference_evaluation']:
        body+='<p><b>既有人审口径：</b>'+html.escape(summary['reference_evaluation']['historical_human_review_summary'])+' <a href="historical_human_review.csv">原人审结论表</a>。本次算法标记数与人审限制数不是同一口径；坐标回归通过不表示这些物理身份问题已解决。</p>'
    body+='<p>红叉为已绑定RAW身份的人审非光学位置；算法自动宽带背景抑制、近邻分辨和训练隔离另有逐项审计。候选数不代表独立物理级次。<a href="coordinates.csv">坐标表</a> · <a href="all_search_evidence.json">完整搜索与拒绝证据</a> · <a href="geometry.json">屏边证据</a></p>'
    body+=views.figure('01_screen_geometry.png','① 由屏板边缘建立尺度。灰线是角点先验；蓝网格为新算的10 mm投影；黄边保留可见性条件。衍射点不参与屏板拟合。')
    body+=views.figure('02_edge_validation.png','② 沿边剖面分训练与留出。即使数值直线拟合很好，物理端点不可见时仍保留条件状态。')
    body+='<div class="gallery">'+''.join(views.figure(f'03_contrast_sigma{s}.png',f'③ 多尺度局部对比，尺度{s}个红采样。显示拉伸只用于看图；检测门槛仍为原生DN。') for s in fields)+'</div>'
    body+=views.figure('04_independent_detection.png','④ 独立识别：青圈为强候选，橙圈为通过局部证据的弱候选。近邻需通过跨尺度二维极值与连通鞍点检查。')
    body+=views.table(['几何族','状态','留一最大误差/间距','允许预测'],[['二维点阵' if q['kind']=='lattice_2d' else '横向峰列','可拟合' if q['status']=='OBSERVED_GEOMETRY' else '几何不足或不一致',q.get('maximum_conditional_loo_fraction'),q.get('prediction_enabled',False)] for q in models])
    body+=f'<p>本协议的条件留一误差/间距门槛为 {cfg.get("maximum_conditional_loo_fraction",.2):.3g}。只有通过该门控的族才进入预测搜索。</p>'
    body+=views.figure('05_guided_detection.png','⑤ 几何模型只提出搜索位置。灰方框表示已尝试的位置，紫圈为有实测信号支持的新增候选。留一检验不合格时关闭该族的补检。')
    body+=views.figure('06_local_evidence.png','⑥ 局部搜索证据：可用时展示成功、拒绝和背景窗口。空预测位置保留拒绝状态；补检禁用时明确保留门控结果。误差核对最终质心，各小图显示范围独立标注。')
    body+='<p><a href="summary.json">结果摘要与参考回归</a> · <a href="geometry_models.json">自由基底/横列及预测门控</a></p><p>所有搜索、训练和质心来自输入像素；历史坐标只在计算结束后用于回归比较，不作为识别输入。正视图经过显示映射与插值，不作为测光数据。</p>'
    views.page(out/'index.html',f"衍射实图 {item['id']}：坐标、识别与预测",body)
    print(item['id'],origin,'reference',summary['reference_evaluation']['status'] if check_reference else 'not used',flush=True)
    return summary


def nearest(peaks,target):
    q=np.array([target['fx_cycles_um'],target['fy_cycles_um']])
    p=min(peaks,key=lambda p:axial_frequency_distance([p['fx_cycles_um'],p['fy_cycles_um']],q)) if peaks else None
    error=float(axial_frequency_distance([p['fx_cycles_um'],p['fy_cycles_um']],q)/np.linalg.norm(q)) if p else None
    return p,error


def analyze_texture(image,sx,sy):
    candidates,power,fx,fy=spectrum_peaks(image,sx,sy,max_peaks=16)
    peaks=candidates['peaks'];descriptor,_,radial,angular=texture_measurement(image,sx,sy)
    primary=select_texture_family(peaks,.25);low=select_texture_family(peaks,.12);pair=select_texture_pair(peaks)
    quality_ok=descriptor['status']=='RESOLVED_CANDIDATES'
    ambiguous=False
    if primary and low:
        ratio=primary['base']['period_um']/low['base']['period_um'];ambiguous=max(ratio,1/ratio)>1.5
    elif low:ambiguous=True
    targets=[]
    if primary and quality_ok:targets.append(('family',primary['base']))
    eligible=bool(descriptor['status']=='RESOLVED_CANDIDATES' and pair and pair['coverage']>=.55 and descriptor.get('dominant_axis_fraction',0)<.8)
    if eligible:targets += [('basis_1',pair['first']),('basis_2',pair['second'])]
    emit('texture.family', primary=primary, low=low, ambiguous=ambiguous, quality=descriptor['status'], peaks=peaks)
    h,w=image.shape[:2];checks=[];crop_spectra=[]
    for frac in (.9,.8):
        cw,ch=int(w*frac),int(h*frac);x,y=(w-cw)//2,(h-ch)//2
        found=spectrum_peaks(image[y:y+ch,x:x+cw],sx,sy,max_peaks=16)[0]['peaks']
        variant_primary=select_texture_family(found,.25);variant_low=select_texture_family(found,.12)
        crop_spectra.append(dict(variant=f'crop_{int(frac*100)}',shape_px=[ch,cw],candidates=found,primary_candidate_period_um=variant_primary['base']['period_um'] if variant_primary else None,lower_support_candidate_period_um=variant_low['base']['period_um'] if variant_low else None))
        for role,target in targets:
            p,error=nearest(found,target);checks.append(dict(variant=f'crop_{int(frac*100)}',role=role,matched=error is not None and error<=.10,relative_frequency_difference=error,period_um=p['period_um'] if p else None))
    tiles=[]
    for iy in range(2):
        for ix in range(2):
            patch=image[iy*(h//2):(iy+1)*(h//2),ix*(w//2):(ix+1)*(w//2)]
            found=spectrum_peaks(patch,sx,sy,max_peaks=16)[0]['peaks'];support=[];matches=[]
            for role,target in targets:
                if not role.startswith('basis'):continue
                p,error=nearest(found,target);good=bool(p and error<=.15 and p['power']>=.10*found[0]['power']);support.append(good)
                matches.append(dict(role=role,matched=good,relative_frequency_difference=error,period_um=p['period_um'] if p else None))
            tiles.append(dict(ix=ix,iy=iy,both_modes_supported=(len(support)==2 and all(support)) if eligible else None,matches=matches,candidates=found))
    result=dict(candidates=candidates,quality_descriptor=descriptor,primary_family=primary,lower_support_family=low,family_ambiguous=ambiguous,
                family_passes_quality_gate=quality_ok and primary is not None,reciprocal_basis_candidate=pair,basis_passes_global_screen=eligible,central_crop_checks=checks,central_crop_spectra=crop_spectra,spatial_tiles=tiles,
                spatial_pair_applicable=eligible,jointly_supported_tiles=sum(q['both_modes_supported'] for q in tiles) if eligible else None,
                spatial_pair_supported=sum(q['both_modes_supported'] for q in tiles)>=3 if eligible else None,
                temporal_pixel_checks_performed=False,frequency_unit='cycles/um',sampling_um=[sx,sy],radial_power=radial.tolist(),angular_power=angular.tolist())
    emit('texture.checks', result=result, image=image)
    return result,power,fx,fy,targets


def microscopy_case(core,item,out,own_image=None):
    out.mkdir();source=Path(own_image) if own_image else core/item['image']
    with Image.open(source) as im:
        if im.mode not in ('L','RGB','RGBA'):raise ValueError('Use an 8-bit grayscale/RGB microscopy image; no silent high-bit-depth conversion')
        whole=np.asarray(im.convert('RGB'))
    x,y,w,h=item['roi_xywh']
    if min(x,y)<0 or min(w,h)<64 or x+w>whole.shape[1] or y+h>whole.shape[0]:raise ValueError('ROI must be inside the image and at least 64 pixels per side for four independent FFT tiles')
    roi=whole[y:y+h,x:x+w];sx,sy=item['pixel_scale_um']
    emit('texture.input', whole=whole, roi=roi, item=item)
    result,power,fx,fy,targets=analyze_texture(roi,sx,sy)
    result.update(id=item['id'],title=item['title'],roi_xywh=item['roi_xywh'],input_image_sha256=digest(source))
    write_json(out/'spectrum_and_families.json',result)
    save_csv(out/'frequency_candidates.csv',result['candidates']['peaks'])
    pair=result['reciprocal_basis_candidate']
    save_csv(out/'basis_integer_members.csv',[dict(rank=q['rank'],integer_a=q['integer_coefficients'][0],integer_b=q['integer_coefficients'][1],relative_residual=q['relative_residual']) for q in pair['members']] if pair else [])
    image=Image.fromarray(whole);d=ImageDraw.Draw(image);d.rectangle((x,y,x+w,y+h),outline='#ffb940',width=4)
    views.label(image,[item['id']+' | native decoded frame and analysis ROI']).save(out/'01_native_frame_roi.png')
    Image.fromarray(roi).save(out/'02_roi.png')
    selected={p['rank'] for _,p in targets};views.fft_view(power,fx,fy,result['candidates']['peaks'],selected).save(out/'03_spectrum.png')
    # Decomposition views expose detrending/windowing without changing input PNG.
    from .microscopy import detrend_plane
    detrended=detrend_plane(roi[:,:,1]);window=np.outer(np.hanning(h),np.hanning(w))
    np.savez_compressed(out/'fft_arrays.npz',power=power,fx_cycles_um=fx,fy_cycles_um=fy)
    for name,a in [('04_detrended',detrended),('05_hann_windowed',detrended*window)]:
        lo,hi=np.percentile(a,[1,99]);Image.fromarray(np.uint8(np.clip((a-lo)/max(hi-lo,1e-12),0,1)*255)).save(out/(name+'.png'))
    if targets:
        role,peak=targets[0];profile=normal_profile(roi,sx,sy,peak)
        save_csv(out/'normal_profile.csv',[dict(distance_um=float(a),intensity_green_DN=float(b)) for a,b in zip(profile['distance_um'],profile['values'])])
        save_csv(out/'autocorrelation.csv',[dict(lag_um=float(a),ACF=float(b)) for a,b in zip(profile['lag_um'],profile['acf'])])
        result['normal_profile_acf_period_um']=profile['acf_period_um']
        emit('texture.profile', profile=profile, selected_period_um=peak['period_um'])
        shown=profile['lag_um']<=min(3*peak['period_um'],profile['lag_um'][-1]/3)
        result['acf_display_maximum_lag_um']=float(profile['lag_um'][shown][-1])
        views.line_svg(out/'06_profile_acf.svg',[dict(title='Measured normal profile of selected frequency',xlabel='normal distance (um)',ylabel='green DN',series=[('observed',profile['distance_um'],profile['values'])]),dict(title='Linear ACF: first three periods; complete lags retained in CSV',xlabel='lag (um)',ylabel='overlap-normalized ACF',series=[('ACF',profile['lag_um'][shown],profile['acf'][shown])])])
    write_json(out/'spectrum_and_families.json',result)
    tileview=Image.fromarray(roi);d=ImageDraw.Draw(tileview);d.line((w//2,0,w//2,h),fill='white',width=2);d.line((0,h//2,w,h//2),fill='white',width=2)
    for q in result['spatial_tiles']:d.text((q['ix']*w//2+8,q['iy']*h//2+8),'N/A' if not result['spatial_pair_applicable'] else ('BOTH' if q['both_modes_supported'] else 'NOT BOTH'),font=views.font(15),fill='#16ff94' if q['both_modes_supported'] else '#ffb940',stroke_width=1,stroke_fill='black')
    tileview.save(out/'07_spatial_tiles.png')
    primary=result['primary_family'];low=result['lower_support_family'];strong=result['candidates']['peaks'][0] if result['candidates']['peaks'] else None
    summary=dict(id=item['id'],title=item['title'],status='COMPUTED',quality_status=result['quality_descriptor']['status'],strongest_period_um=strong['period_um'] if strong else None,
                 family_period_um=primary['base']['period_um'] if result['family_passes_quality_gate'] else None,family_rank=primary['base']['rank'] if result['family_passes_quality_gate'] else None,
                 lower_support_period_um=low['base']['period_um'] if low else None,family_ambiguous=result['family_ambiguous'],
                 pair_coverage=pair['coverage'] if pair else None,pair_angle_deg=pair['angle_deg'] if pair else None,
                 jointly_supported_tiles=result['jointly_supported_tiles'],spatial_pair_applicable=result['spatial_pair_applicable'],spatial_pair_supported=result['spatial_pair_supported'])
    write_json(out/'summary.json',summary)
    emit('texture.final', summary=summary, out=out)
    body=f'<p>{html.escape(item["title"])}</p><p>原生RGB像素重新计算；采样为x={sx}、y={sy} µm/px。裁剪保持原标尺。全部候选保留，频率族和整数基底采用现有研究算法。</p>'
    body+='<p><b>频谱质量门控：</b>'+('通过候选筛选' if result['quality_descriptor']['status']=='RESOLVED_CANDIDATES' else '低对比或非周期；不报告主族周期和ACF周期')+'。原对比度、带内功率与峰/背景分数保留在完整记录。</p>'
    body+=views.table(['最强FFT峰周期 µm','主支持频率族 µm','低阈值族 µm','频率族歧义','整数基底coverage','两模态共同支持分块'],[[summary['strongest_period_um'],summary['family_period_um'],summary['lower_support_period_um'],summary['family_ambiguous'],summary['pair_coverage'],f'{summary["jointly_supported_tiles"]}/4' if result['spatial_pair_applicable'] else '不适用：未提出合格二维基底']])
    body+='<div class="gallery">'+views.figure('01_native_frame_roi.png','① 真实全帧与ROI。选择依据是算法难点，不根据与模型的吻合程度挑区域。')+views.figure('02_roi.png','② 未重采样ROI，绿色通道用于纹理频率计算。')+views.figure('04_detrended.png','③ 去除照明平面趋势。')+views.figure('05_hann_windowed.png','④ 二维Hann窗抑制边界泄漏；不作补零。')+'</div>'
    body+=views.figure('03_spectrum.png','⑤ 带物理坐标的二维功率谱。共轭峰只计一族，绿色标记为选定候选。最大功率峰、谐波与空间频率族分别记录。')
    if result['family_passes_quality_gate']:body+=views.table(['族中倍频','候选秩','周期 µm','频率x','频率y'],[[n,p['rank'],p['period_um'],p['fx_cycles_um'],p['fy_cycles_um']] for n,p in primary['members']])
    if pair:body+=f'<p>两基底夹角 {pair["angle_deg"]:.3f}°，coverage={pair["coverage"]:.4f}；这是保留候选峰功率经残差权重后的解释比例，不是全图功率比例或置信度。长度和夹角自由，未强制等长或正交。<a href="basis_integer_members.csv">各候选的整数系数与残差</a>。</p>'
    body+=views.figure('07_spatial_tiles.png','⑥ 四分块独立重算频谱，检验两种频率是否在同一区域共同出现。至少3/4块共同支持才通过；N/A表示未提出合格二维基底，尚无待检验的两个模态，不能记作0/4失败。')
    body+=views.table(['变化','频率角色','相对频率差','匹配'],[[q['variant'],q['role'],q['relative_frequency_difference'],q['matched']] for q in result['central_crop_checks']])
    body+='<p>窗口变化后的独立频率族候选（探索性，不把低支持候选作为已测周期）：</p>'+views.table(['裁剪','主阈值候选 µm','低支持阈值候选 µm'],[[q['variant'],q['primary_candidate_period_um'],q['lower_support_candidate_period_um']] for q in result['central_crop_spectra']])
    if targets:
        body+=views.table(['所选频率周期 µm','附近ACF峰周期 µm'],[[targets[0][1]['period_um'],result['normal_profile_acf_period_um']]])
        body+=views.figure('06_profile_acf.png','⑦ 沿所选频率法线采样，再作线性自相关。显示前3个所选周期内的短滞后；完整曲线在autocorrelation.csv。按重叠样本数归一后的长滞后估计方差较大，不强制限制为±1。ACF在所选周期附近搜索，是同图一致性检查，不是独立基频认证。')
    body+='<p><a href="frequency_candidates.csv">全部频率候选CSV</a> · <a href="spectrum_and_families.json">族、基底、裁剪和分块完整证据</a> · <a href="fft_arrays.npz">原数值功率谱</a></p><p>这些是图像纹理频率候选，不唯一确定分子取向基频或光学相位。当前静态案例重算裁剪和分块，不声称重做了未随包提供的时间邻帧检验。</p>'
    views.page(out/'index.html',item['id']+'：频谱、谐波与空间一致性',body)
    print(item['id'],summary,flush=True)
    return summary


def run_algorithms(root,output,kind='all',case='all',check_reference=True,image=None,pixel_size_um=None):
    verify_bundle(root,quiet=True);out=new_output(root,output);core=root/'data/core';index=read_json(core/'index.json')
    results=[];started=time.perf_counter()
    if image:
        if pixel_size_um is None or not np.isfinite(pixel_size_um) or pixel_size_um<=0:raise ValueError('Own image needs positive calibrated micrometres per pixel')
        with Image.open(image) as im:w,h=im.size
        item=dict(id='own_image',roi_xywh=[0,0,w,h],pixel_scale_um=[pixel_size_um,pixel_size_um],title='用户提供的静态图像；方形采样假设由输入标尺声明')
        results.append(microscopy_case(core,item,out/'own_image',own_image=image))
    else:
        if kind in ('all','diffraction'):
            chosen=[q for q in index['diffraction'] if case=='all' or q['id']==case]
            for item in chosen:results.append(diffraction_case(core,item,out/item['id'],check_reference))
        if kind in ('all','microscopy'):
            chosen=[q for q in index['microscopy'] if case=='all' or q['id']==case]
            for item in chosen:results.append(microscopy_case(core,item,out/item['id']))
    if not results:raise ValueError('No matching algorithm case')
    report=dict(status='FAIL' if any(q['status']=='FAIL' for q in results) else 'PASS',scope='real-image algorithm computation; conditional image evidence, not a new frozen measurement version',cases=results,elapsed_s=time.perf_counter()-started)
    write_json(out/'algorithm_run.json',report)
    body='<p>本次从原生像素重新计算，点击案例查看每个阶段、实际拒绝状态与数据表。</p>'+views.table(['案例','说明','执行状态'],[[q['id'],q.get('case_title',q.get('title','')),q['status']] for q in results])
    body+=''.join(f'<p><a href="{q["id"]}/index.html">打开 {q["id"]} 的处理全过程</a></p>' for q in results)
    body+='<p><a href="algorithm_run.json">完整运行记录</a>。坐标、识别与补检的历史参考只在图像计算完成后作回归；显微结果由输入图像当场重算。</p>'
    views.page(out/'index.html','实图算法实验结果',body)
    if report['status']=='FAIL':raise ValueError('Real-image reference regression failed; see output, no historical files changed')
    return report
