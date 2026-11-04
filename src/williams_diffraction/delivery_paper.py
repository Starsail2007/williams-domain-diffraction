"""Small, paper-numbered reproductions. Never reads RAW or writes frozen inputs.

Numerical checks recompute observations with the published functions. The saved
paper artwork is copied separately; regenerated SVGs are diagnostic plots, not
claims of pixel-identical typesetting or new experimental measurements.
"""
from collections import defaultdict
import csv
import json
import shutil
import numpy as np
from .delivery_io import read_json, read_csv, write_json, verify_bundle, new_output, parity_svg
from .diffraction_models import fit_one_d, one_d_prediction, fit_spectrum, weights, fit_geometry


def close(actual, expected, label, rtol=2e-6, atol=2e-8):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)


def csv_out(path, rows):
    if not rows:
        raise ValueError('Empty scientific output')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def period_summary(rows):
    """ROI -> historical diffraction frame -> group; residual medians in that order."""
    groups = defaultdict(lambda: defaultdict(list))
    for q in rows:
        if q['video_id'] != 'micro_late':
            raise ValueError('The current paper uses only micro_late')
        residual = q['image_period_um']/(q['correspondence_factor']*q['current_optical_period_um'])-1
        close(residual, q['current_residual'], 'ROI period residual', rtol=1e-12, atol=1e-12)
        groups[q['batch'], q['cycle'], q['video_id'], q['kind']][q['historical_frame_id']].append(q)
    result = []
    for key, frames in sorted(groups.items()):
        parts = []
        for fid, rr in frames.items():
            close([q['current_optical_period_um'] for q in rr], rr[0]['current_optical_period_um'], 'same historical optical period')
            parts.append(dict(frame_id=fid, p_um=float(np.median([q['image_period_um'] for q in rr])),
                              d_um=rr[0]['current_optical_period_um'],
                              residual=float(np.median([q['current_residual'] for q in rr]))))
        result.append(dict(batch=key[0], cycle=key[1], video=key[2], kind=key[3],
                           n_roi_matches=sum(map(len, frames.values())), n_historical_frames=len(parts),
                           p_median_um=float(np.median([q['p_um'] for q in parts])),
                           d_median_um=float(np.median([q['d_um'] for q in parts])),
                           residual_median=float(np.median([q['residual'] for q in parts])), frames=parts))
    return result


def table2(root, out):
    rows = read_json(root/'data/paper/microscopy_matches.json')
    result = period_summary(rows)
    refs = {(q['batch'], q['cycle'], q['kind']):q for q in read_json(root/'data/paper/comparison_summary.json')}
    for q in result:
        ref = refs[q['batch'], q['cycle'], q['kind']]
        assert q['n_historical_frames'] == ref['n_historical_frames']
        close(q['residual_median'], ref['median_residual'], 'table 2 group median')
    assert len(rows) == 31 and [q['n_historical_frames'] for q in result] == [4, 5, 7, 1]
    one_d=[q for q in result if q['kind']=='1d']
    two_d=[q for q in result if q['kind']=='2d']
    csv_out(out/'table02.csv', [{k:v for k,v in q.items() if k!='frames'} for q in one_d])
    write_json(out/'table02_details.json', one_d)
    csv_out(out/'periods_2d_supplement.csv', [{k:v for k,v in q.items() if k!='frames'} for q in two_d])
    # Factors are correspondence hypotheses recorded per match, not fitted here.
    parity_svg(out/'fig11_periods.svg', [((2 if q['cycle']=='original' else 1)*frame['d_um'], frame['p_um']) for q in one_d for frame in q['frames']],
               'Paper period correspondence (16 historical frames)', 'factor * diffraction period (um)', 'image period (um)')
    return dict(status='PASS', operation='recompute frame-level aggregation', roi_matches=31,
                one_dimensional_historical_frames=[4,5,7], two_dimensional_matches=1, summary=one_d, two_dimensional_supplement=two_d)


def rgb(root, out):
    rows = read_csv(root/'data/paper/channel_points_cycle02.csv')
    assert len(rows) == 2138 and all(q['group']=='cycle02' and q['selection']=='included' for q in rows)
    x=np.array([float(q['Q_R_r125']) for q in rows]);y=np.array([float(q['Q_G_r125']) for q in rows])
    slope,b = np.linalg.lstsq(np.c_[x/1e6,np.ones(len(x))], y/1e6, rcond=None)[0];b*=1e6
    pred=slope*x+b;score=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
    ref=read_json(root/'data/paper/channel_affine_reference.json')
    close([slope,b,score],[float(ref['slope']),float(ref['intercept_DN_pixel2']),float(ref['R2'])], 'RG affine regression')
    parity_svg(out/'fig02_RG.svg', zip(y/1e6,pred/1e6), 'RG affine response: observed vs fitted G', 'Observed G (million DN RAW-px^2)', 'Fitted G (million DN RAW-px^2)')
    return dict(status='PASS',operation='refit affine RG response',n=2138,slope=float(slope),intercept_DN_RAW_px2=float(b),R2=float(score))


def prediction(root, out):
    from .guided_detection import prediction_sites
    from .geometry import transform, raw_to_display
    from PIL import Image, ImageDraw
    data=read_json(root/'data/paper/prediction014.json')
    geom=data['geometry']
    Hinv=np.linalg.inv(geom['H_raw_to_screen_mm'])
    sites={(model['kind'],str(m),str(n)):transform(np.atleast_2d(mm),Hinv)[0]
           for model in geom['models'] for m,n,mm in prediction_sites(model,model['kind'])}
    output=[]
    for q in data['predictions']:
        xy=sites[q['model'],q['index_m'],q['index_n']]
        close(xy,[float(q['predicted_u_raw_px']),float(q['predicted_v_raw_px'])], '014 predicted RAW coordinates',atol=1e-6)
        output.append(dict(model=q['model'],index_m=q['index_m'],index_n=q['index_n'],u_raw_px=float(xy[0]),v_raw_px=float(xy[1]),status=q['status']))
    strong=[q for q in data['spots'] if q['detection_origin']=='independent_strong']
    assert (len(data['spots']),len(strong),len(output))==(35,25,50)
    with Image.open(root/'data/samples/detection014.png') as source:
        im=source.convert('RGB')
    draw=ImageDraw.Draw(im);red_h,red_w=data['display_coordinate_red_shape']
    def display(xy):
        # Pixel centres: full visible RAW -> R plane -> retained display size.
        return raw_to_display(np.asarray(xy),(red_h,red_w))
    for q in output:
        x,y=display((q['u_raw_px'],q['v_raw_px']));draw.rectangle((x-4,y-4,x+4,y+4),outline='gray',width=1)
    for q in data['spots']:
        x,y=display((float(q['u_raw_px']),float(q['v_raw_px'])))
        draw.ellipse((x-7,y-7,x+7,y+7),outline='cyan' if q['detection_origin']=='independent_strong' else 'lime',width=2)
    draw.rectangle((8,8,530,80),fill='black')
    draw.text((16,14),'Cyan circles: 25 stored independent strong spots',fill='cyan')
    draw.text((16,35),'Green circles: 10 stored guided additions',fill='lime')
    draw.text((16,56),'Gray squares: 50 replayed proposals; no RAW redetection',fill='white')
    im.save(out/'fig04_prediction_overlay.png');csv_out(out/'fig04_predictions.csv',output)
    return dict(status='PASS',operation='replay stored geometric proposals; measured spots are frozen',strong=25,additional=10,prediction_rows=50,
                status_counts={s:sum(q['status']==s for q in output) for s in sorted({q['status'] for q in output})},
                proposal_models={s:sum(q['model']==s for q in output) for s in ('lattice_2d','row')},raw_redetection_performed=False)


def texture(root, out):
    from PIL import Image
    from .microscopy import spectrum_peaks
    result={}
    for name,roi in [('grid','left'),('stripe','right')]:
        provenance=read_json(root/f'data/examples/microscopy/{name}_provenance.json')
        with Image.open(root/f'data/examples/microscopy/{name}.png') as im:a=np.asarray(im)
        found,power,fx,fy=spectrum_peaks(a,provenance['sx_um'],provenance['sy_um'])
        ref=next(q for q in read_json(root/'data/paper/microscopy_selected.json')['records'] if q['micro_frame_index']==164 and q['roi_id']==roi)
        targets=[ref['pair_period_1_um'],ref['pair_period_2_um']] if name=='grid' else [ref['long_period_um']]
        recovered=[found['peaks'][rank-1]['period_um'] for rank in ref['selected_ranks']]
        close(recovered,targets,'late164 reviewed FFT candidates',rtol=1e-9,atol=1e-9)
        # Autoscaled log-power is a display only, not a calibrated intensity map.
        display=np.log1p(power);display=(255*(display-display.min())/max(float(np.ptp(display)),1e-20)).astype('uint8')
        Image.fromarray(display).save(out/f'fig10_{name}_FFT.png')
        result[name]=dict(selected_periods_um=recovered,reviewed_ranks=ref['selected_ranks'],candidates=found)
    write_json(out/'fig10_texture.json',result)
    csv_out(out/'fig10_periods.csv',[dict(roi=name,reviewed_rank=rank,period_um=period) for name,q in result.items() for rank,period in zip(q['reviewed_ranks'],q['selected_periods_um'])])
    return dict(status='PASS',operation='recompute FFT from decoded RGB crops',frame_index=164,video_id='micro_late',results=result,
                limit='Candidate texture frequencies retain harmonics; not a certified director fundamental')


def geometry(root,out):
    groups=defaultdict(list)
    for q in read_json(root/'data/paper/geometry_nodes.json'):groups[q['cycle'],q['frame_id']].append(q)
    result=[];points=[]
    for ref in read_json(root/'data/paper/geometry_models.json'):
        rr=groups[ref['cycle'],ref['frame_id']];labels=np.array([q['labels'] for q in rr]);xy=np.array([q['observed_s'] for q in rr])
        coefficients,pred=fit_geometry(labels,xy,'rectangular')
        close(coefficients,ref['coefficients'],'rectangular geometry',atol=1e-9)
        close(pred,[q['fitted_s'] for q in rr],'geometry predictions',atol=1e-9)
        period=.6328/np.linalg.norm(coefficients[1]+coefficients[2]);close(period,ref['period_um'],'geometry period')
        result.append(dict(cycle=ref['cycle'],fid=ref['frame_id'],period_um=float(period),n_nodes=len(rr)))
        points.extend(zip(xy.ravel(),pred.ravel()))
    assert len(result)==28 and sum(q['n_nodes'] for q in result)==572
    csv_out(out/'fig12_geometry.csv',result);parity_svg(out/'fig12_geometry.svg',points,'Rectangular geometry: 28 frames','Observed direction sine','Refitted direction sine')
    return dict(status='PASS',operation='refit retained labelled geometry',frames=28,nodes=572,zero_order_and_order_labels_independently_confirmed=False)


def one_dimensional(root,out,refit=False):
    rows=read_csv(root/'data/legacy_original_23/one_dimensional_ratios.csv')
    U=np.array([float(q['voltage_RMS_V']) for q in rows]);y=np.log([[float(q['R21_observed']),float(q['R31_observed'])] for q in rows])
    ref=read_json(root/'data/paper/one_d_reference.json');z=(U-5.69)/(8.20-5.69)
    p=np.array(ref['parameters']);pred=one_d_prediction(p,z)
    close(np.exp(pred),[[float(q['R21_fitted']),float(q['R31_fitted'])] for q in rows],'original 23 frozen-parameter replay')
    if refit:
        p,pred,_=fit_one_d(z,y,False)
        close(np.exp(pred),[[float(q['R21_fitted']),float(q['R31_fitted'])] for q in rows],'original 23 independent refit')
    obs=np.exp(y);values=np.exp(pred)
    scores=[float(1-np.sum((obs[:,k]-values[:,k])**2)/np.sum((obs[:,k]-obs[:,k].mean())**2)) for k in [0,1]]
    close(scores,ref['R2'],'original-ratio-domain R squared')
    csv_out(out/'fig15_predictions.csv',[dict(frame_id=q['frame_id'],U_V=float(U[i]),R21_observed=float(obs[i,0]),R31_observed=float(obs[i,1]),R21_fitted=float(values[i,0]),R31_fitted=float(values[i,1])) for i,q in enumerate(rows)])
    parity_svg(out/'fig15_one_d.svg',zip(obs.ravel(),values.ravel()),'Original 23 frames: one-dimensional ratios','Observed ratio','Model ratio')
    # Teachers can inspect the actual voltage curves without opening a table.
    smooth_U=np.linspace(5.69,8.20,250)
    smooth=np.exp(one_d_prediction(p,(smooth_U-5.69)/(8.20-5.69)))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="440" viewBox="0 0 1080 440"><rect width="1080" height="440" fill="white"/>']
    for k,title in enumerate(('R21','R31')):
        maximum=max(float(obs[:,k].max()),float(smooth[:,k].max()))*1.08
        px=lambda u:70+(float(u)-5.69)/(8.20-5.69)*410
        py=lambda y:350-float(y)/maximum*280
        svg.append(f'<g transform="translate({k*540},0)" font-family="sans-serif" fill="#182732"><text x="70" y="28" font-size="19">{title}: 23 frames, R2 = {scores[k]:.4f}</text><path d="M70 70 V350 H480" fill="none" stroke="#182732"/>')
        for u in np.linspace(5.69,8.20,4):
            svg.append(f'<text x="{px(u)-15:.2f}" y="375" font-size="13">{u:.2f}</text>')
        for y in np.linspace(0,maximum,5):
            svg.append(f'<text x="15" y="{py(y)+4:.2f}" font-size="13">{y:.3f}</text>')
        line=' '.join(f'{px(u):.3f},{py(v):.3f}' for u,v in zip(smooth_U,smooth[:,k]))
        svg.append(f'<polyline points="{line}" fill="none" stroke="#087ead" stroke-width="2"/>')
        for u,v in zip(U,obs[:,k]):svg.append(f'<circle cx="{px(u):.3f}" cy="{py(v):.3f}" r="3.4" fill="#c45b25"/>')
        svg.append('<text x="220" y="410">Voltage (RMS V)</text><text x="70" y="52" font-size="13">orange: observed; blue: double-harmonic model</text></g>')
    svg.append('</svg>');(out/'fig15_voltage_curves.svg').write_text(''.join(svg),encoding='utf-8')
    return dict(status='PASS',operation='refit' if refit else 'frozen-parameter replay',frames=23,parameters=p.tolist(),voltage_window_V=[5.69,8.20],R2_ratio_domain=scores,training_loss='natural-log ratios',train_log_RMSE=float(np.sqrt(np.mean((pred-y)**2))))


def spectrum(root,out,refit=False):
    groups=defaultdict(list)
    for q in read_json(root/'data/paper/spectrum_nodes.json'):groups[q['cycle'],q['fid']].append(q)
    refs={(q['cycle'],q['fid']):q for q in read_json(root/'data/paper/spectrum_parameters.json')}
    frames=[];nodes=[];parity=[]
    for key,rr in groups.items():
        labels=np.array([[q['a'],q['b']] for q in rr]);observed=np.array([q['observed'] for q in rr]);frozen=np.array([q['fitted'] for q in rr])
        ref=refs[key];p=np.array(ref['parameters']);amplitude=ref['amplitude'];pred=amplitude*weights(labels,p,'bessel_low')
        close(pred,frozen,'spectrum frozen-parameter replay',atol=1e-5)
        if refit:
            p,amplitude,pred,_=fit_spectrum(labels,observed,'bessel_low');close(pred,frozen,'spectrum refit',atol=1e-5)
        score=float(np.linalg.norm(pred-observed)/np.linalg.norm(observed));maximum=float(observed.max())
        frames.append(dict(cycle=key[0],fid=key[1],n_nodes=len(rr),NRMSE=score,parameters=p.tolist(),amplitude=float(amplitude)))
        for i,q in enumerate(rr):
            nodes.append({**q,'reproduced':float(pred[i]),'observed_normalized':float(observed[i]/maximum),'fit_normalized':float(pred[i]/maximum)})
        parity.extend(zip(observed/maximum,pred/maximum))
    summary=[]
    for ref in read_json(root/'data/paper/spectrum_metrics.json'):
        selected=[q for q in frames if q['cycle']==ref['cycle']]
        assert len(selected)==ref['n_frames'] and sum(q['n_nodes'] for q in selected)==ref['n_nodes']
        value=float(np.median([q['NRMSE'] for q in selected]));close(value,ref['NRMSE_median'],'per-frame median NRMSE')
        summary.append(dict(cycle=ref['cycle'],frames=len(selected),nodes=ref['n_nodes'],NRMSE_median=value))
    assert len(frames)==26 and len(nodes)==276
    csv_out(out/'fig16_predictions.csv',nodes);write_json(out/'fig16_frames.json',frames)
    parity_svg(out/'fig16_spectrum.svg',parity,'Paper bright-node spectrum: 26 frames, 276 nodes','Observed / frame observed max','Fit / frame observed max')
    return dict(status='PASS',operation='26 per-frame refits' if refit else '26 frozen-parameter replays',summary=summary,
                loss='signed, unnormalized camera-signal L2 per frame',physical_power_calibrated=False)


def run_paper(root,output,figure=None,table=None,all_items=False,refit=False):
    verify_bundle(root,quiet=True);out=new_output(root,output)
    index=read_json(root/'provenance/paper_release.json')['figures']
    if refit and not (all_items or figure in (15,16)):
        raise ValueError('--refit applies to figure 15/16 or --all')
    report=dict(status='PASS',paper_version='williams_final_2026-11-02_v04',refit_requested=refit,targets={})
    functions={2:rgb,4:prediction,10:texture,11:table2,12:geometry,15:one_dimensional,16:spectrum}
    numbers=range(1,18) if all_items else ([figure] if figure else [])
    for number in numbers:
        item=index[number-1];shutil.copyfile(root/item['preview'],out/f'fig{number:02d}_paper.png')
        if number in functions:
            fn=functions[number]
            value=fn(root,out,refit) if number in (15,16) else fn(root,out)
        else:
            value=dict(status='COPIED_REFERENCE',operation='byte-copy frozen paper artwork',evidence=item['inputs'],limit=item['scope'])
        report['targets'][f'figure{number:02d}']=value
    if all_items or table==2:
        report['targets']['table02']=table2(root,out)
    if all_items or table==1:
        shutil.copyfile(root/'data/experiments.csv',out/'table01_experiments.csv')
        report['targets']['table01']=dict(status='COPIED_REFERENCE',operation='export documented experiment settings')
    write_json(out/'paper_checks.json',report)
    (out/'README.md').write_text('# 论文复现输出\n\n逐项状态见 [paper_checks.json](paper_checks.json)。`PASS` 为实际数值核对，`COPIED_REFERENCE` 为冻结图稿复制。`figNN_paper.png`为复制的论文原图；SVG、`fig04_prediction_overlay.png`及`fig10_*_FFT.png`为本次计算的诊断图。没有重新测量RAW、认证物理级次或校准光功率。\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',targets=list(report['targets']),refit=refit),ensure_ascii=False))
    return report
