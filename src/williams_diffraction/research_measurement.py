"""RG validation and signed CFA photometry using the accepted algorithms."""
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from .delivery_io import read_json,read_csv,write_json
from .channel_response import analyze
from .raw_photometry import measure_frame
from . import research_figures as plots
from . import research_pages as pages
from .guide_events import emit

GROUPS={'original':'实验1','cycle01':'实验2','cycle02':'实验3','pooled':'合并对照'}
SUPPORT='data/support/channel_response_audit_2026-10-19_v1'

def csv_rows(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def rg_rows(root):
    rows=read_csv(root/SUPPORT/'point_audit.csv')
    boolean={'central','confirmed_nonoptical','reflection_suspect','reflection_overlap_r50','reflection_overlap_r125','channels_ok','hard_saturated','near_white','reflection_overlap_any'}
    for q in rows:
        for key in boolean:
            value=q[key]
            if value not in ('True','False',''):raise ValueError('Invalid boolean: '+key)
            q[key]=None if value=='' else value=='True'
        for key,value in list(q.items()):
            if key.startswith(('Q_','coverage_min_','G_background_area_')) or key.startswith(('R_','G1_','G2_')) and key not in ('R_background_method','G1_background_method','G2_background_method'):
                q[key]=float(value) if value!='' else None
        finite=all(q[f'Q_{c}_{area}'] is not None and np.isfinite(q[f'Q_{c}_{area}']) for area in ('r125','r50') for c in ('R','G1','G2','G'))
        selection=('central' if q['central'] else 'confirmed_nonoptical' if q['confirmed_nonoptical'] else 'channel_failed' if not q['channels_ok'] else 'hard_saturated' if q['hard_saturated'] else 'nonfinite' if not finite else 'reflection_aperture_overlap' if q['reflection_overlap_any'] else 'included')
        if selection!=q['selection']:raise ValueError('Stored RG selection differs from its input evidence')
        for area in ('r125','r50'):
            if all(q[f'Q_{c}_{area}'] is not None for c in ('G','G1','G2')):
                np.testing.assert_allclose(q[f'Q_G_{area}'],(q[f'Q_G1_{area}']+q[f'Q_G2_{area}'])/2,rtol=1e-12,atol=1e-8)
    return rows

def native_profiles(mosaic,info,out):
    from .green_photometry_trial import channel_patch
    p=next(p for p in info['peaks'] if p['peak_uid']==info['target_uid']);center=[p['u_raw_px'],p['v_raw_px']]
    z,xx,yy,ci=channel_patch(mosaic,info['black_level'],center,128,'R');raw=z+info['black_level'][ci]
    counts=((raw>=info['white_level'])&(np.hypot(xx,yy)<=125)).sum(axis=1)
    if not counts.max():raise ValueError('The declared R clipping example has no clipped native samples')
    choices=np.flatnonzero(counts==counts.max());index=int(choices[np.argmin(abs(yy[choices,0]))])
    actual_y=yy[index,0];rows=[];series=[]
    for name in ('R','G1','G2'):
        z,x,y,ci=channel_patch(mosaic,info['black_level'],center,128,name);i=int(abs(y[:,0]-actual_y).argmin());use=abs(x[i])<=125
        values=z[i,use]+info['black_level'][ci]
        series.append(plots.series(name,x[i,use],values,color={'R':'#c75642','G1':'#167d50','G2':'#5b9b88'}[name]))
        rows += [dict(channel=name,x_from_center_raw_px=float(a),y_from_center_raw_px=float(y[i,0]),raw_DN=float(b)) for a,b in zip(x[i,use],values)]
    series.insert(0,plots.series('white level',[-60,60],[info['white_level']]*2,color='#9ba6aa',width=2))
    plots.plot(out/'01_clipping.png',[dict(title='The same spot: R clips while G retains its peak',xlabel='horizontal offset (RAW px)',ylabel='native RAW DN',xlim=(-60,60),ylim=(0,info['white_level']*1.08),series=series)])
    csv_rows(out/'native_profiles.csv',rows)

def run_rg(root,out):
    out.mkdir();rows=rg_rows(root);cfg=read_json(root/'data/research/rg_config.json')
    emit('rg.selection', counts=dict(Counter(q['selection'] for q in rows)), rows=rows)
    summaries,scores,predictions,bins,quartiles=analyze(rows,cfg)
    emit('rg.fitted', scores=scores, predictions=predictions)
    # Reference scores are opened only after selection and all 80 fits finish.
    keys=('group','area','subset','model','response')
    refs={tuple(q[k] for k in keys):q for q in read_csv(root/SUPPORT/'fit_metrics.csv')}
    if len(scores)!=len(refs):raise ValueError('RG analysis coverage changed')
    for q in scores:
        ref=refs[tuple(q[k] for k in keys)]
        for key,value in q.items():
            if isinstance(value,(int,float)):
                np.testing.assert_allclose(value,float(ref[key]),rtol=3e-6,atol=1e-6,err_msg='RG '+key)
    csv_rows(out/'fits.csv',scores);csv_rows(out/'predictions.csv',predictions);csv_rows(out/'selection.csv',summaries);csv_rows(out/'residual_bins.csv',bins);csv_rows(out/'intensity_quartiles.csv',quartiles)
    selected=[q for q in rows if q['selection']=='included'];primary=[q for q in scores if q['area']=='r125' and q['model']=='affine' and q['subset']=='all_unsaturated' and q['response']=='G_from_R' and q['group']!='pooled']
    all_panels=[]
    for fit in primary:
        rr=[r for r in selected if r['group']==fit['group']];x=np.array([r['Q_R_r125'] for r in rr])/1e6;y=np.array([r['Q_G_r125'] for r in rr])/1e6
        span=[float(x.min()),float(x.max())];model=fit['slope']*np.array(span)+fit['intercept_DN_pixel2']/1e6
        panel=dict(title=f'Experiment {list(GROUPS).index(fit["group"])+1}: R2={fit["R2"]:.4f}',xlabel='Q_R / million DN RAW px^2',ylabel='Q_G / million DN RAW px^2',series=[plots.series('measured',x,y,'scatter'),plots.series('affine fit',span,model)])
        all_panels.append(panel)
        if fit['group']=='cycle02':
            residual=(y-(fit['slope']*x+fit['intercept_DN_pixel2']/1e6))*1e6
            plots.plot(out/'02_rg_and_residual.png',[panel,dict(title='Residuals include weak and signed measurements',xlabel='Q_R / million DN RAW px^2',ylabel='G residual (DN RAW px^2)',series=[plots.series('measured - fitted',x,residual,'scatter')])],columns=2)
            use=x<=np.quantile(x,.20)
            plots.plot(out/'04_weak_signal.png',[dict(title='Lowest 20% in R: view only, same full-data fit',xlabel='Q_R / million DN RAW px^2',ylabel='Q_G / million DN RAW px^2',series=[plots.series('weak observations',x[use],y[use],'scatter'),plots.series('unchanged fit',[x[use].min(),x[use].max()],fit['slope']*np.array([x[use].min(),x[use].max()])+fit['intercept_DN_pixel2']/1e6)])])
    plots.plot(out/'03_three_experiments.png',all_panels,columns=3)
    result=dict(status='PASS',input_candidates=len(rows),included_candidates=len(selected),reference_groups=len(scores),primary=primary,
                signed_nonpositive_r125=sum(q['Q_R_r125']<=0 or q['Q_G_r125']<=0 for q in selected),bootstrap_replicates=cfg['bootstrap_replicates'],
                measurement='Refit frozen CFA integrals; pixel remeasurement is the separate photometry example',radiometric_power_calibration=False)
    write_json(out/'rg.json',result)
    emit('rg.final', result=result, out=out)
    return result

def run_photometry(root,out):
    out.mkdir();base=root/'data/research';info=read_json(base/'photometry125.json')
    with np.load(base/info['array'],allow_pickle=False) as archive:mosaic=archive['mosaic']
    if list(mosaic.shape)!=info['crop_xywh'][3:1:-1] or mosaic.dtype!=np.uint16:raise ValueError('Invalid native CFA sample')
    measured,channels=measure_frame(mosaic,info['black_level'],info['white_level'],info['camera_white_level'],info['peaks'],np.asarray(info['corners_raw_px']),info['reflections'],cfg=info['photometry_config'],green_to_red_ratio=None,green_linearity='WORKING_ASSUMPTION_NOT_RADIOMETRIC_CALIBRATION')
    # No reference photometry is available to the measurer above.
    reference=read_json(base/info['reference']);refs={q['peak_uid']:q for q in reference['photometry']};crefs={(q['peak_uid'],q['channel']):q for q in reference['channels']}
    maximum=0.
    for q in measured:
        for key,value in refs[q['peak_uid']].items():
            if key in ('peak_uid','quality_flags','measurement_status'):
                if q[key]!=value:raise ValueError('Photometry status changed: '+key)
            else:
                np.testing.assert_allclose(q[key],value,rtol=2e-9,atol=1e-6,err_msg=key);maximum=max(maximum,abs(q[key]-value))
    for q in channels:
        ref=crefs[q['peak_uid'],q['channel']]
        for key in ('background_method','background_tiles','background_pixels','background_sectors','saturated_sample_count','near_camera_white_sample_count','measurement_status'):
            if q[key]!=ref[key]:raise ValueError('CFA sampling evidence changed: '+key)
        for key in ('Q_r50','Q_r125','coverage_r50','coverage_r125','area_raw_pixel2_r50','area_raw_pixel2_r125','background_c0_DN','background_cx_per300_DN','background_cy_per300_DN'):
            np.testing.assert_allclose(q[key],ref[key],rtol=2e-9,atol=1e-6,err_msg='CFA '+key)
    write_json(out/'measurements.json',dict(photometry=measured,channels=channels));csv_rows(out/'net_integrals.csv',measured);csv_rows(out/'channels.csv',channels)
    native_profiles(mosaic,info,out);plots.photometry_views(out,mosaic,info,measured,channels)
    result=dict(status='PASS',frame_id='125',experiment=3,voltage_RMS_V=6.77,peaks=len(measured),channels=len(channels),maximum_reference_difference=maximum,
                raw_decoding_at_runtime=False,signed_integrals_preserved=True,green_to_red_conversion_applied=False,reference_loaded_after_measurement=True,
                target=[q for q in measured if q['peak_uid'] in (info['target_uid'],info['comparison_uid'])])
    limited=min(measured,key=lambda q:q['aperture_coverage_min']);result['partition_example']={k:limited[k] for k in ('peak_uid','aperture_coverage_min','quality_flags')}
    write_json(out/'photometry.json',result)
    emit('photometry.final', result=result, out=out)
    return result

def chapter(root,out):
    from .delivery_algorithms import diffraction_case
    rg=run_rg(root,out/'rg');photometry=run_photometry(root,out/'photometry')
    core=root/'data/core';index=read_json(core/'index.json');diffraction=[]
    (out/'diffraction').mkdir()
    for item in index['diffraction']:diffraction.append(diffraction_case(core,item,out/'diffraction'/item['id']))
    if any(q['status']!='PASS' for q in diffraction):raise ValueError('Diffraction pixel regression failed; see each case reference_check.json')
    body='<h2>先确定怎样量光强</h2><p>红色激光的强光斑会使R通道削顶。我们先检验未饱和光斑的RG关系，再采用绿色净积分比较相对光强。</p>'
    body+=pages.figure(out,out/'photometry/01_clipping.png','实验3，6.77 V。同一光斑峰心±60 RAW像素的原生剖面：R峰顶被截平，G仍保留峰形。G1、G2位置略有错开，不插值、不作增益缩放；完整剖面另存CSV。')
    body+=pages.figure(out,out/'rg/02_rg_and_residual.png','实验3的2138个共同未饱和候选。本次重新拟合RG关系，并保留弱信号和有符号积分。残差用于观察直线关系的适用范围。')
    body+=pages.table(['实验','光斑数','斜率','R²'],[[GROUPS[q['group']],q['n'],q['slope'],q['R2']] for q in rg['primary']])
    body+='<p>三轮结果支持采用G作相对测光；它不是绝对功率标定。拟合样本中，G原生采样最高读数为3509 DN，削顶示例达到7317 DN，亮区使用仍包含响应外推。</p>'
    body+=pages.details('RG拟合的进一步检查',pages.figure(out,out/'rg/03_three_experiments.png','三轮分别拟合，未把不同ISO下的绝对读数合为同一光功率。')+pages.figure(out,out/'rg/04_weak_signal.png','弱信号局部放大只改变视窗，不重新挑点拟合。')+'<p>80组实际重算包含两孔径、过原点/带截距/二次模型、G1/G2核对、整帧留出和2000次按帧重采样。实验1保留旧反光审查身份。</p>'+pages.a(out,out/'rg/fits.csv','拟合与留出结果')+' · '+pages.a(out,out/'rg/selection.csv','选样统计')+' · '+pages.a(out,out/'rg/predictions.csv','逐点残差'))
    body+='<h2>再把斜拍照片变成坐标</h2><p>用光屏四边建立毫米坐标。随后先识别可靠光斑，再根据排列预测待查位置，回到原生采样检查是否有真实信号。</p>'
    body+=pages.figure(out,out/'diffraction/104/01_screen_geometry.png','实验3，5.67 V。坐标变换以屏板为依据；条件边仍保留限制，不用点阵是否方正来校正照片。')
    body+=pages.figure(out,out/'diffraction/104/05_guided_detection.png','多尺度识别与预测补检：青色为独立强候选，橙色为独立弱候选，紫色为有实测信号支持的补检点。')
    body+='<p>本例得到40个强候选、15个独立弱候选和1个几何补检。另一个3.48 V案例虽能拟合峰列，但预测检查未通过，因此关闭补检。</p>'
    labels={'104':'密集二维光斑与弱峰补检','108':'宽峰与小点阵共存','197':'屏边支撑不足','200':'预测不可靠时停止补检'}
    body+=pages.details('识别过程与更多案例',''.join('<p>'+pages.a(out,out/'diffraction'/q['id']/'index.html',labels[q['id']])+'</p>' for q in diffraction))
    body+='<h2>最后得到光斑的净积分</h2><p>位置测量与光强测量分开进行。积分回到未插值的RAW采样：分别扣黑、估计背景、处理邻峰分区，再平均G1、G2的净积分。</p>'
    body+=pages.figure(out,out/'photometry/125_audit_p007_measurement.png','测光示例沿用已审阅位置，重新计算22个光斑。黄色和绿色圈为r50、r125；背景图显示进入稳健分块拟合前的可用采样。负残差保留，不截成零。')
    body+=pages.table(['光斑','G1净积分','G2净积分','平均G净积分'],[[q['peak_uid'],q['Q_G1_r125'],q['Q_G2_r125'],q['Q_G_r125']] for q in photometry['target']])
    body+='<p class="muted">积分单位为DN·RAW px²；两路绿色分别积分后取平均。</p>'
    limited=photometry['partition_example']
    body+=pages.details('孔径、背景与邻峰分区',pages.figure(out,out/'photometry/125_audit_p013_measurement.png','未饱和光斑对照。亮度显示拉伸只用于看图，未进入积分。')+pages.figure(out,out/'photometry'/(limited['peak_uid']+'_measurement.png'),f'区域不完整的例子：邻峰分区或屏边限制后，最小通道支持比例为{100*limited["aperture_coverage_min"]:.1f}%。原质量标志保留。')+pages.figure(out,out/'photometry/125_audit_p013_aperture.png','固定同一背景与邻峰分区，比较积分随孔径的变化。r50与r125都保留，不能把孔径变化当成光强变化。')+pages.a(out,out/'photometry/net_integrals.csv','全部净积分')+' · '+pages.a(out,out/'photometry/channels.csv','各通道背景与质量标志')+' · '+pages.a(out,root/'src/williams_diffraction/raw_photometry.py','测光源码'))
    body+='<p><b>得到的测量量：</b>屏面位置和绿色净积分。接下来先看这些图样怎样随电压变化。</p>'
    return body,dict(status='PASS',reader_result='建立屏面坐标与绿色净积分；RG、识别补检和CFA测光均已实际重算。',rg=rg,photometry=photometry,diffraction=diffraction)
