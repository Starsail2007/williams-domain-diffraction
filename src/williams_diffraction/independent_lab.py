"""Independent algorithm runs on explicitly selected experimental samples."""
from collections import Counter
from pathlib import Path
import hashlib
import time
import traceback

import numpy as np
from PIL import Image

from .delivery_io import read_json, read_csv, write_json as save_json, new_output, verify_bundle
from .guide_events import observe
from .guided_lab import Recorder, plain
from .runtime_trace import console_values, log_values
from . import research_figures as plots


def write_json(path, value):
    save_json(path, plain(value))


class TargetReady(Exception):
    """An early array stage has returned; no downstream algorithm is needed."""


class IndependentRecorder(Recorder):
    def __init__(self, root, out, tool, sample, on_state=None):
        self.on_state=on_state
        self.tool, self.sample = tool, sample
        case = dict(id=tool['id'], type=tool['type'], steps=[dict(id=tool['id'])])
        super().__init__(root, out, case)
        self.started = time.monotonic()

    def publish(self, **updates):
        super().publish(**updates)
        if self.on_state is not None:self.on_state(plain(self.state))

    def path(self, path):
        path=Path(path).resolve()
        return path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else str(path)

    def accepts(self, name, data):
        if name.split('.')[0] != self.tool['type']:
            return False
        if name.startswith('photometry.') and 'channel' in data:
            return data['peak'] == self.sample['id'] and data['channel'] == 'G1'
        return True

    def record(self, name, data):
        # These are real prerequisites prepared inside this invocation, not
        # results loaded from a previous run or steps the reader must unlock.
        if name == 'detection.geometry':
            self.geometry = data['geometry']
            with np.load(data['folder']/data['case']['input_arrays'], allow_pickle=False) as a:
                self.red = a['red'].copy()
        if name == 'texture.spectrum' and not hasattr(self, 'spectrum'):
            self.spectrum = data
        if name.startswith('texture.') and name in self.seen:
            return
        if name.startswith('photometry.'):
            if name != self.tool['id']:
                return
            folder = self.out/'selected'; folder.mkdir(exist_ok=True)
            if name == 'photometry.channel':
                self.result(images=[self.bitmap(folder, 'black_subtracted', data['array'])],
                            metrics=[['光斑', data['peak']], ['通道', data['channel']], ['黑电平 / DN', data['black']]])
            elif name == 'photometry.background':
                self.result(images=[self.bitmap(folder, 'background', data['array']), self.bitmap(folder, 'support', data['mask'])],
                            metrics=[['光斑', data['peak']], ['背景系数', data['coefficients'].tolist()]])
            elif name == 'photometry.integral':
                self.result(images=[self.bitmap(folder, 'weights', data['weights']), self.bitmap(folder, 'residual', data['residual'])],
                            metrics=[['光斑', data['peak']], ['G1净积分 / DN·RAW px²', data['integral']], ['有效面积 / RAW px²', data['area']]])
            raise TargetReady()
        super().record(name, data)
        if name == self.tool['id'] and name in {
            'detection.geometry', 'detection.contrast', 'texture.detrend',
            'texture.window', 'texture.spectrum', 'texture.family', 'texture.checks', 'texture.profile'
        }:
            raise TargetReady()

    def result(self, images=None, metrics=None, table=None, links=None):
        event = dict(step=self.tool['id'], sequence=len(self.events)+1, status='computed',
                     images=images or [], metrics=metrics or [], table=table,
                     evidence=[], links=links or [], elapsed_s=round(time.monotonic()-self.started, 3))
        self.events.append(plain(event))
        self.publish(current_step=self.tool['id'], message='正在保存本次结果')


def select_definition(root, algorithm, sample_id):
    catalog = read_json(root/'configs/experiments.json')
    tool = next((q for q in catalog['tools'] if q['id'] == algorithm), None)
    if tool is None:
        raise ValueError('没有这个算法')
    sample = next((q for q in tool['samples'] if q['id'] == sample_id), None)
    if sample is None:
        raise ValueError('此样本不适用于所选算法')
    return tool, sample


def rg_experiment(root, out, tool, sample, recorder):
    from .research_measurement import rg_rows
    from .channel_response import fit_rows, bootstrap, arrays, SCALE
    all_rows = rg_rows(root)
    rows = [q for q in all_rows if sample['id'] == 'pooled' or q['group'] == sample['id']]
    selected = [q for q in rows if q['selection'] == 'included']
    log_values('RG_sample_selection', group=sample['id'], counts=dict(Counter(q['selection'] for q in rows)),
               selected_peak_keys=[q['peak_key'] for q in selected])
    if tool['id'] == 'rg.selection':
        recorder.result(table=dict(headers=['筛选结果', '数量'], rows=list(Counter(q['selection'] for q in rows).items())),
                        metrics=[['候选数', len(rows)], ['纳入数', len(selected)]])
        return
    x, y = arrays(selected, 'r125'); scores=[]; lines=[]
    for model in ('origin', 'affine', 'quadratic'):
        score, pred, cv = fit_rows(selected, 'r125', model, sample['id'])
        scores.append(score)
        if model == 'affine':
            affine, fitted = score, pred
        lines.append(plots.series(model, x[np.argsort(x)], pred[np.argsort(x)]))
    cfg=read_json(root/'data/research/rg_config.json')
    if tool['id'] == 'rg.final':
        interval=bootstrap(selected, 'r125', cfg['bootstrap_replicates'], np.random.default_rng(cfg['seed']))
        log_values('RG_selected_group_intervals', group=sample['id'], **interval)
        affine.update(interval)
    image=out/'RG.png'
    plots.plot(image, [dict(title=sample['label']+' / R-G response', xlabel='Q_R / million DN RAW px²', ylabel='Q_G / million DN RAW px²',
                           series=[plots.series('observed', x, y, 'scatter')]+lines),
                       dict(title='Affine residuals', xlabel='Q_R / million DN RAW px²', ylabel='residual / million DN RAW px²',
                            series=[plots.series('observed - fit', x, y-fitted, 'scatter')])], columns=2)
    write_json(out/'fits.json', scores)
    # Reference regression is evaluation only, opened after the selected fit.
    reference=next(q for q in read_csv(root/'data/support/channel_response_audit_2026-10-19_v1/fit_metrics.csv')
                   if q['group']==sample['id'] and q['area']=='r125' and q['model']=='affine' and q['response']=='G_from_R' and q['subset']=='all_unsaturated')
    np.testing.assert_allclose([affine[k] for k in ('slope','intercept_DN_pixel2','R2','LOFO_R2')],
                               [float(reference[k]) for k in ('slope','intercept_DN_pixel2','R2','LOFO_R2')], rtol=3e-6, atol=1e-6)
    recorder.result(images=[recorder.path(image)], metrics=[['光斑数', len(selected)], ['按帧留出', affine['n_frames']]],
                    table=dict(headers=['模型', '斜率', 'R²', '整帧留出 R²'], rows=[[q['model'],q['slope'],q['R2'],q['LOFO_R2']] for q in scores]),
                    links=[dict(label='拟合与区间', path=recorder.path(out/'fits.json'))])


def photometry_experiment(root, out, tool, sample, recorder):
    from .raw_photometry import measure_frame
    info=read_json(root/'data/research/photometry125.json')
    with np.load(root/'data/research'/info['array'], allow_pickle=False) as a:
        mosaic=a['mosaic']
    # All neighbours must remain present for the original Voronoi partition,
    # shape exclusions, row direction and background support.
    measured, channels=measure_frame(mosaic, info['black_level'], info['white_level'], info['camera_white_level'],
        info['peaks'], np.asarray(info['corners_raw_px']), info['reflections'], cfg=info['photometry_config'],
        green_to_red_ratio=None, green_linearity='WORKING_ASSUMPTION_NOT_RADIOMETRIC_CALIBRATION')
    reference=read_json(root/'data/research'/info['reference'])
    actual=next(q for q in measured if q['peak_uid']==sample['id'])
    expected=next(q for q in reference['photometry'] if q['peak_uid']==sample['id'])
    for key, value in expected.items():
        if isinstance(value, (int,float)):
            np.testing.assert_allclose(actual[key],value,rtol=2e-9,atol=1e-6)
        else:
            if actual[key]!=value:raise ValueError('测光质量标志与验收参考不同：'+key)
    chosen=[q for q in channels if q['peak_uid']==sample['id']]
    log_values('selected_photometry', measurement=actual, channels=chosen, neighbour_count=len(info['peaks']))
    write_json(out/'measurement.json',dict(photometry=actual,channels=chosen))
    recorder.result(metrics=[['光斑',sample['id']],['邻峰上下文',len(info['peaks'])],['质量标志',actual['quality_flags']]],
        table=dict(headers=['孔径', 'G1净积分', 'G2净积分', '平均G净积分'],rows=[['r'+str(r),actual[f'Q_G1_r{r}'],actual[f'Q_G2_r{r}'],actual[f'Q_G_r{r}']] for r in (50,125)]),
        links=[dict(label='测量与逐通道质量',path=recorder.path(out/'measurement.json'))])


def one_d_observations(root, sample):
    if sample['id']=='original':
        rows=read_csv(root/'data/legacy_original_23/one_dimensional_ratios.csv')
        return [q['frame_id'] for q in rows], np.array([float(q['voltage_RMS_V']) for q in rows]), np.log([[float(q['R21_observed']),float(q['R31_observed'])] for q in rows])
    cycle,direction=sample['id'].split('/')
    rows=read_csv(root/'data/runs/scan_2026-10-14_audit_v2/one_d_predictions.csv')
    unique={}
    for q in rows:
        if q['cycle_id']==cycle and q['direction']==direction and q['area']=='r125':
            values=(float(q['U_V']),float(q['observed_log_R21']),float(q['observed_log_R31']))
            if q['frame_id'] in unique and unique[q['frame_id']]!=values:raise ValueError('同帧观测不一致')
            unique[q['frame_id']]=values
    ids=sorted(unique);a=np.array([unique[fid] for fid in ids])
    return ids,a[:,0],a[:,1:]


def intensity_experiment(root, out, tool, sample, recorder):
    from .diffraction_models import fit_one_d,fit_spectrum
    if tool['id']=='intensity.two_d':
        cycle,fid=sample['id'].split('/')
        rows=[q for q in read_json(root/'data/paper/spectrum_nodes.json') if (q['cycle'],q['fid'])==(cycle,fid)]
        labels=np.array([[q['a'],q['b']] for q in rows]);observed=np.array([q['observed'] for q in rows])
        p,scale,pred,_=fit_spectrum(labels,observed,'bessel_low')
        # Compare only after a real fit, with no frozen parameters as inputs.
        np.testing.assert_allclose(pred,[q['fitted'] for q in rows],rtol=2e-6,atol=1e-5)
        norm=float(observed.max());error=float(np.linalg.norm(pred-observed)/np.linalg.norm(observed))
        log_values('selected_2d_spectrum',sample=sample['id'],labels=labels,observed=observed,parameters=p,scale=scale,prediction=pred,residual=pred-observed,NRMSE=error)
        image=out/'spectrum_fit.png';plots.plot(image,[dict(title=sample['label'],xlabel='retained node index',ylabel='signal / observed frame maximum',
            series=[plots.series('observed',np.arange(len(rows)),observed/norm,'scatter'),plots.series('double oblique fit',np.arange(len(rows)),pred/norm)])])
        write_json(out/'parameters.json',dict(parameters=p,scale=scale,NRMSE=error,labels=labels,observed=observed,prediction=pred))
        recorder.result(images=[recorder.path(image)],metrics=[['节点数',len(rows)],['相位幅度',p.tolist()],['相对谱误差',error]])
        return
    ids,U,y=one_d_observations(root,sample)
    if len(ids)<3 or len(np.unique(U))<3:
        raise ValueError('所选数据不足三个不同电压帧，不能识别共享的电压模型参数')
    if tool['id']=='intensity.comparison' and len(ids)<5:
        raise ValueError('相邻两帧留出后至少需要三个训练帧')
    # The declared voltage normalization is the original research convention.
    lo,hi=(5.69,8.20) if sample['id']=='original' or sample['id'].endswith('/up') else (3.70,5.74)
    z=(U-lo)/(hi-lo)
    log_values('selected_1d_observations',sample=sample['id'],frame_ids=ids,U=U,z=z,observed_log_ratios=y,voltage_normalization_V=[lo,hi])
    models=[];series=[[],[]]
    singles=(True,False) if tool['id']=='intensity.comparison' else (tool['id']=='intensity.single_sine',)
    for single in singles:
        label='single_sine' if single else 'double_harmonic'
        p,pred,predict=fit_one_d(z,y,single)
        score=dict(model=label,parameters=p.tolist(),train_log_RMSE=float(np.sqrt(np.mean((pred-y)**2))))
        if sample['id']=='original' and not single:
            np.testing.assert_allclose(p,read_json(root/'data/paper/one_d_reference.json')['parameters'],rtol=2e-6,atol=2e-8)
        if tool['id']=='intensity.comparison':
            folds=[]
            for k in range(len(ids)-1):
                keep=np.ones(len(ids),bool);keep[k:k+2]=False
                hp,_,fn=fit_one_d(z[keep],y[keep],single);held=fn(hp,z[~keep]);mse=float(np.mean((held-y[~keep])**2))
                folds.append(dict(held_frames=ids[k:k+2],parameters=hp.tolist(),MSE=mse))
                log_values('adjacent_frame_holdout',model=label,held_frames=ids[k:k+2],train_frames=[ids[i] for i in np.flatnonzero(keep)],parameters=hp,prediction=held,observed=y[~keep],MSE=mse)
            score['CV_log_RMSE']=float(np.sqrt(np.mean([q['MSE'] for q in folds])))
            write_json(out/(label+'_folds.json'),folds)
        models.append(score)
        order=np.argsort(U)
        for i in (0,1):series[i].append(plots.series(label,U[order],np.exp(pred[order,i])))
        # Actual fitted phase -> complex field -> Fourier intensities.
        mid=float(np.median(z));beta=p[0]*(1-mid)+p[1]*mid;u=np.arange(512)*2*np.pi/512
        phase=beta*np.cos(u)+(0 if single else p[2]*beta**2*np.cos(2*u));field=np.exp(1j*phase);weights=abs(np.fft.fft(field)/512)**2
        log_values('fitted_phase_Fourier',model=label,beta=beta,u=u,phase=phase,field=field,weights=weights)
        plots.plot(out/(label+'_phase.png'),[dict(title=label+' / median sampled voltage',xlabel='x / period',ylabel='phase (rad)',series=[plots.series('phase',u/(2*np.pi),phase)]),
            dict(title='Complex transmission',xlabel='x / period',ylabel='field',series=[plots.series('real',u/(2*np.pi),field.real),plots.series('imaginary',u/(2*np.pi),field.imag)]),
            dict(title='Fourier intensity weights',xlabel='optical order',ylabel='weight',series=[plots.series('squared modulus',np.arange(7),weights[:7],'scatter')])])
    image=out/'ratios.png';plots.plot(image,[dict(title=name,xlabel='RMS voltage (V)',ylabel='green signal ratio',series=[plots.series('observed',U,np.exp(y[:,i]),'scatter')]+series[i]) for i,name in enumerate(('H2/H1','H3/H1'))],columns=2)
    write_json(out/'models.json',models)
    recorder.result(images=[recorder.path(image)]+[recorder.path(out/(q['model']+'_phase.png')) for q in models],
        metrics=[['独立帧数',len(ids)]],table=dict(headers=['模型','参数','训练 log RMSE','相邻两帧留出 log RMSE'],rows=[[q['model'],q['parameters'],q['train_log_RMSE'],q.get('CV_log_RMSE')] for q in models]))


def period_experiment(root,out,tool,sample,recorder):
    from .diffraction_models import fit_geometry
    if tool['id']=='period.horizontal':
        cycle,fid=sample['id'].split('/')
        q=next(q for q in read_json(root/'data/paper/horizontal_periods.json') if (q['cycle'],q['frame_id'])==(cycle,fid))
        r=np.asarray(q['radius_mm']);sine=r/np.sqrt(q['distance_mm']**2+r*r);first=.6328/sine[0];three=14*.6328/np.dot([1,2,3],sine)
        log_values('peak_distance_to_period',radius_mm=r,distance_mm=q['distance_mm'],wavelength_um=.6328,sin_theta=sine,first_period_um=first,three_period_um=three)
        np.testing.assert_allclose([first,three],[q['first_period_um'],q['three_period_um']],rtol=2e-6)
        recorder.result(metrics=[['首峰周期 / µm',float(first)],['三峰共同周期 / µm',float(three)]],table=dict(headers=['峰对 n','半径 / mm','sinθ'],rows=[[i+1,float(a),float(b)] for i,(a,b) in enumerate(zip(r,sine))]))
    elif tool['id']=='period.geometry':
        cycle,fid=sample['id'].split('/')
        rr=[q for q in read_json(root/'data/paper/geometry_nodes.json') if (q['cycle'],q['frame_id'])==(cycle,fid)]
        labels=np.asarray([q['labels'] for q in rr]);xy=np.asarray([q['observed_s'] for q in rr]);scores=[];series=[plots.series('observed',xy[:,0],xy[:,1],'scatter')]
        for model in ('square','rectangular','affine'):
            p,pred=fit_geometry(labels,xy,model);held=[]
            for k in range(len(rr)):
                keep=np.arange(len(rr))!=k;hp,_=fit_geometry(labels[keep],xy[keep],model);held.append(np.r_[1,labels[k]]@hp)
            error=float(np.sqrt(np.mean(np.sum((xy-held)**2,axis=1)))/min(np.linalg.norm(p[1]),np.linalg.norm(p[2])))
            log_values('node_geometry_holdout',model=model,labels=labels,observed=xy,coefficients=p,prediction=pred,held_out_prediction=np.asarray(held),LOO_RMS_halfstep=error)
            scores.append([model,error]);series.append(plots.series(model,pred[:,0],pred[:,1],'scatter',radius=2))
            if model=='rectangular':
                ref=next(q for q in read_json(root/'data/paper/geometry_models.json') if (q['cycle'],q['frame_id'])==(cycle,fid))
                np.testing.assert_allclose(p,ref['coefficients'],rtol=2e-6,atol=1e-9)
        image=out/'geometry.png';plots.plot(image,[dict(title=sample['label'],xlabel='direction sine x',ylabel='direction sine y',equal_aspect=True,series=series)])
        recorder.result(images=[recorder.path(image)],metrics=[['实测节点',len(rr)]],table=dict(headers=['模型','条件留一误差 / 半步'],rows=scores))
    else:
        from .delivery_paper import period_summary
        rows=[q for q in read_json(root/'data/paper/microscopy_matches.json') if q['cycle']==sample['id']]
        summaries=period_summary(rows);log_values('selected_period_correspondence',input_matches=rows,frame_aggregates=summaries)
        recorder.result(table=dict(headers=['形态','显微ROI匹配','衍射帧','偏差中位数'],rows=[[q['kind'],q['n_roi_matches'],q['n_historical_frames'],q['residual_median']] for q in summaries]))


def observation_experiment(root,out,sample,recorder):
    kind,direction=sample['id'].split('/')
    if kind=='diffraction':
        rows=[q for q in read_json(root/'data/samples/diffraction/previews.json') if q['direction']==direction]
        frames=[(root/q['file'],f"{q['voltage_RMS_V']:.2f} V") for q in rows]
    else:
        rows=[q for q in read_json(root/'data/samples/microscopy_frames.json')['frames'] if q['sample']['direction']==direction]
        frames=[(root/'data/samples'/q['file'],f"{q['sample']['voltage_V']:.2f} V") for q in rows]
    log_values('selected_image_sequence',kind=kind,direction=direction,frames=rows)
    image=out/'sequence.png';plots.montage(image,frames)
    recorder.result(images=[recorder.path(image)],metrics=[['图像数',len(frames)],['本项操作','按记录排列原图，不新增测量']])


def discussion_experiment(root,out,tool,sample,recorder):
    from .delivery_algorithms import analyze_texture
    core=root/'data/core';item=next(q for q in read_json(core/'index.json')['microscopy'] if q['id']==sample['id'])
    with Image.open(core/item['image']) as im:whole=np.asarray(im.convert('RGB'))
    x,y,w,h=item['roi_xywh'];q,*_=analyze_texture(whole[y:y+h,x:x+w],*item['pixel_scale_um'])
    pair=q['reciprocal_basis_candidate']
    if tool['id']=='discussion.combination':
        if not pair or not q['basis_passes_global_screen']:
            recorder.result(metrics=[['判断','本区域没有合格二维频率基底，不能给出受支持的和差频']]);return
        a,b=[np.array([p['fx_cycles_um'],p['fy_cycles_um']]) for p in (pair['first'],pair['second'])]
        vectors=[a+b,a-b];lengths=[float(1/np.linalg.norm(v)) for v in vectors]
        log_values('measured_frequency_combinations',first=a,second=b,sum_frequency=vectors[0],difference_frequency=vectors[1],periods_um=lengths)
        rows=[[label,*v.tolist(),float(1/np.linalg.norm(v))] for label,v in zip(('输入频率1','输入频率2','和频','差频'),[a,b,*vectors])]
        recorder.result(metrics=[['局部共同支持分块',q['jointly_supported_tiles']],['解释范围','频率向量的数学组合，局部共存须另行核对']],table=dict(headers=['频率组合','fx / µm⁻¹','fy / µm⁻¹','周期 / µm'],rows=rows))
    else:
        family=q['primary_family']
        if not q['family_passes_quality_gate'] or not family:
            recorder.result(metrics=[['判断','本区域没有通过质量检查的主频率族，未生成倍频示意']]);return
        f=float(1/family['base']['period_um']);distance=np.linspace(0,3/f,513);theta=np.cos(2*np.pi*f*distance);square=theta**2
        error=float(np.max(abs(square-(1+np.cos(4*np.pi*f*distance))/2)))
        log_values('quadratic_response_illustration',measured_frequency_cycles_um=f,distance_um=distance,illustrative_orientation=theta,quadratic_response=square,identity_error=error)
        image=out/'doubling.png';plots.plot(image,[dict(title='Quadratic response / measured frequency as length scale',xlabel='distance (µm)',ylabel='illustrative normalized amplitude',series=[plots.series('cos(qx)',distance,theta),plots.series('cos²(qx)',distance,square)])])
        recorder.result(images=[recorder.path(image)],metrics=[['输入频率族周期 / µm',1/f],['二次响应示意周期 / µm',.5/f],['恒等式数值误差',error]])


def run_experiment(root,output,algorithm,sample_id,image=None,pixel_size_um=None,on_state=None):
    root=Path(root).resolve();verify_bundle(root,quiet=True)
    tool,sample=select_definition(root,algorithm,sample_id)
    out=new_output(root,output);recorder=IndependentRecorder(root,out,tool,sample,on_state)
    try:
        with console_values(),observe(recorder.record,recorder.accepts):
            log_values('independent_run',algorithm=algorithm,sample=sample_id,source=sample.get('source'),input_image=image,pixel_size_um=pixel_size_um)
            compute=out/'calculation'
            try:
                if tool['type'] in ('detection','texture'):
                    from .delivery_algorithms import diffraction_case,microscopy_case
                    core=root/'data/core';index=read_json(core/'index.json')
                    if tool['type']=='detection':
                        item=next(q for q in index['diffraction'] if q['id']==sample_id)
                        result=diffraction_case(core,item,compute)
                        if result.get('status')=='FAIL':raise ValueError('识别结果与验收参考不一致，请查看本次记录')
                    else:
                        if sample_id=='own_image':
                            if image is None or pixel_size_um is None or not np.isfinite(pixel_size_um) or pixel_size_um<=0:raise ValueError('请选择图像并填写有效标尺')
                            with Image.open(image) as im:w,h=im.size
                            item=dict(id='own_image',title='自己的显微图像',roi_xywh=[0,0,w,h],pixel_scale_um=[pixel_size_um]*2)
                        else:item=next(q for q in index['microscopy'] if q['id']==sample_id)
                        microscopy_case(core,item,compute,own_image=image)
                else:
                    compute.mkdir()
                    if tool['type']=='rg':rg_experiment(root,compute,tool,sample,recorder)
                    elif tool['type']=='photometry':photometry_experiment(root,compute,tool,sample,recorder)
                    elif tool['type']=='period':period_experiment(root,compute,tool,sample,recorder)
                    elif tool['type']=='intensity':intensity_experiment(root,compute,tool,sample,recorder)
                    elif tool['type']=='evolution':observation_experiment(root,compute,sample,recorder)
                    elif tool['type']=='discussion':discussion_experiment(root,compute,tool,sample,recorder)
            except TargetReady:
                pass
            if not any(q['status']!='superseded' for q in recorder.events):
                # A quality gate can legitimately prevent a profile. Report that
                # outcome instead of inventing a period or a successful image.
                recorder.result(metrics=[['判断','本样本未得到该算法所需的合格频率族，未生成剖面或唯一周期']])
        report=dict(status='PASS',algorithm=algorithm,sample=sample_id,events=recorder.events,
                    source_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src').rglob('*.py')},
                    user_image_sha256=hashlib.sha256(Path(image).read_bytes()).hexdigest() if image else None,
                    elapsed_s=time.monotonic()-recorder.started)
        write_json(out/'run.json',plain(report));recorder.publish(status='complete',message='本次计算完成',report=recorder.path(out/'run.json'))
        return report
    except BaseException as exc:
        (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        recorder.publish(status='failed',message=str(exc),error_log=recorder.path(out/'error.log'))
        raise
