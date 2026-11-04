"""Run existing algorithms with inspectable, optionally paused stage snapshots."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from .delivery_io import read_json, write_json, new_output, verify_bundle
from .guide_events import observe
from .runtime_trace import console_values, log_values
from . import algorithm_graphics as graphics
from . import research_figures as plots


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value


def atomic_json(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


class Recorder:
    def __init__(self, root, out, case, control=None):
        self.root, self.out, self.case, self.control = root, out, case, control
        self.events=[]; self.seen=set(); self.pass_index=1; self.geometry=None; self.red=None
        self.state=dict(status='running', case=case['id'], events=[], message='读取输入', version=0)
        self.expected={s['id'] for s in case['steps']}
        self.publish()

    def publish(self, **updates):
        self.state.update(updates)
        self.state['version']+=1
        self.state['events']=self.events
        atomic_json(self.out/'state.json', self.state)

    def accepts(self, name, data):
        prefix=self.case['type']
        if name.split('.')[0]!=prefix: return False
        if name.startswith('photometry.') and 'channel' in data:
            return data['peak']=='125_audit_p007' and data['channel']=='G1' and name not in self.seen
        if name.startswith('texture.') and name in self.seen:return False
        return True

    def path(self, p):
        return Path(p).resolve().relative_to(self.root).as_posix()

    def bitmap(self, folder, name, array):
        a=np.asarray(array,float);lo,hi=np.percentile(a,[1,99])
        im=Image.fromarray(np.uint8(np.clip((a-lo)/max(hi-lo,1e-12),0,1)*255))
        p=folder/(name+'.png');im.save(p)
        np.savez_compressed(folder/(name+'.npz'), values=a)
        return self.path(p)

    def point_image(self, folder, data):
        import cv2
        from .geometry import transform
        H=np.asarray(self.geometry['H_raw_to_screen_mm']);scale=7.5
        S=np.array([[scale,0,-.5],[0,scale,-.5],[0,0,1.]])
        im=Image.fromarray(cv2.warpPerspective(graphics.tone(self.red),S@H@np.diag([2.,2.,1.]),(900,900))).convert('RGB')
        draw=ImageDraw.Draw(im);rows=[]
        for p in data['points']:
            raw=(p['q']+data['lo'])*2;xy=transform([raw],H)[0]*scale-.5
            color='#53dfde' if p['origin']=='INDEPENDENT_STRONG' else '#ffb44e' if p['origin']=='INDEPENDENT_WEAK' else '#d389ff'
            x,y=xy;draw.ellipse((x-6,y-6,x+6,y+6),outline=color,width=2)
            rows.append(dict(raw_xy=raw.tolist(),origin=p['origin'],flags=p['review_flags']))
        path=folder/'candidates.png';im.save(path)
        write_json(folder/'candidate_snapshot.json',rows)
        return self.path(path),rows

    def record(self, name, data):
        if name.endswith('.progress'):
            self.publish(message=('正在完成 '+str(data.get('group',''))+' 的拟合与整帧检验' if name.startswith('rg') else f"{data['model']}：留出检验 {data['fold']}/22"))
            return
        if name=='detection.pass':
            self.pass_index=data['pass_index']
            if self.pass_index>1:
                for event in self.events:
                    if event['step'] in ('detection.contrast','detection.strong','detection.weak','detection.models','detection.guided'):
                        event['status']='superseded'
            self.publish(message=f'第 {self.pass_index} 轮识别；前轮候选重新检查');return
        if name=='detection.closure':
            write_json(self.out/'training_closure.json',plain(data))
            self.publish(message='训练证据检查稳定' if data['stable'] else '训练点需要隔离，重新进行识别')
            return
        step=name
        if name=='intensity.fit':step='intensity.'+data['model']
        if step not in self.expected:return
        self.seen.add(name)
        folder=self.out/f'step_{len(self.events)+1:02d}';folder.mkdir()
        images=[];metrics=[];table=None;evidence=[];links=[]
        # Inputs explicitly supplied by the guide runner.
        if name.endswith('.input') and name!='texture.input':
            images=[self.path(p) for p in data.get('images',[])]
            metrics=data.get('metrics',[]);table=data.get('table')
        elif name=='detection.geometry':
            self.geometry=data['geometry']
            with np.load(data['folder']/data['case']['input_arrays'],allow_pickle=False) as a:self.red=a['red'].copy()
            images=[self.path(data['image'])]
            metrics=[['条件屏边',len(self.geometry['conditional_edges'])],['数值映射误差 / mm',self.geometry['corner_mapping_error_mm']]]
        elif name=='detection.contrast':
            for sigma,(field,noise) in data['fields'].items():
                path=folder/f'contrast_{sigma}.png';graphics.contrast_view(field,data['valid'],f'Actual detector field / sigma {sigma}').save(path);images.append(self.path(path))
            metrics=[['实际检测尺度 / R采样',', '.join(map(str,data['fields']))]]
        elif name in ('detection.strong','detection.weak','detection.models','detection.guided'):
            picture,rows=self.point_image(folder,data);images=[picture]
            counts=Counter(q['origin'] for q in rows)
            metrics=[['强候选',counts['INDEPENDENT_STRONG']],['独立弱候选',counts['INDEPENDENT_WEAK']],['预测补检',sum(n for k,n in counts.items() if k.startswith('GUIDED'))]]
            if 'models' in data:
                table=dict(headers=['几何族','条件留一误差 / 间距','允许预测'],rows=[['二维点阵' if m['kind']=='lattice_2d' else '横向峰列',m.get('maximum_conditional_loo_fraction'),m.get('prediction_enabled',False)] for m in data['models']])
            if 'training' in data:metrics.append(['可参与几何训练',len(data['training']['eligible_raw_xy'])])
            if name=='detection.guided':evidence=self.local_evidence(folder,data['audit'])
        elif name=='detection.final':
            q=data['summary'];images=[self.path(data['out']/'05_guided_detection.png')]
            metrics=[['候选总数',q['candidates']],['待复核标记',q['review_required']],['质量标志',q['with_quality_flags']]]
            evidence=self.local_evidence(folder,data['audit'])
            links=[dict(label='坐标与来源',path=self.path(data['out']/'coordinates.csv')),dict(label='完整搜索证据',path=self.path(data['out']/'all_search_evidence.json'))]
        elif name=='texture.input':
            p=folder/'roi.png';Image.fromarray(data['roi']).save(p);images=[self.path(p)]
            whole=Image.fromarray(data['whole']);draw=ImageDraw.Draw(whole);x,y,w,h=data['item']['roi_xywh'];draw.rectangle((x,y,x+w,y+h),outline='#ffb940',width=4)
            p=folder/'whole.png';whole.save(p);images.insert(0,self.path(p))
            metrics=[['标尺 / µm·px⁻¹',data['item']['pixel_scale_um'][0]],['ROI像素',f'{w} × {h}']]
        elif name in ('texture.detrend','texture.window','photometry.channel'):
            images=[self.bitmap(folder,'array',data['array'])]
            if 'black' in data:metrics=[['G1黑电平 / DN',data['black']],['光斑','125_audit_p007']]
        elif name=='texture.spectrum':
            p=folder/'spectrum.png';graphics.fft_view(data['power'],data['fx'],data['fy'],data['peaks'],set()).save(p);images=[self.path(p)]
            self.spectrum=data
            table=dict(headers=['候选','周期 / µm','fx / µm⁻¹','fy / µm⁻¹'],rows=[[p['rank'],p['period_um'],p['fx_cycles_um'],p['fy_cycles_um']] for p in data['peaks'][:8]])
        elif name=='texture.family':
            primary=data['primary'];selected={p['rank'] for _,p in primary['members']} if primary else set()
            if hasattr(self,'spectrum'):
                a=self.spectrum;p=folder/'family.png';graphics.fft_view(a['power'],a['fx'],a['fy'],a['peaks'],selected).save(p);images=[self.path(p)]
            metrics=[['质量检查',data['quality']],['频率族有歧义',data['ambiguous']]]
            table=dict(headers=['谐波倍数','谱峰编号','周期 / µm'],rows=[[n,p['rank'],p['period_um']] for n,p in primary['members']] if primary else [])
        elif name=='texture.checks':
            q=data['result'];im=Image.fromarray(data['image']);w,h=im.size;draw=ImageDraw.Draw(im);draw.line((w//2,0,w//2,h),fill='white',width=3);draw.line((0,h//2,w,h//2),fill='white',width=3)
            p=folder/'tiles.png';im.save(p);images=[self.path(p)]
            metrics=[['二维共同支持',str(q['jointly_supported_tiles'])+'/4' if q['spatial_pair_applicable'] else '不适用：没有合格二维基底']]
            table=dict(headers=['裁剪','频率角色','相对频率差','匹配'],rows=[[c['variant'],c['role'],c['relative_frequency_difference'],c['matched']] for c in q['central_crop_checks']])
        elif name=='texture.profile':
            q=data['profile'];period=data['selected_period_um'];use=q['lag_um']<=min(3*period,q['lag_um'][-1]/3)
            p=folder/'profile.png';plots.plot(p,[dict(title='Profile along the selected normal',xlabel='distance (um)',ylabel='G / DN',series=[plots.series('native profile',q['distance_um'],q['values'])]),dict(title='Linear autocorrelation / short lags',xlabel='lag (um)',ylabel='ACF',series=[plots.series('ACF',q['lag_um'][use],q['acf'][use])])]);images=[self.path(p)]
            metrics=[['所选谱峰周期 / µm',period],['附近ACF峰 / µm',q['acf_period_um']]]
        elif name=='texture.final':
            q=data['summary'];metrics=[['最强谱峰周期 / µm',q['strongest_period_um']],['通过质量检查的频率族 / µm',q['family_period_um']],['频率族歧义',q['family_ambiguous']]]
            links=[dict(label='全部频率候选',path=self.path(data['out']/'frequency_candidates.csv'))]
        elif name=='rg.selection':
            table=dict(headers=['选样结果','候选数'],rows=list(data['counts'].items()))
            keep=[q for q in data['rows'] if q['selection']=='included'];metrics=[['总候选',len(data['rows'])],['纳入',len(keep)]]
            self.rg_rows=keep
            p=folder/'selected.png';rr=[q for q in keep if q['group']=='cycle02'];plots.plot(p,[dict(title='Experiment 3 / selected signed integrals',xlabel='Q_R / million DN px2',ylabel='Q_G / million DN px2',series=[plots.series('selected',np.array([q['Q_R_r125'] for q in rr])/1e6,np.array([q['Q_G_r125'] for q in rr])/1e6,'scatter')])]);images=[self.path(p)]
        elif name=='rg.fitted':
            q=[x for x in data['scores'] if x['area']=='r125' and x['model']=='affine' and x['subset']=='all_unsaturated' and x['response']=='G_from_R' and x['group']!='pooled']
            table=dict(headers=['实验','光斑','斜率','R²','整帧留出 R²'],rows=[[{'original':'1','cycle01':'2','cycle02':'3'}[x['group']],x['n'],x['slope'],x['R2'],x['LOFO_R2']] for x in q])
            metrics=[['完成拟合组数',len(data['scores'])]]
        elif name=='rg.final':
            images=[self.path(data['out']/s) for s in ('02_rg_and_residual.png','04_weak_signal.png','03_three_experiments.png')]
            metrics=[['按帧重采样次数',data['result']['bootstrap_replicates']]]
            links=[dict(label='拟合、留出和区间',path=self.path(data['out']/'fits.csv'))]
        elif name=='photometry.background':
            images=[self.bitmap(folder,'background',data['array']),self.bitmap(folder,'support',data['mask'])]
            metrics=[['平面常数 / DN',float(data['coefficients'][0])]]
        elif name=='photometry.integral':
            images=[self.bitmap(folder,'weights',data['weights']),self.bitmap(folder,'residual',data['residual'])]
            metrics=[['G1净积分 / DN·RAW px²',data['integral']],['实际支持面积 / RAW px²',data['area']]]
        elif name=='photometry.final':
            images=[self.path(data['out']/s) for s in ('125_audit_p007_measurement.png','125_audit_p013_aperture.png')]
            metrics=[['光斑数',data['result']['peaks']],['通道记录',data['result']['channels']],['对原验收积分最大差',data['result']['maximum_reference_difference']]]
            table=dict(headers=['光斑','G1净积分','G2净积分','平均G净积分'],rows=[[q['peak_uid'],q['Q_G1_r125'],q['Q_G2_r125'],q['Q_G_r125']] for q in data['result']['target']])
            links=[dict(label='完整积分与质量标志',path=self.path(data['out']/'net_integrals.csv'))]
        elif name=='period.horizontal':
            images=[self.path(data['out']/'01_diffraction_periods.png')];q=next(q for q in data['inputs'] if q['cycle']=='cycle02');radius=np.array(q['radius_mm']);sine=radius/np.sqrt(q['distance_mm']**2+radius**2)
            table=dict(headers=['峰对 n','峰距 r / mm','sinθ = r / √(L²+r²)'],rows=[[i+1,float(r),float(s)] for i,(r,s) in enumerate(zip(radius,sine))])
            metrics=[['代表帧',q['frame_id']],['L / mm',q['distance_mm']],['首对峰周期 / µm',.6328/sine[0]],['三对峰周期 / µm',14*.6328/np.dot([1,2,3],sine)],['全部帧',len(data['rows'])]]
        elif name=='period.geometry':
            images=[self.path(data['out']/'02_lattice.png')];q=data['geometry'];metrics=[['几何帧',q['frames']],['节点',q['nodes']],['条件留一拟合',q['conditional_LOO_fits']]]
            table=dict(headers=['模型','留一误差 / 半步，中位数'],rows=[[m,float(np.median([r['LOO_RMS_halfstep'] for r in q['model_comparison'] if r['model']==m]))] for m in ('square','rectangular','affine')])
        elif name=='period.comparison':
            images=[self.path(data['out']/'03_period_comparison.png')]
            table=dict(headers=['实验组','配对衍射帧数','偏差中位数'],rows=[[q['cycle'],q['n_historical_frames'],f"{q['residual_median']*100:+.2f}%"] for q in data['comparison']['summary']])
        elif name=='intensity.fit':
            p=data['parameters'];z=.5;beta=p[0]*(1-z)+p[1]*z;u=np.arange(512)*2*np.pi/512
            phase=beta*np.cos(u)+(0 if len(p)==2 else p[2]*beta**2*np.cos(2*u));field=np.exp(1j*phase);weights=abs(np.fft.fft(field)/512)**2
            # Verify the exposed FFT against the production model's own prediction.
            from .diffraction_models import one_d_prediction
            from scipy.special import jv
            expected=np.log(jv(np.arange(1,4),beta)[1:]**2/jv(1,beta)**2) if len(p)==2 else one_d_prediction(p,np.array([z]))[0]
            np.testing.assert_allclose(np.log(weights[2:4]/weights[1]),expected,rtol=1e-10,atol=1e-10)
            path=folder/'phase_to_spectrum.png';plots.plot(path,[dict(title=data['model']+' / fitted mid-voltage phase',xlabel='x / period',ylabel='phase (rad)',series=[plots.series('phase',u/(2*np.pi),phase)]),dict(title='Complex transmission exp(i phase)',xlabel='x / period',ylabel='field component',series=[plots.series('real',u/(2*np.pi),field.real),plots.series('imaginary',u/(2*np.pi),field.imag)]),dict(title='Fourier coefficients squared',xlabel='optical order',ylabel='normalized weight',series=[plots.series('intensity',np.arange(7),weights[:7],'scatter',radius=6)])]);images=[self.path(path)]
            metrics=[['示范电压 / V',6.945],['β',beta],['H2/H1',weights[2]/weights[1]],['H3/H1',weights[3]/weights[1]]]
            np.savez_compressed(folder/'phase_to_spectrum.npz',u=u,phase=phase,field=field,weights=weights)
        elif name=='intensity.comparison':
            images=[self.path(data['out']/'01_one_d_models.png')]
            table=dict(headers=['模型','训练 log RMSE','相邻两帧留出 log RMSE'],rows=[[q['model'],q['train_log_RMSE'],q['adjacent_two_frame_CV_log_RMSE']] for q in data['models']])
            links=[dict(label='留出帧与全部参数',path=self.path(data['out']/'one_d_folds.json'))]
        elif name=='intensity.two_d':
            images=[self.path(data['out']/'02_two_d_spectra.png')]
            table=dict(headers=['实验组','帧数','亮节点','相对谱误差中位数'],rows=[[q['cycle'],q['frames'],q['nodes'],q['NRMSE_median']] for q in data['result']['summary']])
        elif name=='discussion.doubling':
            images=[self.path(data['out']/'01_phase_doubling.png')];metrics=[['恒等式数值误差',data['error']]]
        elif name=='discussion.combination':
            q=data['result']['recomputed_microscopy_lengths_um'];metrics=[['和差频对应长度 / µm',q['combination']],['另一条纹区 / µm',q['stripe']]]
        elif name=='evolution.final':
            images=[self.path(self.out/'calculation'/p) for p in ('diffraction_up.png','diffraction_down.png','microscopy_up.png','microscopy_down.png')]
            metrics=[['本步操作','按记录排列图样；没有新增测量']]
        event=dict(step=step,sequence=len(self.events)+1,pass_index=self.pass_index,images=images,metrics=metrics,table=table,evidence=evidence,links=links,elapsed_s=round(time.monotonic()-self.started,3),status='computed')
        write_json(folder/'snapshot.json',plain(event));self.events.append(plain(event))
        pause=bool(self.control) and step!=self.case['steps'][-1]['id']
        self.publish(status='waiting' if pause else 'running',message='本步已完成',current_step=step)
        if pause:
            while True:
                command=read_json(self.control)
                if command.get('cancel'):raise InterruptedError('用户停止本次计算')
                if command.get('all') or command.get('through',0)>=len(self.events):break
                time.sleep(.15)
            self.publish(status='running',message='计算下一阶段')

    def local_evidence(self, folder, audit):
        candidates=[a for a in audit if 'predicted_raw_xy' in a]
        chosen=[]
        for accepted in (True,False):
            q=next((a for a in candidates if (a.get('status')=='DETECTED')==accepted and ('measured_raw_xy' in a if accepted else a.get('status')=='NO_ACCEPTED_LOCAL_MAXIMUM')),None)
            if q is None:continue
            xy=np.asarray(q['predicted_raw_xy'])/2;x,y=np.rint(xy).astype(int)
            left,top=max(0,x-24),max(0,y-24);a=self.red[top:y+25,left:x+25]
            if not a.size:continue
            path=self.bitmap(folder,'accepted' if accepted else 'rejected',a)
            # Mark exact coordinates on a nearest-neighbour enlargement. The
            # unmodified scientific crop remains in the adjacent NPZ file.
            with Image.open(self.root/path) as im:marked=im.convert('RGB').resize((a.shape[1]*6,a.shape[0]*6),Image.Resampling.NEAREST)
            draw=ImageDraw.Draw(marked)
            cx,cy=(xy-np.array([left,top])+.5)*6
            draw.line((cx-10,cy,cx+10,cy),fill='#ffcf55',width=2);draw.line((cx,cy-10,cx,cy+10),fill='#ffcf55',width=2)
            if accepted:
                mx,my=(np.asarray(q['measured_raw_xy'])/2-np.array([left,top])+.5)*6
                draw.ellipse((mx-14,my-14,mx+14,my+14),outline='#53dfde',width=2)
            marked.save(self.root/path)
            chosen.append(dict(label='找到实测光斑' if accepted else '预测位置被拒绝',image=path,record=plain(q)))
        return chosen


def run_guided(root, output, case_id, control=None, image=None, pixel_size_um=None):
    root=Path(root).resolve();verify_bundle(root,quiet=True);out=new_output(root,output)
    catalog=read_json(root/'configs/guide.json');case=next(q for q in catalog['cases'] if q['id']==case_id)
    recorder=Recorder(root,out,case,Path(control) if control else None);recorder.started=time.monotonic()
    try:
        with console_values(), observe(recorder.record,recorder.accepts):
            log_values('run_inputs',case=case_id,kind=case['type'],input_image=image,pixel_size_um=pixel_size_um)
            from .delivery_algorithms import diffraction_case,microscopy_case
            core=root/'data/core';index=read_json(core/'index.json');compute=out/'calculation'
            if case['type']=='detection':
                item=next(q for q in index['diffraction'] if q['id']==case['sample']);info=read_json(core/item['case'])
                recorder.record('detection.input',dict(images=[core/Path(item['case']).parent/info['raw_display']],metrics=[['实验','3'],['帧号',case['sample']],['电压 / V',info['voltage_RMS_V']]]))
                result=diffraction_case(core,item,compute)
            elif case['type']=='texture':
                if image:
                    if pixel_size_um is None or not np.isfinite(pixel_size_um) or pixel_size_um<=0:raise ValueError('请输入有限的正数标尺（µm/px）')
                    with Image.open(image) as im:w,h=im.size
                    item=dict(id='own_image',title='自己的显微图像',roi_xywh=[0,0,w,h],pixel_scale_um=[pixel_size_um]*2)
                else:item=next(q for q in index['microscopy'] if q['id']==case['sample'])
                result=microscopy_case(core,item,compute,own_image=image)
            elif case['type'] in ('rg','photometry'):
                from .research_measurement import run_rg,run_photometry,native_profiles
                info=read_json(root/'data/research/photometry125.json');preview=out/'input';preview.mkdir()
                with np.load(root/'data/research'/info['array'],allow_pickle=False) as a:native_profiles(a['mosaic'],info,preview)
                recorder.record(case['type']+'.input',dict(images=[preview/'01_clipping.png'],metrics=[['剖面来源','实验3，第125帧'],['电压 / V',6.77]]))
                result=(run_rg if case['type']=='rg' else run_photometry)(root,compute)
            else:
                from . import research_analysis
                compute.mkdir()
                prefix=case['type'];input_path=root/case['preview']
                initial=dict(images=[],metrics=[['输入身份',case['identity']]])
                if prefix=='period':
                    inputs=read_json(root/'data/paper/horizontal_periods.json');q=next(q for q in inputs if q['cycle']=='cycle02')
                    initial['table']=dict(headers=['输入项','值'],rows=[['实验3代表帧',q['frame_id']],['峰距 / mm',', '.join(map(str,q['radius_mm']))],['屏距 / mm',q['distance_mm']],['波长 / µm',.6328]])
                elif prefix=='intensity':
                    from .delivery_io import read_csv
                    rows=read_csv(root/'data/legacy_original_23/one_dimensional_ratios.csv');p=out/'observations.png'
                    plots.plot(p,[dict(title='Two observed ratios / 23 frames',xlabel='RMS voltage (V)',ylabel='green signal ratio',series=[plots.series(key,[float(q['voltage_RMS_V']) for q in rows],[float(q[key]) for q in rows],'scatter',radius=5) for key in ('R21_observed','R31_observed')])]);initial['images']=[p]
                elif prefix=='evolution':
                    first=read_json(root/'data/samples/diffraction/previews.json')[0];initial['images']=[root/first['file']];initial['metrics'].append(['图中电压 / V',first['voltage_RMS_V']])
                recorder.record(prefix+'.input',initial)
                body,result=getattr(research_analysis,case['function'])(root,compute)
                from .research_pages import write_page
                write_page(compute/'index.html',case['title'],'<h1>'+case['title']+'</h1>'+body)
                if prefix=='evolution':recorder.record('evolution.final',dict())
        if result.get('status')=='FAIL':raise ValueError('计算与验收参考不一致，请查看详细记录')
        source_files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src').rglob('*.py')}
        report=dict(status='PASS',case=case_id,result=result,events=recorder.events,source_sha256=source_files,input_manifest_sha256=hashlib.sha256((root/'provenance/input_manifest.json').read_bytes()).hexdigest(),user_image_sha256=hashlib.sha256(Path(image).read_bytes()).hexdigest() if image else None,recording='Snapshots copied from actual calculation stages; no timing animation',elapsed_s=time.monotonic()-recorder.started)
        write_json(out/'run.json',plain(report));recorder.publish(status='complete',message='本次计算完成',report=recorder.path(out/'run.json'))
        return report
    except BaseException as exc:
        (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        recorder.publish(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',message=str(exc),error_log=recorder.path(out/'error.log'))
        raise
