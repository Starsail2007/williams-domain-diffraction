"""Final six-frame tail comparison and explicitly conditional model analysis."""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .green_channel_study import BASE, RAW, read_csv, save_csv, save_json, digest, aperture_weights
from .green_photometry_trial import (TRIAL, OLD, channel_patch, estimate_shape, classify,
                                    support_mask, fit_tiled_background)
from .geometry import display_to_raw


def region_cell(xx,yy,center,centers,inside):
    cell=inside.copy()
    for nx,ny in centers:
        if np.hypot(nx-center[0],ny-center[1])>3:
            cell &= xx*xx+yy*yy <= (xx+center[0]-nx)**2+(yy+center[1]-ny)**2
    return cell


def fit_power(U,y,p):
    """Two parameters only; zero background after explicit subtraction."""
    U=np.asarray(U,float);y=np.asarray(y,float)
    def evaluate(uc):
        x=np.maximum(U-uc,0)**p
        c=max(float(x@y/(x@x)),0)
        return float(np.sum((y-c*x)**2)),c
    result=minimize_scalar(lambda uc:evaluate(uc)[0],bounds=(0,float(U.min())-1e-6),method='bounded')
    candidates=[0.,float(result.x),float(U.min())-1e-6]
    uc=min(candidates,key=lambda a:evaluate(a)[0]);sse,c=evaluate(uc)
    total=float(np.sum((y-y.mean())**2))
    return {'exponent_fixed':p,'Uc_V':uc,'C':c,'SSE':sse,'R_squared':1-sse/total if total>0 else None,
            'n':len(U),'parameter_count':2,'Uc_at_boundary':uc==0 or uc==candidates[-1],
            'definition':'I=C*max(U-Uc,0)^p; conditional exploratory fit, not independently measured threshold'}


def run(config_path,out):
    import rawpy
    from .paths import require_writable_run
    out=require_writable_run(out)
    config_path=config_path.resolve();out=out.resolve()
    cfg=json.loads(config_path.read_text());base_cfg=json.loads((BASE/'annotations/photometry_trial_green_v1.json').read_text())
    baseline_dir=BASE/cfg['baseline_run'];previous=read_csv(baseline_dir/'measurements.csv')
    model=json.loads((baseline_dir/'frozen_prediction_model.json').read_text());a=model['slope'];b=model['intercept_DN'];s0=model['zero_intercept_slope']
    spots=read_csv(TRIAL/'spots.csv')
    spots += [{**r,'model_groups':'row'} for r in read_csv(OLD/'spots.csv') if r['frame_id']=='070']
    rejected=read_csv(TRIAL/'rejected_candidates.csv')
    samples={s['frame_id']:s for s in json.loads((BASE/'annotations/sample_config.json').read_text())['samples']}
    out.mkdir(parents=True,exist_ok=False);(out/'comparisons').mkdir()
    os.environ['MPLCONFIGDIR']=str(out/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Circle
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':10,'axes.unicode_minus':False,'savefig.dpi':155})
    table=[];curves=[];profiles=[];hashes=[]
    radii=cfg['growth_radii_raw_px'];adopted=cfg['adopted_aperture_radius_raw_px']
    for fid in cfg['sample_ids']:
        path=RAW/f'{fid}.ARW';before=digest(path)
        with rawpy.imread(str(path)) as raw:
            if raw.raw_pattern.tolist()!=[[0,1],[3,2]]:raise ValueError('Unexpected CFA')
            mosaic=raw.raw_image_visible.copy();black=raw.black_level_per_channel
        g=(mosaic[0::2,1::2].astype(float)-black[1]+mosaic[1::2,0::2].astype(float)-black[3])/2
        frame_spots=[r for r in spots if r['frame_id']==fid]
        centers=np.array([[float(r['u_raw_px']),float(r['v_raw_px'])] for r in frame_spots])
        anchor=int(np.argmax([float(r.get('contrast_DN',r.get('detection_contrast_DN',0))) for r in frame_spots]))
        near=centers[np.abs(centers[:,1]-centers[anchor,1])<80]
        angle=float(np.arctan2(*np.linalg.svd(near-near.mean(axis=0),full_matrices=False)[2][0][::-1]))%np.pi if len(near)>=3 else 0.
        shapes=[{'center':c,'shape':estimate_shape(g,c,classify(r,fid),angle,base_cfg)} for r,c in zip(frame_spots,centers)]
        reflections=np.array([[float(r['u_raw_px']),float(r['v_raw_px'])] for r in rejected if r['frame_id']==fid]).reshape(-1,2)
        corners=display_to_raw(samples[fid]['corners_display'],g.shape)
        for old in [r for r in previous if r['frame_id']==fid]:
            key=f'{fid}_{old["spot_id"]}';center=np.array([float(old['u_raw_px']),float(old['v_raw_px'])])
            rec={'key':key,'frame_id':fid,'spot_id':old['spot_id'],'voltage_RMS_V':float(old['voltage_RMS_V']),
                 'scan_direction':old['scan_direction'],'baseline_Q_G_r50':float(old['baseline_Q_G']),
                 'v3_Q_G_r50':float(old['new_r50_Q_G']),'v3_adaptive_Q_G':float(old['adaptive_Q_G']),
                 'v3_residual_fraction':float(old['frozen_model_r50_residual_fraction']),
                 'v3_plateau_found':old['plateau_found']=='True','source_quality_flags':old['quality_flags'],
                 'inference_basis':'USER_AUTHORIZED_GREEN_LINEARITY_ASSUMPTION'}
            methods=['v3','protected_same_degree','final_plane']
            if key in cfg['focus_keys']:methods.append('outer220_plane')
            data={};channel_growth={};channel_profiles={}
            try:
                for name in ('R','G1','G2'):
                    z,xx,yy,_=channel_patch(mosaic,black,center,base_cfg['background_patch_half_raw_px'],name)
                    mask,inside=support_mask(xx,yy,center,shapes,reflections,corners,base_cfg)
                    cell=region_cell(xx,yy,center,centers,inside);rr=np.hypot(xx,yy)
                    fields={}
                    for method in methods:
                        radius=90 if method=='v3' else (cfg['background_alternative_radius_raw_px'] if method=='outer220_plane' else cfg['background_protected_radius_raw_px'])
                        degree=int(old[name+'_bg_degree']) if method in ('v3','protected_same_degree') else cfg['final_background_degree']
                        fields[method]=fit_tiled_background(z,xx,yy,mask&(rr>=radius),degree,base_cfg)
                        w50=aperture_weights(xx,yy,50)
                        rec[f'{method}_{name}_r50']=float(4*np.sum(w50*(z-fields[method]['field'])))
                        rec[f'{method}_{name}_bg_center_DN']=float(fields[method]['coef'][0])
                        channel_growth[name,method]=[]
                        for radius_g in radii:
                            w=aperture_weights(xx,yy,radius_g);full=float(w.sum());w=w*cell
                            channel_growth[name,method].append({'radius':radius_g,'Q':float(4*np.sum(w*(z-fields[method]['field']))),
                                                               'coverage':float(w.sum()/full),'area':float(4*w.sum())})
                        channel_profiles[name,method]=[]
                        for lo,hi in ((0,50),(50,80),(80,110),(110,150),(150,180)):
                            use=(rr>=lo)&(rr<hi)&cell
                            channel_profiles[name,method].append(float(np.mean((z-fields[method]['field'])[use])) if np.any(use) else None)
                    data[name]=(z,xx,yy,fields,mask,cell)
                for method in methods:
                    Qg=(rec[f'{method}_G1_r50']+rec[f'{method}_G2_r50'])/2;Qr=rec[f'{method}_R_r50']
                    rec[f'{method}_Q_G_r50']=Qg;rec[f'{method}_residual_fraction']=(Qg-a*Qr-b)/(a*Qr+b) if a*Qr+b>0 else None
                    values=[]
                    for i,radius in enumerate(radii):
                        g1=channel_growth['G1',method][i];g2=channel_growth['G2',method][i];red=channel_growth['R',method][i]
                        Q=(g1['Q']+g2['Q'])/2
                        values.append({'key':key,'method':method,'radius_raw_px':radius,'Q_G':Q,'Q_R':red['Q'],
                                       'coverage':min(g1['coverage'],g2['coverage']),
                                       'increment_fraction':None if i==0 else (Q-values[-1]['Q_G'])/max(abs(Q),1e-12)})
                    curves.extend(values)
                    chosen=next(v for v in values if v['radius_raw_px']==adopted)
                    rec[f'{method}_Q_G_r125']=chosen['Q_G'];rec[f'{method}_coverage_r125']=chosen['coverage']
                    for i,(lo,hi) in enumerate(((0,50),(50,80),(80,110),(110,150),(150,180))):
                        G1=channel_profiles['G1',method][i];G2=channel_profiles['G2',method][i]
                        G=(G1+G2)/2 if G1 is not None and G2 is not None else None;R=channel_profiles['R',method][i]
                        profiles.append({'key':key,'method':method,'inner_radius_raw_px':lo,'outer_radius_raw_px':hi,
                                         'mean_net_G_DN':G,'mean_net_R_DN':R,'G_over_R':G/R if G is not None and R is not None and R>0 else None,
                                         'profile_status':'MEASURED' if G is not None and R is not None else 'EMPTY_NEIGHBOUR_SCREEN_CELL_RING'})
                rec['v3_recomputed_difference_Q_G']=rec['v3_Q_G_r50']-float(old['new_r50_Q_G'])
                if abs(rec['v3_recomputed_difference_Q_G'])>1e-5:raise ValueError('v3 background reproduction mismatch')
                rec['tail_protection_same_aperture_change_fraction']=(rec['protected_same_degree_Q_G_r50']/rec['v3_Q_G_r50']-1) if rec['v3_Q_G_r50'] else None
                rec['background_colour_error_change_Q']=((float(old['new_r50_Q_G'])-float(old['baseline_Q_G']))-a*(float(old['new_r50_Q_R'])-float(old['baseline_Q_R'])))
                rec['brightness_red_equivalent_r50']=(rec['final_plane_Q_G_r50']-b)/a
                rec['brightness_red_equivalent_r125']=rec['final_plane_Q_G_r125']/s0
                rec['adopted_aperture_radius_raw_px']=adopted
                rec['analysis_use_under_assumptions']=rec['final_plane_Q_G_r125']>0 and not any(f in old['quality_flags'] for f in ('REFLECTION_OVERLAP','SCREEN_EDGE','DIFFUSE_STATE'))
                if key in cfg['focus_keys']:
                    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
                    z,xx,yy,fields,mask,cell=data['G1']
                    extent=[float(xx.min()),float(xx.max()),float(yy.max()),float(yy.min())]
                    for ax,method in zip(axes[0],('v3','final_plane')):
                        ax.imshow(z-fields[method]['field'],extent=extent,cmap='coolwarm',vmin=-5,vmax=30)
                        ax.set(xlim=(-250,250),ylim=(250,-250),xlabel='相对中心 / RAW像素',ylabel='相对中心 / RAW像素')
                        ax.add_patch(Circle((0,0),125,fill=False,color='#efb340',lw=2))
                        ax.add_patch(Circle((0,0),90 if method=='v3' else 180,fill=False,color='#222',ls='--'))
                        ax.set_title('上一轮背景（橙圈统一显示r125）' if method=='v3' else '本轮：保护180像素尾部＋平面背景')
                    for method,label in [('v3','上一轮背景'),('protected_same_degree','只把背景区移远'),('final_plane','最终：远背景平面')]:
                        v=[c for c in curves if c['key']==key and c['method']==method]
                        axes[1,0].plot(radii,[c['Q_G']/1e6 for c in v],'o-',label=label)
                    axes[1,0].axhline(float(old['adaptive_Q_G'])/1e6,c='#888',ls='--',label='上一轮自适应积分')
                    axes[1,0].axvline(125,c='#444',ls=':')
                    axes[1,0].set(xlabel='固定圆孔径半径 / RAW像素',ylabel='绿色积分 / 10⁶ DN·RAW pixel²',title='扩大区域收回低幅翼；125像素处停止')
                    axes[1,0].legend(fontsize=9)
                    for method,label in [('v3','上一轮背景'),('final_plane','最终背景')]:
                        p=[r for r in profiles if r['key']==key and r['method']==method]
                        axes[1,1].plot([r['outer_radius_raw_px'] for r in p],[r['mean_net_G_DN'] for r in p],'o-',label=label)
                    axes[1,1].axhline(0,c='#666',lw=1)
                    axes[1,1].set(xlabel='环带外半径 / RAW像素',ylabel='环带平均净绿色 / DN',title='宽翼信号与背景；非阈值裁切')
                    axes[1,1].legend(fontsize=9)
                    fig.suptitle(f'{fid}/{old["spot_id"]} | 相同r50：只移远背景后QG变化 {rec["tail_protection_same_aperture_change_fraction"]*100:.2f}%\n旧模型残差 {rec["v3_residual_fraction"]*100:.2f}% → 最终平面 {rec["final_plane_residual_fraction"]*100:.2f}%',fontsize=13)
                    fig.savefig(out/'comparisons'/f'{key}.png',bbox_inches='tight');plt.close(fig)
            except ValueError as error:
                rec['measurement_error']=str(error);rec['analysis_use_under_assumptions']=False
            table.append(rec)
        after=digest(path)
        if before!=after:raise RuntimeError('RAW changed')
        hashes.append({'frame_id':fid,'sha256_before':before,'sha256_after':after})
        print(fid,'done',flush=True)
    save_csv(out/'measurements.csv',table);save_csv(out/'growth_curves.csv',curves);save_csv(out/'tail_profiles.csv',profiles)
    save_json(out/'raw_integrity.json',hashes);save_json(out/'run_config.json',cfg)
    selected=[next(r for r in table if r['key']==key) for key in cfg['up_model_keys']]
    if any('measurement_error' in r for r in selected):raise RuntimeError('Model demonstration inputs failed')
    U=np.array([r['voltage_RMS_V'] for r in selected]);y=np.array([r['final_plane_Q_G_r125']/1e6 for r in selected])
    fits=[fit_power(U,y,p) for p in (1,2)]
    save_json(out/'conditional_model_fit.json',{'selected_keys':cfg['up_model_keys'],'U_RMS_V':U.tolist(),'Q_G_r125_in_million':y.tolist(),'fits':fits,'green_response_linearity':'ASSUMED_BY_USER','full80_processed':False})
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
    for field,label in [('baseline_Q_G_r50','最初r50'),('v3_Q_G_r50','上一轮r50'),('final_plane_Q_G_r125','本轮r125')]:
        axes[0].plot(U,[r[field]/1e6 for r in selected],'o-',label=label)
    axes[0].set(xlabel='有效值电压 / V',ylabel='绿色积分 / 10⁶',title='同一右侧约13 mm候选峰：上升段')
    axes[0].legend()
    grid=np.linspace(float(U.min())-.1,float(U.max())+.1,200)
    axes[1].plot(U,y,'ko',label='本轮固定区域信号')
    for f in fits:
        axes[1].plot(grid,f['C']*np.maximum(grid-f['Uc_V'],0)**f['exponent_fixed'],label=f'p={f["exponent_fixed"]}，R²={f["R_squared"]:.3f}')
    axes[1].set(xlabel='有效值电压 / V',ylabel='绿色积分 / 10⁶',title='工作假设下：I=C(U−Uc)^p，仅3个样本')
    axes[1].legend();fig.savefig(out/'conditional_voltage_fit.png',bbox_inches='tight');fig.savefig(out/'conditional_voltage_fit.pdf',bbox_inches='tight');plt.close(fig)
    good=[r for r in table if 'measurement_error' not in r]
    summary={'sample_ids':cfg['sample_ids'],'measured_regions':len(good),'attempted_regions':len(table),
             'focus_results':[{k:r[k] for k in ('key','v3_Q_G_r50','protected_same_degree_Q_G_r50','tail_protection_same_aperture_change_fraction','v3_adaptive_Q_G','final_plane_Q_G_r125','final_plane_coverage_r125','v3_residual_fraction','final_plane_residual_fraction','outer220_plane_residual_fraction','background_colour_error_change_Q')} for r in table if r['key'] in cfg['focus_keys']],
             'adopted_rule':'off-peak background plane at r>=180; fixed r125 signed integration within neighbour/screen cell; green linearity assumed; no further platform requirement',
             'conditional_fits':fits,'full80_processed':False}
    save_json(out/'summary.json',summary)
    make_report(out,summary,table,selected,model)
    save_json(out/'source_hashes.json',{str(source_label(p)):digest(p) for p in [Path(__file__).resolve(),config_path,BASE/'annotations/analysis_assumptions_v1.json',PACKAGE_DIR/'green_photometry_trial.py',PACKAGE_DIR/'green_channel_study.py',baseline_dir/'measurements.csv',baseline_dir/'frozen_prediction_model.json',TRIAL/'spots.csv',TRIAL/'rejected_candidates.csv',OLD/'spots.csv',BASE/'annotations/sample_config.json']})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def make_report(out,summary,rows,selected,model):
    from .paths import require_writable_run
    out=require_writable_run(out)
    focus=summary['focus_results']
    by_key={r['key']:r for r in rows}
    decomposition=[]
    for r in focus:
        same=by_key[r['key']]['v3_Q_G_r125'];old=r['v3_adaptive_Q_G'];new=r['final_plane_Q_G_r125']
        decomposition.append([r['key'],f'{same/1e6:.4f}',f'{100*(same-old)/(new-old):.2f}%',f'{100*(new/same-1):.2f}%'])
    def tab(headers,values):
        return '<table><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in values)+'</table>'
    data=tab(['弱峰','只移远背景：r50变化','上一轮自适应Q / 10⁶','本轮r125 Q / 10⁶','旧模型残差：v3→平面','r125保留面积'],[[r['key'],f'{100*r["tail_protection_same_aperture_change_fraction"]:.2f}%',f'{r["v3_adaptive_Q_G"]/1e6:.4f}',f'{r["final_plane_Q_G_r125"]/1e6:.4f}',f'{100*r["v3_residual_fraction"]:.2f}% → {100*r["final_plane_residual_fraction"]:.2f}%',f'{100*r["final_plane_coverage_r125"]:.2f}%'] for r in focus])
    decomposition_table=tab(['弱峰','旧背景扩大到r125：Q / 10⁶','扩大区域解释的新增信号比例','同r125：再改背景的信号变化'],decomposition)
    fits=summary['conditional_fits']
    fit_table=tab(['模型（两个参数）','Uc / V','R²','样本数'],[[f'I=C(U−Uc)^{f["exponent_fixed"]}',f'{f["Uc_V"]:.3f}'+('（边界）' if f['Uc_at_boundary'] else ''),f'{f["R_squared"]:.4f}',f['n']] for f in fits])
    images=''.join(f'<figure><img src="comparisons/{r["key"]}.png"><figcaption>{r["key"]}：图像采用相同净绿色色标。黑虚线为背景保护内径，橙圈为固定125像素孔径。增长曲线保留邻峰/屏边分区，末端平坦不自动解释为收全。</figcaption></figure>' for r in focus)
    best=min(fits,key=lambda f:f['SSE'])
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>弱宽峰最后一轮优化与假设下模型分析</title><style>body{{font:17px/1.85 -apple-system,"PingFang SC",sans-serif;color:#23363d;background:#f3f6f7}}main{{max-width:1100px;margin:auto;padding:30px 24px 65px}}h1{{line-height:1.4}}h2{{margin-top:38px}}img{{max-width:100%}}figure{{margin:24px 0;background:white;padding:12px}}figcaption{{font-size:14px;color:#52666d}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px;border:1px solid #d9e2e5;text-align:left}}th{{background:#e1edf0}}.box{{background:#dfedef;padding:18px;border-radius:8px}}code,pre{{white-space:pre-wrap;font:14px/1.8 monospace;background:#e6ecee;padding:12px}}a{{color:#08758a}}.scroll{{overflow-x:auto}}</style><main>
<p>2026年10月10日 · 原6张代表图 · 22区域 · 本轮未处理全80张</p><h1>弱宽峰：最后一轮优化，然后按线性假设推进</h1>
<p class="box"><b>采用你的工作假设：绿色净积分与相对光功率成正比。</b>本轮将背景移到180像素以外、简化为平面，采用固定125像素孔径（保留邻峰/屏边分区），不再要求每个峰都找到严格平台。此规则用于后续探索性亮度与电压分析；算法优化到这里收住。</p>
    <h2>尾部是否被扣掉</h2><p>有低幅宽翼，但原先主要的问题是形态阈值把宽度估小，积分范围随之偏小。三弱峰上一轮最大积分椭圆半轴只有约52×69、53×38、57×106像素，低幅信号仍延伸至130–180像素。保持背景阶数、只将保护内径90移到180，中心r50净积分仅改变约0.2%–3.7%。这支持背景混入尾部不是中心信号损失的主要来源；026的外围还混有绿色背景分量，不能全算作光斑尾部。</p>
    <div class="scroll">{data}</div><p>再做同背景对照：先保持上一轮背景，只把积分区域扩大到r125；然后才改背景。区域扩大解释最终增量的89%–97%，直接支持积分范围偏小是主要原因。同r125下，背景改变仍影响约0.57%–8.51%，064最明显，不能把中心r50的变化比例泛化为整个宽区域。</p><div class="scroll">{decomposition_table}</div>
    <p>更大的区域收回更多低幅信号，也可能包含背景分量；这些增量不是“准确度提高倍数”。固定125像素提供简单可复现的区域定义。三张弱峰的r125完整保留；全部22区域中有9个被邻峰/屏边分区裁切，覆盖率随CSV保存，因此这些值是区域信号，不保证等面积或完整全峰功率。各区域保留原始负残差。</p>{images}
<h2>为什么更偏离旧G/R拟合</h2><p>同一r50下，三弱峰新旧颜色误差增量ΔG−0.11996ΔR分别约为{focus[0]['background_colour_error_change_Q']:.0f}、{focus[1]['background_colour_error_change_Q']:.0f}、{focus[2]['background_colour_error_change_Q']:.0f}积分单位。移远背景并不能普遍消除偏差；064的红背景原来使用二次面，改成远处平面后明显接近旧模型。其他峰还涉及背景颜色分量及弱翼颜色比变化。</p>
<p>070的宽翼G/R接近0.12，支持其含有真实激光尾部；026的外围比值逐渐升高，说明扩大区域同时收进了更多绿色背景分量。这里的颜色诊断只帮助解释，不用于反向挑背景来让拟合更好。旧模型在旧背景方案下训练，斜率关系仍有用，但原截距与本轮弱信号背景不完全兼容。可以继续按线性假设做分析，不必由这些残差阻止后续工作。</p>
<h2>亮度怎样换算，假设怎样使用</h2><pre>固定旧r50模型：Q_G=0.11996067 Q_R−2118.765
r50红等效信号：R_eq=(Q_G+2118.765)/0.11996067
本轮宽区域：R_eq=Q_G/0.119371358
相对亮度：I_rel=Q_G/Q_ref（参考区域明确且可用时）</pre><p>宽区域采用原来过原点的比例系数，不把固定r50截距搬到更大的面积。直接使用Q_G画趋势同样可以。线性换算只改变刻度；采用固定截距的仿射换算也只是线性缩放加常数，不能自行创造出新的电压二次关系。这里的“亮度”指扣背景的积分相对信号，暂用任意单位，不需要绝对瓦数才能拟合趋势。</p>
    <h2>现在就试算一次线性和二次关系</h2><p>选择上升段同一右侧约13 mm候选：014/020、018/014、026/005；有效值电压为5.69、5.87、7.04 V。在绿色线性及这三个区域可比的假设下，直接用本轮r125信号拟合I=C(U−Uc)和I=C(U−Uc)²；各模型均为两个参数，不另加自由背景截距。第一个区域保留97.15%的孔径面积，另外两个为100%，这里继续使用区域可比的工作假设。</p>
<figure><img src="conditional_voltage_fit.png"><figcaption>左图显示测量规则改变后的趋势；右图直接比较两个假设模型。上升和下降不混成一条曲线。</figcaption></figure><div class="scroll">{fit_table}</div>
<p>当前三点的残差更小的是p={best['exponent_fixed']}模型。这是实际试算结果，不能为了想要二次关系而改数据或强行写成二次。三个样本只够作趋势演示，Uc是外推拟合参数，不是已经测得的阈值。后续整组分析再用同一种区域指标逐段比较，就能回答二次关系在哪个电压区间适用。</p>
    <p>二次关系仍有明确的理论候选：若这些候选峰可按近法向弱相位光栅的主导偶数级（文献具体为第二级）处理，采用近阈值模式振幅A²∝ε、相位调制β∝A²和I₂∝β²，则绿色线性假设给出Q_G∝I₂∝A⁴∝ε²。文献使用ε=U/Uc−1，因此固定Uc时等价于I₂∝(U−Uc)²；若改用(U²−Uc²)/Uc²，两种ε在近阈值才成固定比例。<a href="https://arxiv.org/html/cond-mat/0209583v3">John等，式38及图8</a>。这是待数据比较的物理假设；波长未知不影响比较曲线形状。</p>
<h2>接下来怎么做</h2><p>本轮结束算法优化，保留“远背景平面＋固定宽区域＋绿色线性”这条探索路线。之后重点输出：峰位置随电压、相对信号随电压、同一分支的线性/二次模型比较，以及升降路径差异。显然的杂光重叠、屏边截断与弥散区采用独立区域记录，不把它们填成0。无需把其他标定细节当作模型分析的前置门槛。</p>
<p><a href="measurements.csv">22区域前后测量与亮度换算</a> · <a href="growth_curves.csv">固定孔径增长曲线</a> · <a href="tail_profiles.csv">宽翼颜色诊断</a> · <a href="conditional_model_fit.json">条件模型参数</a> · <a href="conditional_voltage_fit.pdf">矢量模型图</a> · <a href="summary.json">结果摘要</a></p>
<pre>'图像分析/.venv/bin/python' '图像分析/stage1/src/weak_tail_trial.py' --out '图像分析/stage1/outputs/weak_tail_rerun_NEW'</pre><p>假设保存于annotations/analysis_assumptions_v1.json；规则保存于annotations/weak_tail_trial_v1.json。所有6张RAW哈希前后一致，旧输出保留。</p></main></html>'''
    with (out/'report.html').open('w',encoding='utf-8') as f:f.write(doc)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=BASE/'annotations/weak_tail_trial_v1.json');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.out)
