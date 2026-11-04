"""Observation, spatial periods, model comparison and physical interpretation."""
from collections import defaultdict
import json
import numpy as np
from .delivery_io import read_json,read_csv,write_json,digest
from .guide_events import emit
from .diffraction_models import fit_geometry,fit_one_d,one_d_prediction
from .research_measurement import csv_rows,GROUPS
from . import research_pages as pages
from . import research_figures as plots

def evolution(root,out):
    diffraction=read_json(root/'data/samples/diffraction/previews.json');micro=read_json(root/'data/samples/microscopy_frames.json')
    exports={q['destination']:q['export_sha256'] for q in read_json(root/'provenance/artifact_map.json')}
    for q in diffraction:
        if digest(root/q['file'])!=exports[q['file']]:raise ValueError('Evolution display changed')
    for q in micro['frames']:
        if digest(root/'data/samples'/q['file'])!=q['receipt']['sha256']:raise ValueError('Native microscopy frame changed')
    for direction in ('up','down'):
        rr=[q for q in diffraction if q['direction']==direction]
        plots.montage(out/f'diffraction_{direction}.png',[(root/q['file'],f'{q["voltage_RMS_V"]:.2f} V') for q in rr])
        mm=[q for q in micro['frames'] if q['sample']['direction']==direction]
        plots.montage(out/f'microscopy_{direction}.png',[(root/'data/samples'/q['file'],f'{q["sample"]["voltage_V"]:.2f} V') for q in mm])
    write_json(out/'image_sequence.json',dict(diffraction=diffraction,microscopy=micro,operation='Rearrange existing display/native frames in recorded scan order; no measurement or temporal inference'))
    body='<p>建立测量方法以后，先回到图像本身。这里按记录顺序排列实验3的升降压照片；显微扫描分时进行，用来观察结构变化。</p><h2>升压：二维点阵到横向峰列</h2>'
    body+=pages.figure(out,out/'diffraction_up.png','升压过程中，小二维点阵出现，随后与横向宽峰共存，再以横向峰列为主。高压下光斑展宽。')
    body+=pages.figure(out,out/'microscopy_up.png','同一液晶盒的一次显微升压扫描：格纹与条纹分区共存，随后条纹占主导。显微照片与衍射照片不是同时拍摄。')
    body+='<h2>降压没有重走同样的过程</h2>'
    body+=pages.figure(out,out/'diffraction_down.png','降压时可见横向峰列，但没有重现升压中的小二维点阵。相近电压不一定给出相同图样。')
    body+=pages.figure(out,out/'microscopy_down.png','显微降压中先出现近均匀区域和稀疏折线，再出现较密条纹。')
    body+='<p>这提出了两个需要定量回答的问题：<b>各区域的周期有多大？各级光强的配比怎样变化？</b>扫描路径差异可以观察到；现有等待记录还不能独立确认稳态滞回。</p>'
    body+=pages.details('图像与实验记录',pages.a(out,out/'image_sequence.json','帧序列、电压和来源')+'<p>本节重新生成图组，不声称进行了新测量。后续周期与模型计算使用各自明确的实验子集。</p>')
    return body,dict(status='PASS',reader_result='观察二维点阵、横向峰列及升降压差异，由此提出周期和光强问题。',display_frames=len(diffraction),native_microscopy_frames=len(micro['frames']),operation='display arrangement, not measurement')

def geometry_comparison(root,out):
    groups=defaultdict(list)
    for q in read_json(root/'data/paper/geometry_nodes.json'):groups[q['cycle'],q['frame_id']].append(q)
    refs={(q['cycle'],q['frame_id']):q for q in read_json(root/'data/paper/geometry_models.json')}
    scores=[];nodes=[];periods=[]
    for key,rr in groups.items():
        labels=np.array([q['labels'] for q in rr]);xy=np.array([q['observed_s'] for q in rr])
        for model in ('square','rectangular','affine'):
            coefficients,pred=fit_geometry(labels,xy,model)
            held=[]
            for k in range(len(rr)):
                keep=np.arange(len(rr))!=k;c,_=fit_geometry(labels[keep],xy[keep],model);held.append(np.r_[1,labels[k]]@c)
            halfstep=min(np.linalg.norm(coefficients[1]),np.linalg.norm(coefficients[2]))
            score=float(np.sqrt(np.mean(np.sum((xy-np.asarray(held))**2,axis=1)))/halfstep)
            scores.append(dict(cycle=key[0],frame_id=key[1],model=model,n=len(rr),LOO_RMS_halfstep=score))
            if model=='rectangular':
                period=float(.6328/np.linalg.norm(coefficients[1]+coefficients[2]))
                np.testing.assert_allclose(coefficients,refs[key]['coefficients'],rtol=2e-6,atol=1e-9)
                np.testing.assert_allclose(period,refs[key]['period_um'],rtol=2e-6)
                periods.append(dict(cycle=key[0],frame_id=key[1],period_um=period,coefficients=coefficients.tolist()))
                nodes.extend(dict(cycle=key[0],frame_id=key[1],labels=q['labels'],observed_s=q['observed_s'],fitted_s=p.tolist()) for q,p in zip(rr,pred))
    csv_rows(out/'geometry_model_comparison.csv',scores);write_json(out/'geometry_nodes.json',nodes);write_json(out/'geometry_periods.json',periods)
    panels=[]
    for cycle in ('original','cycle01','cycle02'):
        fid=next(q['frame_id'] for q in periods if q['cycle']==cycle);rr=[q for q in nodes if q['cycle']==cycle and q['frame_id']==fid]
        xy=np.asarray([q['observed_s'] for q in rr]);pred=np.asarray([q['fitted_s'] for q in rr])
        panels.append(dict(title=GROUPS[cycle].replace('实验','Experiment ')+' / '+fid,xlabel='direction sine x',ylabel='direction sine y',equal_aspect=True,series=[plots.series('observed',xy[:,0],xy[:,1],'scatter',radius=4),plots.series('centered rectangular',pred[:,0],pred[:,1],'scatter',radius=2)]))
    plots.plot(out/'02_lattice.png',panels,columns=3)
    return dict(frames=len(periods),nodes=len(nodes),conditional_LOO_fits=sum(q['n'] for q in scores),model_comparison=scores)

def periods(root,out):
    from .delivery_paper import table2
    from .delivery_algorithms import microscopy_case
    core=root/'data/core';index=read_json(core/'index.json');texture=[];(out/'texture').mkdir()
    for item in index['microscopy']:texture.append(microscopy_case(core,item,out/'texture'/item['id']))
    horizontal=[]
    for ref in read_json(root/'data/paper/horizontal_periods.json'):
        radius=np.asarray(ref['radius_mm']);sine=radius/np.sqrt(ref['distance_mm']**2+radius**2)
        first=.6328/sine[0];three=14*.6328/np.dot([1,2,3],sine)
        np.testing.assert_allclose([first,three],[ref['first_period_um'],ref['three_period_um']],rtol=2e-6)
        horizontal.append(dict(cycle=ref['cycle'],frame_id=ref['frame_id'],direction=ref['direction'],U_V=ref['voltage_V'],first_period_um=float(first),three_period_um=float(three),distance_mm=ref['distance_mm']))
    csv_rows(out/'horizontal_periods.csv',horizontal)
    panels=[]
    for cycle in ('cycle01','cycle02'):
        ss=[]
        for direction in ('up','down'):
            rr=[q for q in horizontal if q['cycle']==cycle and q['direction']==direction]
            for name in ('first_period_um','three_period_um'):
                ss.append(plots.series(direction+(' / first pair' if name.startswith('first') else ' / 3 pairs'),[q['U_V'] for q in rr],[q[name] for q in rr]))
        panels.append(dict(title=GROUPS[cycle].replace('实验','Experiment ')+': period estimates',xlabel='RMS voltage (V)',ylabel='optical period (um)',series=ss))
    plots.plot(out/'01_diffraction_periods.png',panels,columns=2)
    emit('period.horizontal', rows=horizontal, inputs=read_json(root/'data/paper/horizontal_periods.json'), out=out)
    geo=geometry_comparison(root,out)
    emit('period.geometry', geometry=geo, out=out)
    comparison=table2(root,out)
    rows=comparison['summary'];points={q['cycle']:q['frames'] for q in rows}
    plots.plot(out/'03_period_comparison.png',[dict(title='Microscopy compared with the declared optical correspondence',xlabel='k x diffraction period (um)',ylabel='microscopy period (um)',series=[plots.series(GROUPS[cycle].replace('实验','Experiment '),[(2 if cycle=='original' else 1)*r['d_um'] for r in rr],[r['p_um'] for r in rr],'scatter',radius=5) for cycle,rr in points.items()]+[plots.series('equal periods',[10,15],[10,15],color='#889c9b')])])
    write_json(out/'periods.json',dict(horizontal_frames=len(horizontal),geometry=geo,comparison=comparison,texture=texture))
    emit('period.comparison', comparison=comparison, out=out)
    body='<h2>从衍射峰距计算周期</h2><p>光屏毫米坐标先换算成衍射角，再结合波长632.8 nm计算光学周期。实验2、3使用前三对峰共同估计，减少单个首峰偏移的影响；盒屏距采用独立确认的30 cm。</p>'
    body+=pages.figure(out,out/'01_diffraction_periods.png','首对峰与三对峰估计的比较。升降方向分开；局部跳变保留，需结合原图判断。距离来自实验记录，不通过拟合显微周期调节。')
    body+=pages.figure(out,out/'02_lattice.png','二维点阵以中心矩形几何描述。这里拟合实测光斑位置，不反过来要求照片中的点阵方正。')
    body+='<h2>从显微频谱测量纹理周期</h2><p>去除缓慢变化的背景、加Hann窗，再计算二维傅里叶频谱。频率向量给出纹理法向，其长度的倒数给出周期。</p>'
    body+='<div class="gallery">'+pages.figure(out,out/'texture/micro164_left/02_roi.png','5.39 V同帧左侧格纹。')+pages.figure(out,out/'texture/micro164_left/03_spectrum.png','格纹给出两组非平行频率，进一步用裁剪和分块检验。')+'</div>'
    body+='<p>不能总把最亮频谱峰当作基峰：196帧右区最强峰约6.16 µm，而支持的频率族基峰约12.29 µm。164帧中区虽有全局双峰，四分块却没有共同支持两种模态。</p>'
    body+=pages.details('谐波、空间混合与更多频谱案例',''.join('<p>'+pages.a(out,out/'texture'/q['id']/'index.html',q['title'])+'</p>' for q in texture))
    body+='<h2>比较两种周期</h2>'
    body+=pages.figure(out,out/'03_period_comparison.png','实验1按显微周期约为两倍光学周期比较，实验2、3按同周期比较。对应关系是明确的研究假设，不在本节拟合倍率。')
    body+=pages.table(['实验','配对衍射帧','偏差中位数'],[[GROUPS[q['cycle']],q['n_historical_frames'],f'{100*q["residual_median"]:+.1f}%'] for q in rows])
    two=comparison['two_dimensional_supplement'][0]
    body+=f'<p>唯一符合筛选条件的二维近电压配对中，显微平均周期为{two["p_median_um"]:.2f} µm，衍射周期为{two["d_median_um"]:.2f} µm，相差{100*two["residual_median"]:+.1f}%。</p>'
    body+='<p>本节重算31条既定匹配的汇总：先在同一历史衍射帧内合并ROI，再按帧统计。二维近电压配对只对应一个显微时刻。工具演示、模型数据和分时显微匹配各有来源，并非同一批照片连续生成。</p>'
    geom_table=[]
    for model,label in [('square','方格'),('rectangular','中心矩形'),('affine','一般仿射')]:
        ss=[q['LOO_RMS_halfstep'] for q in geo['model_comparison'] if q['model']==model];geom_table.append([label,len(ss),float(np.median(ss))])
    body+=pages.details('几何模型比较与计算记录',pages.table(['模型','帧数','条件留一误差/半步中位数'],geom_table)+'<p>该检查以既定候选和整数标签为条件，不是盲识别精度。显微静态样本重算了裁剪与分块；时间邻帧像素未随包提供，未重做时间检验。</p>'+pages.a(out,out/'geometry_model_comparison.csv','逐帧模型比较')+' · '+pages.a(out,out/'horizontal_periods.csv','一维周期')+' · '+pages.a(out,out/'table02_details.json','配对和汇总细节'))
    body+='<p><b>周期说明重复得多密，却不能单独决定各峰的亮度。</b>下一步用相位光栅模型解释光强配比。</p>'
    return body,dict(status='PASS',reader_result='分别测量衍射与显微周期，比较同周期和两倍周期对应，并保留谐波与混合反例。',horizontal_frames=len(horizontal),geometry_frames=geo['frames'],geometry_nodes=geo['nodes'],conditional_LOO_fits=geo['conditional_LOO_fits'],comparison=comparison,texture=texture)

def one_d_models(root,out):
    rows=read_csv(root/'data/legacy_original_23/one_dimensional_ratios.csv');U=np.array([float(q['voltage_RMS_V']) for q in rows]);z=(U-5.69)/(8.20-5.69)
    y=np.log([[float(q['R21_observed']),float(q['R31_observed'])] for q in rows]);smooth_U=np.linspace(5.69,8.20,240);zz=(smooth_U-5.69)/(8.20-5.69)
    models=[];folds=[];predictions=[];curves={}
    for single,label in ((True,'single_sine'),(False,'double_harmonic')):
        p,pred,predict=fit_one_d(z,y,single);fold_mse=[]
        emit('intensity.fit', model=label, parameters=p, z=z, y=y, prediction=pred, U=U)
        for held in range(len(rows)-1):
            keep=np.ones(len(rows),bool);keep[held:held+2]=False;hp,_,fn=fit_one_d(z[keep],y[keep],single);held_pred=fn(hp,z[~keep]);mse=float(np.mean((held_pred-y[~keep])**2));fold_mse.append(mse)
            folds.append(dict(model=label,held_frames=[rows[i]['frame_id'] for i in np.flatnonzero(~keep)],train_frames=[rows[i]['frame_id'] for i in np.flatnonzero(keep)],MSE_log_ratio=mse,parameters=hp.tolist()))
            emit('intensity.progress', model=label, fold=held+1)
        obs=np.exp(y);fit=np.exp(pred);scores=[float(1-np.sum((obs[:,i]-fit[:,i])**2)/np.sum((obs[:,i]-obs[:,i].mean())**2)) for i in (0,1)]
        models.append(dict(model=label,parameters=p.tolist(),train_log_RMSE=float(np.sqrt(np.mean((pred-y)**2))),adjacent_two_frame_CV_log_RMSE=float(np.sqrt(np.mean(fold_mse))),ratio_R2=scores))
        curves[label]=np.exp(predict(p,zz))
        predictions += [dict(model=label,frame_id=q['frame_id'],U_V=float(u),R21_observed=float(a[0]),R31_observed=float(a[1]),R21_fitted=float(b[0]),R31_fitted=float(b[1])) for q,u,a,b in zip(rows,U,obs,fit)]
    ref=read_json(root/'data/paper/one_d_reference.json');np.testing.assert_allclose(models[1]['parameters'],ref['parameters'],rtol=2e-6,atol=2e-8);np.testing.assert_allclose(models[1]['ratio_R2'],ref['R2'],rtol=2e-6)
    panels=[]
    for i,name in enumerate(('H2 / H1','H3 / H1')):panels.append(dict(title=name,xlabel='RMS voltage (V)',ylabel='green signal ratio',series=[plots.series('measured',U,np.exp(y[:,i]),'scatter',radius=4),plots.series('single sine',smooth_U,curves['single_sine'][:,i]),plots.series('double harmonic',smooth_U,curves['double_harmonic'][:,i])]))
    plots.plot(out/'01_one_d_models.png',panels,columns=2)
    csv_rows(out/'one_d_predictions.csv',predictions);write_json(out/'one_d_folds.json',folds)
    # The original experiment's fitted parameters are transferred unchanged.
    transfer=[];base=root/'data/runs/scan_2026-10-14_audit_v2';data=read_csv(base/'one_d_predictions.csv')
    for r in read_csv(base/'legacy_model_transfer.csv'):
        ids=json.loads(r['frame_ids']);lookup={}
        for q in data:
            if q['cycle_id']==r['cycle_id'] and q['direction']==r['direction'] and q['area']=='r125' and q['frame_id'] in ids:lookup[q['frame_id']]=q
        rr=[lookup[fid] for fid in ids];u=np.array([float(q['U_V']) for q in rr]);obs=np.array([[float(q['observed_log_R21']),float(q['observed_log_R31'])] for q in rr]);pred=one_d_prediction(np.array(models[1]['parameters']),(u-5.69)/(8.20-5.69));score=float(np.sqrt(np.mean((pred-obs)**2)))
        np.testing.assert_allclose(score,float(r['test_log_RMSE']),rtol=3e-6,atol=1e-7)
        transfer.append(dict(cycle=r['cycle_id'],direction=r['direction'],frames=len(rr),log_RMSE=score,refitted=False,frame_ids=ids))
    write_json(out/'one_d_models.json',dict(models=models,transfer=transfer,loss='natural log ratios; two ratios share parameters',CV_aggregation='sqrt(mean(fold MSE)); every adjacent pair and both ratios removed together'))
    emit('intensity.comparison', models=models, transfer=transfer, out=out)
    return models,transfer

def intensity(root,out):
    from .delivery_paper import spectrum
    models,transfer=one_d_models(root,out);two=spectrum(root,out,True)
    frames=read_json(out/'fig16_frames.json');rows=read_csv(out/'fig16_predictions.csv');panels=[]
    for cycle in ('original','cycle01','cycle02'):
        fid=next(q['fid'] for q in frames if q['cycle']==cycle);rr=[q for q in rows if q['cycle']==cycle and q['fid']==fid];observed=np.array([float(q['observed_normalized']) for q in rr]);pred=np.array([float(q['fit_normalized']) for q in rr])
        panels.append(dict(title=GROUPS[cycle].replace('实验','Experiment ')+' / '+fid,xlabel='retained bright-node index',ylabel='signal / observed frame maximum',series=[plots.series('measured',np.arange(1,len(rr)+1),observed,'scatter',radius=4),plots.series('double oblique fit',np.arange(1,len(rr)+1),pred)]))
    plots.plot(out/'02_two_d_spectra.png',panels,columns=3)
    emit('intensity.two_d', result=two, out=out)
    body='<p>空间频率决定光斑位置，傅里叶权重决定光强。现在把绿色净积分与相位光栅模型比较：同一组参数是否能解释不同级次的配比？</p><h2>先检验最简单的单正弦</h2><p>单正弦相位已经能产生多个衍射级，但它不能同时描述实验1的两条峰比曲线。于是加入第二空间谐波，仍让两条曲线共用一组参数。</p>'
    diagram=next(q['preview'] for q in read_json(root/'provenance/paper_release.json')['figures'] if q['label']=='fig:modelmeaning')
    body+=pages.figure(out,root/diagram,'论文中的模型示意：一维改变周期内的相位轮廓，二维由两条斜向频率组合节点。该示意图直接引用；下方曲线与评分由本次重拟合生成。')
    body+=pages.figure(out,out/'01_one_d_models.png','实验1的23帧实际重拟合。单正弦与双谐波使用相同观测；优化在对数峰比中进行，图中显示原峰比。')
    body+=pages.table(['模型','训练log RMSE','相邻两帧留出log RMSE'],[['单正弦' if q['model']=='single_sine' else '双谐波',q['train_log_RMSE'],q['adjacent_two_frame_CV_log_RMSE']] for q in models])
    body+='<p>双谐波改善了该区间的描述和留出预测。它仍是有效光学模型，电压参数关系并未在所有扫描中成立。</p>'
    body+=pages.details('跨轮检验与模型范围',pages.table(['实验','方向','帧数','直接转移log RMSE'],[[GROUPS[q['cycle']],'升压' if q['direction']=='up' else '降压',q['frames'],q['log_RMSE']] for q in transfer])+'<p>直接使用实验1的参数，不以待测轮数据重新拟合；实验3的二阶峰比存在失配。22折检验每次同时移走相邻两帧的两条峰比，保留全部训练/留出身份。</p>'+pages.a(out,out/'one_d_folds.json','留出计算')+' · '+pages.a(out,out/'one_d_models.json','参数、损失与转移结果')+' · '+pages.a(out,out/'one_d_predictions.csv','实测与预测'))
    body+='<h2>把二维位置与强度联系起来</h2><p>前一节的两条斜向步长给出了点阵位置。假设有两组斜向正弦相位，其复透射场的傅里叶分量给出双Bessel强度谱。位置与亮度因此成为同一光学模型的两个部分。</p>'
    body+=pages.figure(out,out/'02_two_d_spectra.png','三轮代表帧的实测与拟合。每帧只有两个相位幅度和一个整体尺度，没有为每个节点单独调节亮度。')
    body+=pages.table(['实验','帧数','亮节点','逐帧相对谱误差中位数'],[[GROUPS[q['cycle']],q['frames'],q['nodes'],f'{100*q["NRMSE_median"]:.1f}%'] for q in two['summary']])
    body+='<p>本次重拟合26帧、276个既定亮节点。模型概括主要光强配比，左右不均与局部弱信号仍有偏离；拟合通过不能唯一确定液晶取向。</p>'
    body+=pages.details('计算方法与进一步探索','<p>二维拟合保留原多起点、低相位分支、非负整体尺度和有符号损失。亮节点的选择来自原测量子集，不按本次拟合好坏重新挑点。完整历史还检查过弱节点、相位分支、孔径和局部重叠；这些限制不因本次重拟合而消失。</p>'+pages.a(out,out/'fig16_frames.json','全部逐帧参数')+' · '+pages.a(out,out/'fig16_predictions.csv','全部节点')+' · '+pages.a(out,root/'src/williams_diffraction/diffraction_models.py','完整数值算法'))
    body+='<p><b>模型把测量推进到解释，也给出了新的问题：</b>这些光学相位分量可能来自怎样的微观结构？</p>'
    return body,dict(status='PASS',reader_result='单正弦不能同时解释两条峰比；双谐波改善预测，二维双斜向谱概括主要亮节点并保留失配。',one_d=models,cross_scan=transfer,two_d=two)

def discussion(root,out):
    # Illustrative orientation-to-phase maps; never fitted to material data.
    u=np.linspace(0,2*np.pi,513);theta=np.cos(u);square=theta**2
    np.testing.assert_allclose(square,(1+np.cos(2*u))/2,atol=1e-14)
    plots.plot(out/'01_phase_doubling.png',[dict(title='A quadratic optical response can double spatial frequency',xlabel='qx (rad)',ylabel='normalized illustrative amplitude',series=[plots.series('orientation: cos(qx)',u,theta),plots.series('quadratic response',u,square)])])
    emit('discussion.doubling', out=out, error=float(np.max(abs(square-(1+np.cos(2*u))/2))))
    from .delivery_algorithms import analyze_texture
    from PIL import Image
    core=root/'data/core';items=read_json(core/'index.json')['microscopy'];periods={};vectors=[]
    for name in ('micro164_left','micro164_right'):
        item=next(q for q in items if q['id']==name)
        with Image.open(core/item['image']) as im:image=np.asarray(im.convert('RGB'))
        x,y,w,h=item['roi_xywh'];q,_,_,_,_=analyze_texture(image[y:y+h,x:x+w],*item['pixel_scale_um'])
        if name.endswith('left'):
            pair=q['reciprocal_basis_candidate'];vectors=[np.array([p['fx_cycles_um'],p['fy_cycles_um']]) for p in (pair['first'],pair['second'])]
        else:periods['stripe']=q['primary_family']['base']['period_um']
    periods['combination']=sorted([float(1/np.linalg.norm(vectors[0]+sign*vectors[1])) for sign in (-1,1)])
    ref=read_json(root/'data/paper/combination_lengths.json')[0];np.testing.assert_allclose(periods['combination'],sorted(ref['combination_periods_um']),rtol=2e-6)
    result=dict(status='PASS',reader_result='周期对应、倍频与组合频率提供结构线索；现有结果仍不能唯一反演取向场。',illustrative_identity_max_error=float(np.max(abs(square-(1+np.cos(2*u))/2))),recomputed_microscopy_lengths_um=periods,material_parameters_fitted=False)
    write_json(out/'mechanism_checks.json',result)
    emit('discussion.combination', result=result)
    body='<p>显微强度纹理与衍射透射场可以突出不同的空间频率。周期对应和光强拟合提供结构线索，但两种测量不能直接等同于分子取向。</p><h2>为什么可能出现两倍周期？</h2>'
    body+=pages.figure(out,out/'01_phase_doubling.png','示意计算：正弦取向的平方包含常数项与二倍空间频率。若两种观察突出不同分量，可能出现显微周期约为两倍光学周期。此图不是对材料取向的拟合。')
    body+='<p>非线性的取向—光程映射也可能产生第二相位谐波。不过，二次关系本身不能判定取向基频，最简单的弱倾斜修正也不足以解释全部拟合幅度。</p><h2>两方向结构能否产生组合频率？</h2>'
    body+=pages.table(['本次从同帧像素重算','周期 µm'],[['左侧两频率的和差组合',', '.join(f'{x:.2f}' for x in periods['combination'])],['右侧条纹',f'{periods["stripe"]:.2f}']])
    body+='<p>约13.83、14.33 µm的组合周期与另一条纹区的13.92 µm接近，但区域不同，方向仍有差别。长度联系不能独立证明同一个取向模态或相干结构。</p>'
    body+='<h2>研究走到了哪里</h2><p>我们从图样观察出发，建立位置与相对测光方法，再用衍射和显微两条路线测量周期，检验光强分配模型。已有结果支持若干有效光学描述，跨轮失配与微观对应尚未闭合。</p><p>下一步需要同区域、同状态的配对观察，以及更充分的响应标定和时间记录。</p>'
    body+=pages.details('两种傅里叶计算的区别','<p>显微图的强度FFT用于测纹理频率；相位模型对复透射场exp(iφ)求傅里叶分量，用其模平方计算衍射强度。两者在研究问题上相互联系，物理输入和含义不同。</p>'+pages.a(out,out/'mechanism_checks.json','本次计算范围')+' · '+pages.a(out,root/'paper/manuscript.pdf','论文讨论'))
    return body,result
