"""Recompute declared frozen observations with the original numerical functions.

This validates a table-based reproduction, not RAW measurement or model truth.
Geometry labels (m,n) and optical-spectrum labels (a,b) remain separate.
"""
from collections import defaultdict
import json
import time
import numpy as np
import scipy
from scipy.special import jv
from .diffraction_models import fit_geometry, fit_one_d, fit_spectrum, metric
from .voltage_spectrum import SIX, fit_voltage_spectrum, fit_signal_law, predict_voltage_spectrum
from .delivery_io import read_json, read_csv, write_json, new_output, verify_bundle, parity_svg


def grouped(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def check(actual, expected, label, *, rtol=2e-6, atol=2e-8):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)


def one_predict(p, U, window, single):
    from .diffraction_models import one_d_prediction
    z = (np.asarray(U)-window[0])/(window[1]-window[0])
    if not single:
        return one_d_prediction(np.asarray(p), z)
    w = jv(np.arange(1,4)[None,:], (p[0]*(1-z)+p[1]*z)[:,None])**2
    return np.log(np.maximum(w[:,1:],1e-24)/np.maximum(w[:,:1],1e-24))


def reproduce(root, output, full=False):
    manifest = verify_bundle(root, quiet=True)
    out = new_output(root, output)
    start = time.perf_counter()
    data = root / 'data'
    scan = data / 'runs/scan_2026-10-14_audit_v2'
    article = data / 'runs/three_model_audit_2026-10-18_v3'
    final = data / 'runs/final_paper_2026-10-27_v2'
    checks = []
    # Same retained frame identities and exact original loss, no fit-based reselection.
    one_rows = read_csv(scan/'one_d_predictions.csv')
    one_groups = grouped(one_rows, ('cycle_id','direction','area','model'))
    fitted, one_outputs, parity = {}, [], []
    for r in read_csv(scan/'one_d_models.csv'):
        key = tuple(r[k] for k in ('cycle_id','direction','area','model'))
        rows = one_groups[key]
        if [v['frame_id'] for v in rows] != json.loads(r['frame_ids']):
            raise ValueError('One-dimensional frame identity/order mismatch')
        U = np.array([float(v['U_V']) for v in rows])
        y = np.array([[float(v['observed_log_R21']),float(v['observed_log_R31'])] for v in rows])
        window = json.loads(r['voltage_window']); z = (U-window[0])/(window[1]-window[0])
        p, pred, _ = fit_one_d(z,y,r['model']=='single_sine')
        reference = np.array([[float(v['fit_log_R21']),float(v['fit_log_R31'])] for v in rows])
        check(pred, reference, '1D fitted log ratios')
        score = float(np.sqrt(np.mean((pred-y)**2)))
        check(score,float(r['train_log_RMSE']),'1D training score')
        fitted[key] = (p,window)
        one_outputs.append({'key':list(key),'parameters':p.tolist(),'n_frames':len(rows),'train_log_RMSE':score})
        parity.extend(zip(y.ravel(),pred.ravel()))
    checks.append({'check':'one_dimensional_refits','groups':len(one_outputs),'status':'PASS'})
    legacy=read_csv(data/'legacy_original_23/one_dimensional_ratios.csv')
    legacy_p=json.loads(read_csv(scan/'legacy_model_transfer.csv')[0]['parameters'])
    legacy_pred=one_predict(legacy_p,[float(r['voltage_RMS_V']) for r in legacy],[5.69,8.20],False)
    check(np.exp(legacy_pred),[[float(r['R21_fitted']),float(r['R31_fitted'])] for r in legacy],'original 23-frame frozen-parameter replay')
    checks.append({'check':'original_23_frame_parameter_replay','frames':len(legacy),'status':'PASS','refitted':False})
    cross=[]
    for r in read_csv(scan/'cross_cycle_predictions.csv'):
        key=(r['train_cycle'],r['direction'],r['area'],r['model'])
        target=one_groups[(r['test_cycle'],r['direction'],r['area'],r['model'])]
        assert [a['frame_id'] for a in target]==json.loads(r['test_frame_ids'])
        U=[float(a['U_V']) for a in target]
        y=np.array([[float(a['observed_log_R21']),float(a['observed_log_R31'])] for a in target])
        p,w=fitted[key];pred=one_predict(p,U,w,r['model']=='single_sine')
        score=float(np.sqrt(np.mean((pred-y)**2)))
        check(score,float(r['test_log_RMSE']),'cross-cycle score')
        cross.append({'train':r['train_cycle'],'test':r['test_cycle'],'direction':r['direction'],'area':r['area'],'model':r['model'],'log_RMSE':score})
    checks.append({'check':'cross_cycle_predictions','groups':len(cross),'status':'PASS'})
    write_json(out/'one_dimensional.json',{'fits':one_outputs,'cross_cycle':cross,'selection':'unchanged frozen frame lists'})
    parity_svg(out/'one_dimensional.svg',parity,'1D observed vs refitted log ratios', 'Observed log ratio','Refitted log ratio')
    print('PASS: one-dimensional fits and cross-cycle predictions',flush=True)

    # Final 300 mm geometry for new cycles, original 120 mm for the first batch.
    nodes=read_json(final/'geometry_nodes.json'); ggroups=grouped(nodes,('cycle','frame_id'))
    geo=[];gparity=[]
    for ref in read_json(final/'geometry_models.json'):
        rr=ggroups[(ref['cycle'],ref['frame_id'])]
        labels=np.array([a['labels'] for a in rr]);xy=np.array([a['observed_s'] for a in rr])
        c,p=fit_geometry(labels,xy,'rectangular')
        check(c,ref['coefficients'],'geometry coefficients',atol=1e-9)
        check(p,[a['fitted_s'] for a in rr],'geometry predictions',atol=1e-9)
        h,v=c[1:];period=.6328/np.linalg.norm(h+v)
        check(period,ref['period_um'],'geometry optical period')
        scale=min(np.linalg.norm(h),np.linalg.norm(v));loo=[]
        for k in range(len(rr)):
            keep=np.arange(len(rr))!=k;held,_=fit_geometry(labels[keep],xy[keep],'rectangular')
            loo.append(np.r_[1,labels[k]]@held)
        loo_score=float(np.sqrt(np.mean(np.sum((xy-np.array(loo))**2,axis=1)))/scale)
        check(loo_score,ref['LOO_RMS_halfstep'],'conditional geometry LOO',atol=2e-7)
        geo.append({'cycle':ref['cycle'],'frame_id':ref['frame_id'],'period_um':float(period),'coefficients':c.tolist(),'LOO_RMS_halfstep':loo_score,'n_nodes':len(rr)})
        gparity.extend(zip(xy.ravel(),p.ravel()))
    horizontal=[]
    for ref in read_json(final/'horizontal_periods.json'):
        radius=np.asarray(ref['radius_mm']);sine=radius/np.sqrt(ref['distance_mm']**2+radius**2)
        slope=np.linalg.lstsq(np.arange(1,4)[:,None],sine,rcond=None)[0][0]
        period=float(.6328/slope)
        check(period,ref['three_period_um'],'three-peak period')
        horizontal.append({'frame_id':ref['frame_id'],'cycle':ref['cycle'],'three_period_um':period})
    checks.append({'check':'final_geometry_and_periods','geometry_frames':len(geo),'conditional_LOO_fits':len(nodes),'horizontal_frames':len(horizontal),'status':'PASS'})
    write_json(out/'geometry.json',{'geometry':geo,'horizontal':horizontal,'wavelength_um':.6328,'new_distance_mm':300.,'original_distance_mm':120.})
    matches=read_json(final/'microscopy_matches.json')
    for r in matches:
        check(r['image_period_um']/r['current_expected_image_period_um']-1,r['current_residual'],'microscopy correspondence residual')
    checks.append({'check':'final_microscopy_correspondence_residuals','conditional_matches':len(matches),'status':'PASS'})
    parity_svg(out/'geometry.svg',gparity,'Observed vs fitted direction cosines')
    print('PASS: final geometry, conditional leave-one-node-out and periods',flush=True)

    # Include both models on identical observation sets; keep signed observations.
    sp_rows=read_csv(scan/'spectrum_predictions.csv')
    keys=('frame_id','subset','area','model')
    sp_groups=grouped(sp_rows,keys)
    selected={key for key in sp_groups if full or (key[1] in ('bright','all') and key[2]=='r50')}
    selected.update(key for key,rr in sp_groups.items() if any(float(a['observed'])<0 for a in rr))
    reference_models={tuple(r[k] for k in keys):r for r in read_csv(scan/'spectrum_models.csv')}
    spectra=[];sparity=[];negative=0
    for key in sorted(selected):
        rows=sp_groups[key]; labels=np.array([[int(a['a']),int(a['b'])] for a in rows]);q=np.array([float(a['observed']) for a in rows])
        p,amp,pred,attempts=fit_spectrum(labels,q,key[-1])
        reference=np.array([float(a['fitted']) for a in rows])
        check(pred,reference,'spectrum predictions',atol=max(float(np.linalg.norm(q)),1)*2e-7)
        score=metric(q,pred);ref=reference_models[key]
        check(score['NRMSE_L2'],float(ref['NRMSE_L2']),'spectrum score')
        n_negative=int((q<0).sum());negative+=n_negative
        spectra.append({'key':list(key),'parameters':p.tolist(),'amplitude':amp,'n_nodes':len(q),'n_negative':n_negative,'metrics':score,'attempts':attempts})
        if key[1]=='bright' and key[2]=='r50':sparity.extend(zip(q,pred))
    checks.append({'check':'spectrum_refits','groups':len(spectra),'negative_observations_retained':negative,'full_aperture_sensitivity':full,'status':'PASS'})
    write_json(out/'spectra.json',spectra)
    parity_svg(out/'spectra.svg',sparity,'Bright-node camera signal vs refitted spectrum')
    print(f'PASS: {len(spectra)} spectrum cases, signed observations retained',flush=True)

    # Refit phase and both scale laws. Every held frame is removed, including its Q6.
    voltage=[]
    references=read_csv(article/'shared_voltage_node_predictions.csv')
    envelope_refs=read_csv(article/'positive_envelope_nodes.csv')
    for ref in read_csv(article/'shared_voltage_models.csv'):
        series=ref['series'];window=json.loads(ref['voltage_window']);ids=json.loads(ref['train_frames'])
        rr=[r for r in references if r['series']==series and r['evaluation']=='shared_fit']
        groups=[]
        for fid in ids:
            part=[r for r in rr if r['frame_id']==fid];lab=np.array([[int(a['a']),int(a['b'])] for a in part]);q=np.array([float(a['observed']) for a in part])
            mask=np.array([tuple(a) in SIX for a in lab]);assert mask.sum()==6
            groups.append({'fid':fid,'U':float(part[0]['U_V']),'labels':lab,'q':q,'Q6':float(q[mask].sum()),'rows':part})
        parameters,linear,attempts=fit_voltage_spectrum(groups,window)
        check(parameters,json.loads(ref['parameters']),'shared phase endpoints')
        check(linear,json.loads(ref['Q6_law_at_U6']),'linear Q6 law',atol=1e-5)
        for held in groups:
            train=[g for g in groups if g is not held]
            hp,hl,ha=fit_voltage_spectrum(train,window)
            for evaluation,par,training,law in [('shared_fit',parameters,groups,linear),('frame_holdout',hp,train,hl)]:
                for family,refs in [('linear',references),('log_quadratic',envelope_refs)]:
                    use_law=law if family=='linear' else fit_signal_law(training,'log_quadratic')
                    pred=predict_voltage_spectrum(held['labels'],held['U'],par,use_law,window,family=family)
                    reference={a['peak_uid']:float(a['fitted']) for a in refs if a['series']==series and a['frame_id']==held['fid'] and a['evaluation']==evaluation}
                    check(pred,[reference[a['peak_uid']] for a in held['rows']],'voltage-only node prediction',atol=max(float(np.linalg.norm(held['q'])),1)*2e-7)
                    voltage.append({'series':series,'frame_id':held['fid'],'evaluation':evaluation,'family':family,'parameters':par.tolist(),'law':use_law.tolist(),'train_frames':[g['fid'] for g in training],'held_uids':[a['peak_uid'] for a in held['rows']],'no_held_signal_anchor':evaluation=='frame_holdout','fitted':pred.tolist(),'observed':held['q'].tolist()})
    checks.append({'check':'voltage_only_refit_and_frame_holdout','evaluations':len(voltage),'status':'PASS'})
    write_json(out/'voltage_prediction.json',voltage)
    result={'status':'PASS','scope':'R1 frozen-table refits; not RAW remeasurement, not blind detection','checks':checks,'elapsed_s':time.perf_counter()-start,'immutable_inputs':len(manifest['files']),'runtime':{'numpy':np.__version__,'scipy':scipy.__version__},'scientific_limits':'Conditional retained labels/frames; camera signal, not radiometric power. Failures and negative scores in original tables remain.'}
    write_json(out/'reproduction.json',result)
    (out/'README.md').write_text('# 模型复现结果\n\n本次重新拟合与参考表逐项核对通过，实际检查范围见 reproduction.json。\n\n[一维](one_dimensional.svg) · [位置](geometry.svg) · [谱形](spectra.svg)\n\n这些图由本次计算生成；模型本身的失配不因数值复现通过而消失。\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
