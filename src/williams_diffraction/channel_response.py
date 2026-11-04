"""Unchanged numerical functions from the accepted RG response analysis."""
from collections import Counter
import numpy as np
from .green_channel_study import regress
from .guide_events import emit
from .runtime_trace import log_values

LABELS = {'original': '原实验', 'cycle01': '新第一组', 'cycle02': '新第二组', 'pooled': '三组合并'}

MODELS = {'origin': '过原点比例', 'affine': '带截距直线', 'quadratic': '二次曲线'}

SCALE = 1e6

def design(x, model):
    if model == 'origin':
        return x[:, None]
    if model == 'affine':
        return np.column_stack((x, np.ones(len(x))))
    if model == 'quadratic':
        return np.column_stack((x, np.ones(len(x)), x*x))
    raise ValueError(model)

def solve(x, y, model, weights=None):
    a = design(x, model)
    if weights is not None:
        w = np.sqrt(weights)
        return np.linalg.lstsq(a*w[:, None], y*w, rcond=None)[0]
    return np.linalg.lstsq(a, y, rcond=None)[0]

def metrics(y, pred):
    e = y-pred; sse = float(e@e); sst = float((y-y.mean())@(y-y.mean()))
    return {'R2': 1-sse/sst, 'RMSE_DN_pixel2': float(np.sqrt(np.mean(e*e)))*SCALE,
            'MAE_DN_pixel2': float(np.mean(abs(e)))*SCALE,
            'RMSE_pct_range': float(np.sqrt(np.mean(e*e))/(y.max()-y.min())*100),
            'SSE_million_units': sse}

def arrays(rows, area, green_pair=False):
    x = np.array([r[f'Q_G1_{area}' if green_pair else f'Q_R_{area}'] for r in rows])/SCALE
    y = np.array([r[f'Q_G2_{area}' if green_pair else f'Q_G_{area}'] for r in rows])/SCALE
    return x, y

def fit_rows(rows, area, model, group, subset='all_unsaturated', green_pair=False):
    x, y = arrays(rows, area, green_pair)
    p = solve(x, y, model); pred = design(x, model)@p
    log_values('RG_fit',group=group,area=area,model=model,subset=subset,x=x,y=y,parameters=p,predicted=pred,residual=y-pred)
    keys = np.array([r['frame_key'] for r in rows]); unique = np.unique(keys)
    cv = np.empty(len(rows))
    for key in unique:
        test = keys == key
        fold_parameters=solve(x[~test],y[~test],model)
        cv[test] = design(x[test], model)@fold_parameters
        log_values('RG_leave_frame_out',frame=key,training_count=int((~test).sum()),test_count=int(test.sum()),parameters=fold_parameters,observed=y[test],predicted=cv[test],residual=y[test]-cv[test])
    result = {'group': group, 'area': area, 'subset': subset, 'model': model,
              'response': 'G2_from_G1' if green_pair else 'G_from_R', 'n': len(rows), 'n_frames': len(unique),
              'slope': float(p[0]), 'intercept_DN_pixel2': float(p[1])*SCALE if model != 'origin' else 0.,
              'quadratic_per_million': float(p[2]) if model == 'quadratic' else 0.,
              'R_min': float(x.min())*SCALE, 'R_max': float(x.max())*SCALE,
              'G_min': float(y.min())*SCALE, 'G_max': float(y.max())*SCALE,
              **metrics(y, pred), **{'LOFO_'+k: v for k, v in metrics(y, cv).items()}}
    log_values('RG_scores',**result)
    return result, pred, cv

def bootstrap(rows, area, n, rng):
    """Resample whole frames within each scan; keep all peaks in each draw."""
    blocks = {}
    x, y = arrays(rows, area)
    groups = sorted({r['group'] for r in rows})
    for g in groups:
        frames = sorted({r['frame_key'] for r in rows if r['group'] == g})
        blocks[g] = [np.array([i for i, r in enumerate(rows) if r['frame_key'] == f]) for f in frames]
    ps = []
    for iteration in range(n):
        idx = np.concatenate([blocks[g][i] for g in groups for i in rng.integers(0, len(blocks[g]), len(blocks[g]))])
        ps.append(solve(x[idx], y[idx], 'affine'))
        if iteration<3 or (iteration+1)%100==0:
            log_values('RG_bootstrap_sample',iteration=iteration+1,total=n,sample_indices=idx,parameters=ps[-1])
    ps = np.array(ps)
    return {'slope_ci_low': float(np.quantile(ps[:, 0], .025)), 'slope_ci_high': float(np.quantile(ps[:, 0], .975)),
            'intercept_ci_low_DN_pixel2': float(np.quantile(ps[:, 1], .025))*SCALE,
            'intercept_ci_high_DN_pixel2': float(np.quantile(ps[:, 1], .975))*SCALE}

def analyze(rows, cfg):
    keep = [r for r in rows if r['selection'] == 'included']
    scores, predictions, summaries, bins, quartiles = [], [], [], [], []
    rng = np.random.default_rng(cfg['seed'])
    for g in LABELS:
        allg = [r for r in rows if g == 'pooled' or r['group'] == g]
        selected = [r for r in keep if g == 'pooled' or r['group'] == g]
        counts = Counter(r['selection'] for r in allg)
        summary = {'group': g, 'n_candidates': len(allg), **{k: counts[k] for k in
                    ('central', 'confirmed_nonoptical', 'channel_failed', 'hard_saturated', 'nonfinite', 'reflection_aperture_overlap', 'included')},
                   'n_frames_with_included': len({r['frame_key'] for r in selected}),
                   'near_white_included': sum(r['near_white'] for r in selected),
                   'reflection_suspect_included': sum(r['reflection_suspect'] for r in selected),
                   'legacy_reflection_overlap_not_assessed':sum(r.get('reflection_overlap_audit_status')=='LEGACY_FROZEN_NOT_ASSESSED' for r in selected),
                   'green_max_raw_DN_included': max(max(r['G1_max_raw_DN'], r['G2_max_raw_DN']) for r in selected)}
        red_clipped = [r for r in allg if r['selection'] == 'hard_saturated' and r['R_saturated_sample_count'] > 0
                       and r['G1_saturated_sample_count'] == 0 and r['G2_saturated_sample_count'] == 0 and not r.get('reflection_overlap_any',False)]
        summary['red_clipped_green_clean_count'] = len(red_clipped)
        summary['green_max_raw_DN_red_clipped'] = max((max(r['G1_max_raw_DN'], r['G2_max_raw_DN']) for r in red_clipped), default=None)
        for area in cfg['apertures']:
            x, y = arrays(selected, area)
            summary[f'nonpositive_R_or_G_{area}'] = int(np.sum((x <= 0) | (y <= 0)))
            summary[f'min_coverage_{area}'] = min(r[f'coverage_min_{area}'] for r in selected)
            sets = {'all_unsaturated': selected,
                    'no_near_white': [r for r in selected if not r['near_white']],
                    'no_reflection_suspect': [r for r in selected if not r['reflection_suspect']],
                    'coverage_ge_0.95': [r for r in selected if r[f'coverage_min_{area}'] >= .95],
                    'no_weak_shape_flag': [r for r in selected if 'WEAK_SHAPE_EVIDENCE' not in r['quality_flags']]}
            if cfg.get('per_aperture_reflection_clean_sensitivity',False):
                sets['per_aperture_reflection_clean']=[r for r in allg if r['selection'] in ('included','reflection_aperture_overlap') and not r['reflection_overlap_'+area]]
            if any(r['candidate_status']!='LEGACY_CANDIDATE' for r in selected):
                sets['no_detection_review_flag']=[r for r in selected if r['candidate_status']!='REVIEW_REQUIRED']
            for subset, rs in sets.items():
                for model in (MODELS if subset == 'all_unsaturated' else ('affine',)):
                    score, pred, cv = fit_rows(rs, area, model, g, subset)
                    if subset == 'all_unsaturated' and model == 'affine':
                        score.update(bootstrap(rs, area, cfg['bootstrap_replicates'], rng))
                        # Independent check with the project's original response regression.
                        old = regress(rs, f'Q_R_{area}', f'Q_G_{area}')
                        if not np.allclose([score['slope'], score['intercept_DN_pixel2'], score['R2']],
                                           [old['slope'], old['intercept_DN'], old['R_squared']], rtol=1e-10, atol=1e-6):
                            raise ValueError('Regression differs from existing helper')
                        score['soft_l1_slope'] = old['robust_slope']
                        score['soft_l1_intercept_DN_pixel2'] = old['robust_intercept_DN']
                        for r, p, v in zip(rs, pred, cv):
                            predictions.append({'fit_group': g, 'area': area, 'peak_key': r['peak_key'],
                                                'group': r['group'], 'frame_key': r['frame_key'],
                                                'Q_R': r[f'Q_R_{area}'], 'Q_G': r[f'Q_G_{area}'],
                                                'pred_G': float(p)*SCALE, 'residual_G': r[f'Q_G_{area}']-float(p)*SCALE,
                                                'LOFO_pred_G': float(v)*SCALE})
                        residual = y-pred
                        # Quartiles use every signed R observation. Ratios alone need R>0;
                        # this is descriptive grouping, never a regression selection.
                        for q, idx in enumerate(np.array_split(np.argsort(x), 4)):
                            positive = idx[x[idx] > 0]
                            ratio = y[positive]/x[positive]
                            quartiles.append({'group': g, 'area': area, 'quartile': q+1,
                                              'n_all': len(idx), 'n_ratio': len(positive),
                                              'n_nonpositive_R': int(np.sum(x[idx] <= 0)),
                                              'R_min': float(x[idx].min())*SCALE, 'R_max': float(x[idx].max())*SCALE,
                                              'ratio_q25': float(np.quantile(ratio, .25)),
                                              'ratio_median': float(np.median(ratio)),
                                              'ratio_q75': float(np.quantile(ratio, .75)),
                                              'median_abs_affine_residual_DN_pixel2': float(np.median(abs(residual[idx])))*SCALE,
                                              'median_abs_origin_relative_residual_pct': float(np.median(abs(y[positive]-old['zero_intercept_slope']*x[positive])/(old['zero_intercept_slope']*x[positive]))*100)})
                        for i, idx in enumerate(np.array_split(np.argsort(x), 8)):
                            bins.append({'group': g, 'area': area, 'bin': i+1, 'n': len(idx),
                                         'R_min': float(x[idx].min())*SCALE, 'R_max': float(x[idx].max())*SCALE,
                                         'R_median': float(np.median(x[idx]))*SCALE,
                                         'G_median': float(np.median(y[idx]))*SCALE,
                                         'residual_mean': float(np.mean(residual[idx]))*SCALE,
                                         'residual_median': float(np.median(residual[idx]))*SCALE,
                                         'residual_q10': float(np.quantile(residual[idx], .1))*SCALE,
                                         'residual_q90': float(np.quantile(residual[idx], .9))*SCALE,
                                         'MAE': float(np.mean(abs(residual[idx])))*SCALE})
                    scores.append(score)
            gg, _, _ = fit_rows(selected, area, 'affine', g, green_pair=True)
            scores.append(gg)
        summaries.append(summary)
        emit('rg.progress', group=g, completed_fits=len(scores))
    # A common line is compared with three separate lines on exactly the same points.
    for area in cfg['apertures']:
        y = np.array([r[f'Q_G_{area}'] for r in keep])/SCALE
        pred, cv = [], []
        lookup = {(r['area'], r['fit_group'], r['peak_key']): r for r in predictions}
        for r in keep:
            p = lookup[area, r['group'], r['peak_key']]
            pred.append(p['pred_G']/SCALE); cv.append(p['LOFO_pred_G']/SCALE)
        scores.append({'group': 'pooled', 'area': area, 'subset': 'all_unsaturated', 'model': 'group_affine',
                       'response': 'G_from_R', 'n': len(keep), 'n_frames': len({r['frame_key'] for r in keep}),
                       **metrics(y, np.array(pred)), **{'LOFO_'+k: v for k, v in metrics(y, np.array(cv)).items()}})
    return summaries, scores, predictions, bins, quartiles
