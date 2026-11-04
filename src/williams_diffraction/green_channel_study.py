"""Stage-one CFA response study; no RAW edits and no production photometry.

Reuses eight trial frames plus the unsaturated 070 main-peak control.
All intensities are computed before display transforms, without demosaicing.
"""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .paths import LEGACY_STAGE1 as BASE
from .paths import RAW_DIR as RAW
TRIAL = BASE / 'outputs/algorithm_trial_v2_r4'
OLD = BASE / 'outputs/stage1_20261010_corrected'
FRAME_IDS = ['001', '012', '014', '018', '026', '046', '064', '070', '077']
CHANNELS = {'R': (0, 0, 0), 'G1': (0, 1, 1), 'G2': (1, 0, 3), 'B': (1, 1, 2)}


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def save_json(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def robust_sigma(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def background(z, xx, yy, mask):
    """Robust local plane in RAW coordinates; black already removed once."""
    if np.sum(mask) < 150:
        raise ValueError('Insufficient independent background support')
    design = np.column_stack([np.ones(np.sum(mask)), xx[mask] / 150, yy[mask] / 150])
    target = z[mask]
    keep = np.ones(len(target), bool)
    for _ in range(5):
        coeff = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        resid = target - design @ coeff
        scale = max(robust_sigma(resid), 1 / np.sqrt(12))
        keep = np.abs(resid - np.median(resid)) < 3 * scale
    plane = coeff[0] + coeff[1] * xx / 150 + coeff[2] * yy / 150
    # Eight angular sectors expose spatial/background variation; this is an
    # empirical uncertainty indicator, not a calibrated detector-noise model.
    theta = np.arctan2(yy[mask], xx[mask])
    sector = np.minimum(7, ((theta + np.pi) / (2 * np.pi) * 8).astype(int))
    levels = [np.mean(resid[(sector == i) & keep]) for i in range(8)
              if np.sum((sector == i) & keep) >= 20]
    bg_se = float(np.std(levels, ddof=1) / np.sqrt(len(levels))) if len(levels) > 1 else scale
    return plane, coeff, scale, bg_se, float(np.median(target)), int(np.sum(mask))


def aperture_weights(xx, yy, radius):
    """Fractional 2x2 RAW-cell coverage; values themselves are not interpolated."""
    w = np.zeros_like(xx, dtype=float)
    for sy in [-.75, -.25, .25, .75]:
        for sx in [-.75, -.25, .25, .75]:
            w += ((xx + sx) ** 2 + (yy + sy) ** 2 <= radius ** 2)
    return w / 16


def regress(rows, xkey, ykey):
    if len(rows) < 3:
        return {'n': len(rows)}
    x = np.array([r[xkey] for r in rows]); y = np.array([r[ykey] for r in rows])
    a, b = np.polyfit(x, y, 1)
    a0 = float(x @ y / (x @ x))
    residual = y - (a * x + b)
    rel = residual / np.maximum(np.abs(a * x + b), 1e-12)
    scale = max(robust_sigma(residual), .1)
    robust = least_squares(lambda p: y - (p[0] * x + p[1]), [a, b], loss='soft_l1', f_scale=scale)
    return {'n': len(rows), 'x_range_DN':[float(np.min(x)),float(np.max(x))],
            'y_range_DN':[float(np.min(y)),float(np.max(y))],
            'slope': float(a), 'intercept_DN': float(b), 'zero_intercept_slope': a0,
            'R_squared': float(1 - (residual @ residual) / np.sum((y - np.mean(y)) ** 2)),
            'median_abs_relative_residual_pct': float(np.median(np.abs(rel)) * 100),
            'p95_abs_relative_residual_pct': float(np.percentile(np.abs(rel), 95) * 100),
            'ratio_p05_p50_p95': np.percentile(y / x, [5, 50, 95]).tolist(),
            'robust_slope': float(robust.x[0]), 'robust_intercept_DN': float(robust.x[1])}


def run(out):
    import rawpy
    import exifread
    from .paths import require_writable_run
    out=require_writable_run(out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(out / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 160, 'axes.spines.top': False, 'axes.spines.right': False})

    trial_spots = read_csv(TRIAL / 'spots.csv')
    old_spots = read_csv(OLD / 'spots.csv')
    control = dict(next(r for r in old_spots if r['frame_id'] == '070' and r['spot_id'] == 'spot_004'))
    control['detection_origin'] = 'unsaturated_control_only'
    spots = trial_spots + [control]
    rejected = read_csv(TRIAL / 'rejected_candidates.csv')
    metadata = []; aperture_rows = []; blocks = []; profile_rows = []
    channels_summary = read_csv(BASE / 'outputs/rawcheck_20261010/raw_channel_diagnostics.csv')
    global_summary = {}
    for name, (_, _, ci) in CHANNELS.items():
        ss = [r for r in channels_summary if int(r['channel_index']) == ci]
        global_summary[name] = {'frames': len(ss), 'min_frame_max_DN': min(int(r['max']) for r in ss),
            'max_DN': max(int(r['max']) for r in ss),
            'frames_at_white': sum(int(r['n_at_white']) > 0 for r in ss),
            'frames_at_camera_white': sum(int(r['n_at_camera_white']) > 0 for r in ss)}

    montage = []
    for fid in FRAME_IDS:
        path = RAW / f'{fid}.ARW'; hash_before = digest(path)
        with path.open('rb') as f:
            exif = exifread.process_file(f, details=True)
        with rawpy.imread(str(path)) as raw:
            assert raw.raw_pattern.tolist() == [[0, 1], [3, 2]]
            assert raw.color_desc == b'RGBG'
            mosaic = raw.raw_image_visible.copy()
            black = raw.black_level_per_channel
            white = raw.camera_white_level_per_channel
            assert white is not None
        assert mosaic.shape == (4688, 7028)
        planes = {name: mosaic[dy::2, dx::2].astype(float) - black[ci]
                  for name, (dy, dx, ci) in CHANNELS.items()}
        red_limit = .8 * (white[0] - black[0])
        frame_spots = [s for s in spots if s['frame_id'] == fid]
        exclusion_spots = [s for s in (trial_spots if fid != '070' else old_spots) if s['frame_id'] == fid]
        centres = np.array([[float(s['u_raw_px']), float(s['v_raw_px'])] for s in exclusion_spots])
        reflections = np.array([[float(s['u_raw_px']), float(s['v_raw_px'])] for s in rejected if s['frame_id'] == fid]).reshape(-1, 2)
        seen = set()
        for s in frame_spots:
            cx, cy = float(s['u_raw_px']), float(s['v_raw_px'])
            patches = {}; stats = {}
            for name, (dy, dx, ci) in CHANNELS.items():
                j0 = max(0, int(np.floor((cx - 175 - dx) / 2))); j1 = int(np.ceil((cx + 175 - dx) / 2))
                i0 = max(0, int(np.floor((cy - 175 - dy) / 2))); i1 = int(np.ceil((cy + 175 - dy) / 2))
                xx, yy = np.meshgrid(np.arange(j0, j1) * 2 + dx - cx, np.arange(i0, i1) * 2 + dy - cy)
                z = planes[name][i0:i1, j0:j1]
                rr = np.hypot(xx, yy)
                valid = np.ones_like(rr, bool)
                for nx, ny in centres:
                    if np.hypot(nx - cx, ny - cy) > 1:
                        valid &= (xx + cx - nx) ** 2 + (yy + cy - ny) ** 2 > 70 ** 2
                for nx, ny in reflections:
                    valid &= (xx + cx - nx) ** 2 + (yy + cy - ny) ** 2 > 90 ** 2
                bg, coeff, sigma, bgse, med, nbg = background(z, xx, yy, valid & (rr >= 85) & (rr <= 120))
                alt_bg, _, _, _, _, _ = background(z, xx, yy, valid & (rr >= 110) & (rr <= 150))
                patches[name] = (z, xx, yy, bg, coeff)
                st = {'bg_DN': float(coeff[0]), 'bg_sigma_DN': sigma, 'bg_se_indicator_DN': bgse, 'bg_n': nbg,
                      'max_raw_within81_DN':float(np.max((z+black[ci])[rr<=81]))}
                for rad in (35, 50, 65):
                    w = aperture_weights(xx, yy, rad); n = np.sum(w)
                    st[f'mean_r{rad}_DN'] = float(np.sum(w * (z - bg)) / n)
                    st[f'weighted_n_r{rad}'] = float(n)
                    if rad == 50:
                        st['mean_alt_background_DN'] = float(np.sum(w * (z - alt_bg)) / n)
                        st['mean_median_background_DN'] = float(np.sum(w * (z - med)) / n)
                        st['sum_r50_DN'] = float(np.sum(w * (z - bg)))
                        st['random_se_indicator_DN'] = float(sigma * np.sqrt(np.sum(w ** 2)) / n)
                        st['max_raw_DN'] = float(np.max((z + black[ci])[w > 0]))
                        st['n_at_camera_white'] = int(np.sum((z + black[ci] >= white[ci]) & (w > 0)))
                stats[name] = st
            row = {'frame_id': fid, 'spot_id': s['spot_id'], 'detection_origin': s['detection_origin'],
                   'quality_flags': s['quality_flags'], 'u_raw_px': cx, 'v_raw_px': cy,
                   'photometry_valid': False, 'purpose': 'fixed_aperture_channel_diagnostic_only'}
            for name in CHANNELS:
                row.update({name + '_' + k: v for k, v in stats[name].items()})
            for rad in (35, 50, 65):
                g = (stats['G1'][f'mean_r{rad}_DN'] + stats['G2'][f'mean_r{rad}_DN']) / 2
                row[f'G_mean_r{rad}_DN'] = g
                r = stats['R'][f'mean_r{rad}_DN']
                row[f'G_over_R_r{rad}'] = g / r if r != 0 else None
            g = row['G_mean_r50_DN']
            g_alt = (stats['G1']['mean_alt_background_DN'] + stats['G2']['mean_alt_background_DN']) / 2
            g_med = (stats['G1']['mean_median_background_DN'] + stats['G2']['mean_median_background_DN']) / 2
            bg_variation = max(abs(g_alt - g), abs(g_med - g))
            # Correlated background errors need not cancel between green planes.
            bgse = (stats['G1']['bg_se_indicator_DN'] + stats['G2']['bg_se_indicator_DN']) / 2
            rndse = np.hypot(stats['G1']['random_se_indicator_DN'], stats['G2']['random_se_indicator_DN']) / 2
            empirical_error = max(float(np.hypot(bgse, rndse)), bg_variation)
            row.update({'G_background_variation_DN': bg_variation, 'G_error_indicator_DN': empirical_error,
                        'G_background_separation_score': g / max(empirical_error, 1e-12),
                        'G_background_variation_fraction': bg_variation / max(abs(g), 1e-12),
                        'G2_G1_fraction_difference': (stats['G2']['mean_r50_DN'] - stats['G1']['mean_r50_DN']) / max(abs(g), 1e-12),
                        'Q_R_density_corrected_DN_rawpixel2': 4 * stats['R']['sum_r50_DN'],
                        'Q_G_density_corrected_DN_rawpixel2': 2 * (stats['G1']['sum_r50_DN'] + stats['G2']['sum_r50_DN'])})
            row['G_diagnostic_stable'] = bool(g >= 5 * empirical_error)
            # Common selection for all three apertures, including 16 RAW px
            # beyond the largest radius. Do not let a larger aperture include
            # an untested red core simply because the radius-50 aperture is safe.
            row['R_below_conservative_cap'] = stats['R']['max_raw_within81_DN'] - black[0] < red_limit
            aperture_rows.append(row)

            # Common complete RAW blocks (16 or 32 pixels per side). Each CFA
            # phase has exactly equal sample count in each block.
            for bs in (16, 32):
                for y in range(int((cy - 65) // bs) * bs, int(cy + 65), bs):
                    for x in range(int((cx - 65) // bs) * bs, int(cx + 65), bs):
                        if np.hypot(x + (bs-1)/2 - cx, y + (bs-1)/2 - cy) > 65:
                            continue
                        key = (bs, x, y)
                        if key in seen:
                            continue
                        seen.add(key)
                        # A block in the buffer of a red over-range sample is
                        # excluded from R-G regression. Buffer is sensitivity
                        # protection, NOT a measured Sony contamination radius.
                        yr, xr = y // 2, x // 2; b = bs // 2
                        red_neighbour = planes['R'][yr-8:yr+b+8, xr-8:xr+b+8]
                        rec = {'frame_id': fid, 'spot_id': s['spot_id'], 'block_raw_px': bs, 'x_raw': x, 'y_raw': y,
                               'R_safe_neighbour': bool(np.max(red_neighbour) < red_limit),
                               'R_clipped_neighbour': bool(np.max(red_neighbour) + black[0] >= white[0]),
                               'overlap_risk': 'OVERLAP' in s['quality_flags']}
                        for name, (dy, dx, ci) in CHANNELS.items():
                            a = planes[name][yr:yr+b, xr:xr+b]
                            coeff = patches[name][4]
                            bx, by = x + dx + b - 1 - cx, y + dy + b - 1 - cy
                            rec[name] = float(np.mean(a) - (coeff[0] + coeff[1] * bx / 150 + coeff[2] * by / 150))
                        rec['G'] = (rec['G1'] + rec['G2']) / 2
                        blocks.append(rec)

        # Brightest existing candidate is only a display anchor, not established zero order.
        anchor = max([r for r in aperture_rows if r['frame_id'] == fid], key=lambda r:r['G_mean_r50_DN'])
        cx, cy = anchor['u_raw_px'], anchor['v_raw_px']
        if fid in ('014', '046', '070'):
            for step in range(-152, 153, 8):
                rec = {'frame_id': fid, 'vertical_offset_raw_px': step + 3.5}
                for name, (dy, dx, ci) in CHANNELS.items():
                    xs = np.arange(planes[name].shape[1]) * 2 + dx
                    ys = np.arange(planes[name].shape[0]) * 2 + dy
                    v = planes[name][np.ix_((ys >= cy + step) & (ys < cy + step + 8), (xs >= cx - 20) & (xs < cx + 20))]
                    rec[name] = float(np.mean(v))
                profile_rows.append(rec)
        if fid == '014':
            # Display only; equal transform across normalized response views.
            xs = [float(r['u_raw_px']) for r in frame_spots]; ys = [float(r['v_raw_px']) for r in frame_spots]
            x0, x1 = int((min(xs)-120)//2), int((max(xs)+120)//2)
            y0, y1 = int((min(ys)-120)//2), int((max(ys)+120)//2)
            montage = [planes['R'][y0:y1, x0:x1],
                       (planes['G1'][y0:y1, x0:x1]+planes['G2'][y0:y1, x0:x1])/2]
        hash_after = digest(path)
        assert hash_before == hash_after, 'RAW changed during read-only analysis'
        metadata.append({'frame_id': fid, 'sha256_before': hash_before, 'sha256_after': hash_after,
                         'black_levels': black, 'camera_white_levels': white,
                         'compression': str(exif.get('EXIF SubIFD0 Compression', 'unknown')),
                         'bits_per_sample': str(exif.get('EXIF SubIFD0 BitsPerSample', 'unknown')),
                         'exposure_time': str(exif.get('EXIF ExposureTime', 'unknown')),
                         'ISO': str(exif.get('EXIF ISOSpeedRatings', 'unknown')),
                         'f_number': str(exif.get('EXIF FNumber', 'unknown')),
                         'focal_length': str(exif.get('EXIF FocalLength', 'unknown'))})
        print(f'{fid}: measured {len(frame_spots)} diagnostic apertures; RAW hash unchanged', flush=True)

    fits = {}; per_frame = []; bins = []; validation = []
    for bs in (16, 32):
        clean = [r for r in blocks if r['block_raw_px'] == bs and r['R_safe_neighbour'] and not r['overlap_risk'] and r['R'] >= 100]
        fits[str(bs)] = regress(clean, 'R', 'G')
        for fid in FRAME_IDS:
            part = [r for r in clean if r['frame_id'] == fid]
            per_frame.append({'frame_id': fid, 'block_raw_px': bs, **regress(part, 'R', 'G')})
            train = [r for r in clean if r['frame_id'] != fid]
            fit = regress(train, 'R', 'G')
            if part:
                err = [(r['G'] - fit['slope'] * r['R'] - fit['intercept_DN']) / (fit['slope'] * r['R'] + fit['intercept_DN']) for r in part]
                validation.append({'frame_id': fid, 'block_raw_px': bs, 'n': len(part),
                                   'median_abs_prediction_residual_pct': float(np.median(np.abs(err))*100),
                                   'p95_abs_prediction_residual_pct': float(np.percentile(np.abs(err),95)*100)})
        for lo, hi in ((100,300), (300,1000), (1000,3000), (3000,12000)):
            part = [r for r in clean if lo <= r['R'] < hi]
            bins.append({'block_raw_px': bs, 'R_min':lo, 'R_max':hi, **regress(part,'R','G')})
        for label, predicate in [('all',lambda r:True), ('R_safe',lambda r:r['R_safe_neighbour']),
                                  ('R_clipped_neighbour',lambda r:r['R_clipped_neighbour'])]:
            part=[r for r in blocks if r['block_raw_px']==bs and r['G']>=10 and predicate(r)]
            fits[f'G2_G1_{bs}_{label}']=regress(part,'G1','G2')

    trial_rows = [r for r in aperture_rows if r['frame_id'] != '070']
    guided = [r for r in trial_rows if r['detection_origin'].startswith('guided')]
    strong = [r for r in trial_rows if not r['detection_origin'].startswith('guided')]
    clean_apertures = [r for r in aperture_rows if r['R_below_conservative_cap'] and r['R_mean_r50_DN'] >= 30 and 'OVERLAP' not in r['quality_flags']]
    aperture_fits = {str(rad):regress(clean_apertures, f'R_mean_r{rad}_DN', f'G_mean_r{rad}_DN') for rad in (35,50,65)}
    aperture_frame_fits = [{'frame_id':fid, **regress([r for r in clean_apertures if r['frame_id']==fid],
                            'R_mean_r50_DN','G_mean_r50_DN')} for fid in FRAME_IDS]
    aperture_loo = []
    for fid in FRAME_IDS:
        train = [r for r in clean_apertures if r['frame_id'] != fid]
        test = [r for r in clean_apertures if r['frame_id'] == fid]
        f=regress(train,'R_mean_r50_DN','G_mean_r50_DN')
        if test:
            pred=np.array([f['slope']*r['R_mean_r50_DN']+f['intercept_DN'] for r in test])
            err=np.abs(np.array([r['G_mean_r50_DN'] for r in test])-pred)/pred
            aperture_loo.append({'frame_id':fid,'n':len(test),'median_abs_relative_residual_pct':float(np.median(err)*100),
                                 'p95_abs_relative_residual_pct':float(np.percentile(err,95)*100)})
    control_result=next(r for r in aperture_rows if r['frame_id']=='070')
    cf=aperture_fits['50']; cpred=cf['slope']*control_result['R_mean_r50_DN']+cf['intercept_DN']
    control_summary={'R_mean_DN':control_result['R_mean_r50_DN'],'G_mean_DN':control_result['G_mean_r50_DN'],
                     'G_over_R':control_result['G_over_R_r50'],'G_predicted_DN':cpred,
                     'prediction_relative_difference_pct':100*(control_result['G_mean_r50_DN']-cpred)/cpred,
                     'included_in_conservative_aperture_fit':False,
                     'reason':'R max 12718 DN is below recorded white but above the deliberately stricter 80% usable-range cutoff.'}
    group_summary={}
    for name, rows in [('trial_all',trial_rows),('guided_weak',guided),('original_strong',strong)]:
        group_summary[name]={'n':len(rows), 'G_stable_count':sum(r['G_diagnostic_stable'] for r in rows),
            'G_mean_range_DN':[min(r['G_mean_r50_DN'] for r in rows),max(r['G_mean_r50_DN'] for r in rows)],
            'G_background_variation_fraction_p50_p95':np.percentile([r['G_background_variation_fraction'] for r in rows],[50,95]).tolist(),
            'G2_G1_abs_difference_fraction_p50_p95':np.percentile([abs(r['G2_G1_fraction_difference']) for r in rows],[50,95]).tolist()}
    result={'scope':'Eight existing trial frames, 98 candidates; one additional 070 main-peak control. Existing 80-frame summary reused, not reprocessed.',
            'all80_existing_diagnostics':global_summary, 'block_fits':fits, 'aperture_fits':aperture_fits,
            'groups':group_summary, '070_held_out_aperture_control':control_summary,
            'method':{'aperture_radii_RAW_pixels':[35,50,65], 'primary_radius':50,
                'primary_background_annulus':[85,120], 'alternate_background_annulus':[110,150],
                'background':'iteratively clipped local plane; compare alternative plane and primary-annulus median',
                'block_sizes_RAW_pixels':[16,32], 'R_threshold_DN_after_black':100,
                'R_cap_fraction_of_camera_white_after_black':.8, 'R_cap_neighbour_buffer_RAW_pixels':16,
                'aperture_red_safety_radius_RAW_pixels':81,
                'G_stability_rule':'G_mean >= 5*max(background choice change, empirical sector+sampling error)',
                'stability_rule_is_not':'Formal SNR, a detector calibration, an accuracy guarantee, or diffraction classification.',
                'negative_residuals_preserved':True, 'demosaic':False, 'photometry_valid':False},
            'limitations':['R/G consistency is not an independent optical-exposure linearity calibration.',
                'Aperture proxies are not total diffraction-order power; broad peaks and overlapping/background tails require a separate integration model.',
                'No measured Sony saturation-crosstalk radius, dark/gain/flat-field calibration, or laser power monitoring.',
                'G1 and G2 block averages have offset sampling positions; local gradients/speckle can change their ratios.',
                'Laser wavelength unknown; user confirms same laser and unchanged power setting, not measured constant output.']}
    save_csv(out/'aperture_diagnostics.csv',aperture_rows)
    save_csv(out/'block_diagnostics.csv',blocks)
    save_csv(out/'frame_fits.csv',per_frame)
    save_csv(out/'brightness_bin_fits.csv',bins)
    save_csv(out/'leave_one_frame_out.csv',validation)
    save_csv(out/'aperture_frame_fits.csv',aperture_frame_fits)
    save_csv(out/'aperture_leave_one_frame_out.csv',aperture_loo)
    save_csv(out/'core_profiles.csv',profile_rows)
    save_json(out/'sample_metadata.json',metadata)
    save_json(out/'results.json',result)

    fig, ax = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    selected = [r for r in blocks if r['block_raw_px']==32 and r['R_safe_neighbour'] and not r['overlap_risk'] and r['R']>=100]
    for fid in FRAME_IDS:
        pp=[r for r in selected if r['frame_id']==fid]
        if pp: ax[0].scatter([r['R'] for r in pp],[r['G'] for r in pp],s=14,alpha=.7,label=fid)
    f=fits['32']; xx=np.linspace(100,max(r['R'] for r in selected),200)
    ax[0].plot(xx,f['slope']*xx+f['intercept_DN'],'k--',lw=1,label=f"fit: {f['slope']:.4f} R {f['intercept_DN']:+.3f}")
    ax[0].set(xlabel='R, background-subtracted block mean (DN)',ylabel='G = (G1 + G2)/2 (DN)',title=f"Unsaturated common blocks; n={len(selected)}, R²={f['R_squared']:.5f}")
    ax[0].legend(fontsize=8)
    for fid in FRAME_IDS:
        pp=[r for r in selected if r['frame_id']==fid]
        ax[1].scatter([r['R'] for r in pp],[r['G']/r['R'] for r in pp],s=14,alpha=.7)
    ax[1].axhline(f['zero_intercept_slope'],color='k',ls='--',lw=1)
    ax[1].set(xlabel='R, background-subtracted block mean (DN)',ylabel='G / R',xscale='log',title='Ratio scatter reveals limits hidden by R²')
    fig.savefig(out/'01_red_green_relation.png'); plt.close(fig)

    fig, ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for label,color,part in [('Original candidates','#2471a3',strong),('Guided weak candidates','#b33a3a',guided)]:
        ax[0].scatter([r['R_mean_r50_DN'] for r in part],[r['G_mean_r50_DN'] for r in part],label=label,c=color,s=28)
        ax[1].scatter([r['G_mean_r50_DN'] for r in part],[r['G_background_variation_fraction']*100 for r in part],label=label,c=color,s=28)
    xx=np.geomspace(1,16000,200); ax[0].plot(xx, fits['32']['zero_intercept_slope']*xx,'k--',lw=1,label='Unsaturated-block ratio')
    ax[0].set(xscale='symlog',yscale='symlog',xlabel='R mean in radius-50 aperture (DN)',ylabel='G mean in same aperture (DN)',title='Same aperture; red-clipped points are included here')
    ax[0].legend(fontsize=8)
    ax[1].set(xscale='symlog',yscale='symlog',xlabel='G mean in radius-50 aperture (DN)',ylabel='Change with background choice (%)',title='Weak green signals are background-sensitive')
    ax[1].axhline(20,c='k',ls='--',lw=1)
    fig.savefig(out/'02_weak_signal_limits.png'); plt.close(fig)

    fig, ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for axis,fid in zip(ax,('014','046','070')):
        pp=[r for r in profile_rows if r['frame_id']==fid]
        for name,color,mult in [('R','#b33a3a',fits['32']['zero_intercept_slope']),('G1','#21966b',1),('G2','#376db0',1)]:
            axis.plot([r['vertical_offset_raw_px'] for r in pp],[r[name]*mult for r in pp],label=f'{name}'+(' × fitted G/R' if name=='R' else ''),c=color)
        axis.set(title=f'{fid} main-peak stripe',xlabel='Vertical offset (RAW px)',ylabel='Black-subtracted stripe mean (DN)')
        axis.legend(fontsize=8)
    fig.savefig(out/'03_core_profiles.png'); plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    ratio=fits['32']['zero_intercept_slope']
    for axis,data,title in zip(ax,[montage[0],montage[1]/ratio],['014: R CFA display','014: G CFA / fitted G:R ratio (display only)']):
        axis.imshow(np.arcsinh(np.maximum(data,0)/8),cmap='gray',vmin=0,vmax=np.arcsinh((16383-512)/8))
        axis.set_title(title); axis.axis('off')
    fig.savefig(out/'04_014_channel_views.png'); plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for fid in FRAME_IDS:
        pp=[r for r in clean_apertures if r['frame_id']==fid]
        if pp:
            ax[0].scatter([r['R_mean_r50_DN'] for r in pp],[r['G_mean_r50_DN'] for r in pp],s=30,label=fid)
            ax[1].scatter([r['R_mean_r50_DN'] for r in pp],[r['G_over_R_r50'] for r in pp],s=30)
    f=aperture_fits['50']; xx=np.linspace(30,650,200)
    ax[0].plot(xx,f['slope']*xx+f['intercept_DN'],'k--',lw=1,label=f"G = {f['slope']:.5f} R {f['intercept_DN']:+.3f}")
    ax[0].set(xlabel='R, radius-50 aperture mean (DN)',ylabel='G, radius-50 aperture mean (DN)',title=f"{len(clean_apertures)} unsaturated apertures; R²={f['R_squared']:.6f}")
    ax[0].legend(fontsize=9)
    ax[1].axhline(f['zero_intercept_slope'],c='k',ls='--',lw=1)
    ax[1].scatter([control_result['R_mean_r50_DN']],[control_result['G_over_R_r50']],marker='*',s=150,c='black',label='070 held-out unsaturated control')
    ax[1].set(xlabel='R, radius-50 aperture mean (DN)',ylabel='G / R',xscale='log',title='Area averaging reduces CFA position / speckle differences')
    ax[1].legend(fontsize=8)
    fig.savefig(out/'05_aperture_relation.png'); plt.close(fig)

    save_json(out/'provenance.json',{'script':str(Path(__file__).resolve()),'script_sha256':digest(Path(__file__)),
             'inputs':[{'path':str(p),'sha256':digest(p)} for p in [TRIAL/'spots.csv',TRIAL/'rejected_candidates.csv',
                 OLD/'spots.csv',BASE/'outputs/rawcheck_20261010/raw_channel_diagnostics.csv']],
             'versions':{'numpy':np.__version__,'rawpy':rawpy.__version__,'matplotlib':matplotlib.__version__}})
    print(json.dumps({'block_fits':fits,'groups':group_summary,'aperture_fits':aperture_fits},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
