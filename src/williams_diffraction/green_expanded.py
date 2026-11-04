"""Search all 80 RAW for strong, independent-channel response samples.

The search is an expanded stage-one experiment, not production diffraction
classification. Candidate selection never uses G/R or regression residuals.
"""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

from .green_channel_study import BASE, RAW, CHANNELS, read_csv, save_csv, save_json, digest, background, aperture_weights, regress
from .geometry import centroid, FONT_PATH
from .guided_detection import reflection_bank, find_reflections


def measure(mosaic, black, center, neighbours, reflections):
    cx,cy=center; allstats={}
    for name,(dy,dx,ci) in CHANNELS.items():
        j0=int(np.floor((cx-170-dx)/2)); j1=int(np.ceil((cx+170-dx)/2))
        i0=int(np.floor((cy-170-dy)/2)); i1=int(np.ceil((cy+170-dy)/2))
        xx,yy=np.meshgrid(np.arange(j0,j1)*2+dx-cx,np.arange(i0,i1)*2+dy-cy)
        z=mosaic[dy::2,dx::2][i0:i1,j0:j1].astype(float)-black[ci]
        rr=np.hypot(xx,yy); valid=np.ones_like(rr,bool)
        for nx,ny in neighbours:
            if np.hypot(nx-cx,ny-cy)>3: valid &= (xx+cx-nx)**2+(yy+cy-ny)**2>70**2
        for nx,ny in reflections: valid &= (xx+cx-nx)**2+(yy+cy-ny)**2>90**2
        plane,coef,sigma,bgse,med,nbg=background(z,xx,yy,valid&(rr>=85)&(rr<=120))
        alt,*_=background(z,xx,yy,valid&(rr>=110)&(rr<=150))
        st={'bg_DN':float(coef[0]),'bg_n':nbg,'bg_sigma_DN':sigma,
            'max_raw_within81_DN':float(np.max((z+black[ci])[rr<=81]))}
        for rad in (35,50,65):
            w=aperture_weights(xx,yy,rad); n=float(w.sum())
            st[f'mean_r{rad}_DN']=float(np.sum(w*(z-plane))/n)
            st[f'Q_r{rad}']=float(4*np.sum(w*(z-plane)))
            st[f'weighted_n_r{rad}']=n
            if rad==50:
                st['mean_alt_DN']=float(np.sum(w*(z-alt))/n)
                st['mean_med_DN']=float(np.sum(w*(z-med))/n)
        allstats[name]=st
    row={f'{name}_{k}':v for name,s in allstats.items() for k,v in s.items()}
    for rad in (35,50,65):
        row[f'G_mean_r{rad}_DN']=(allstats['G1'][f'mean_r{rad}_DN']+allstats['G2'][f'mean_r{rad}_DN'])/2
        row[f'Q_G_r{rad}']=(allstats['G1'][f'Q_r{rad}']+allstats['G2'][f'Q_r{rad}'])/2
        row[f'Q_R_r{rad}']=allstats['R'][f'Q_r{rad}']
        row[f'Q_B_r{rad}']=allstats['B'][f'Q_r{rad}']
        row[f'Q_RGB_r{rad}']=row[f'Q_R_r{rad}']+row[f'Q_G_r{rad}']+row[f'Q_B_r{rad}']
    r=row['R_mean_r50_DN']
    row['R_background_change_fraction']=max(abs(allstats['R']['mean_alt_DN']-r),abs(allstats['R']['mean_med_DN']-r))/max(abs(r),1e-12)
    g=row['G_mean_r50_DN']
    ga=(allstats['G1']['mean_alt_DN']+allstats['G2']['mean_alt_DN'])/2
    gm=(allstats['G1']['mean_med_DN']+allstats['G2']['mean_med_DN'])/2
    row['G_background_change_fraction']=max(abs(ga-g),abs(gm-g))/max(abs(g),1e-12)
    row['G2_G1_fraction_difference']=(allstats['G2']['mean_r50_DN']-allstats['G1']['mean_r50_DN'])/max(abs(g),1e-12)
    return row


def image_patch(red,c,half):
    x,y=np.rint(np.array(c)/2).astype(int)
    z=red[y-half:y+half,x-half:x+half]
    return np.uint8(np.clip(np.arcsinh(np.maximum(z,0)/8)/np.arcsinh((16383-512)/8)*255,0,255))


def search(out):
    import rawpy
    from .paths import require_writable_run
    out=require_writable_run(out)
    out.mkdir(parents=True,exist_ok=False); (out/'candidate_views').mkdir()
    config=json.loads((BASE/'annotations/algorithm_trial_v2.json').read_text())
    old=read_csv(BASE/'outputs/stage1_20261010_corrected/spots.csv')
    bank=reflection_bank(config,old)
    frames=read_csv(BASE/'outputs/stage1_20261010_corrected/frames.csv')
    frame_map={r['frame_id']:r for r in frames}; candidates=[]; summaries=[]; reflected=[]
    cap=512+.8*(15360-512)
    for fid in [f'{i:03d}' for i in range(1,81)]:
        path=RAW/f'{fid}.ARW'
        before=digest(path)
        assert before==frame_map[fid]['sha256']
        with rawpy.imread(str(path)) as raw:
            assert raw.raw_pattern.tolist()==[[0,1],[3,2]] and raw.color_desc==b'RGBG'
            mosaic=raw.raw_image_visible.copy(); black=raw.black_level_per_channel
        red=mosaic[::2,::2].astype(np.float32)-black[0]
        small=ndi.gaussian_filter(red[::4,::4],1)
        ya,xa=np.unravel_index(np.argmax(small),small.shape); xa*=4; ya*=4
        local=ndi.gaussian_filter(red[ya-35:ya+36,xa-35:xa+36],4)
        iy,ix=np.unravel_index(np.argmax(local),local.shape); anchor=np.array([xa-35+ix,ya-35+iy],float)
        x0=max(100,int(anchor[0])-720);x1=min(red.shape[1]-100,int(anchor[0])+720)
        y0=max(100,int(anchor[1])-500);y1=min(red.shape[0]-100,int(anchor[1])+500)
        sub=red[y0:y1,x0:x1]
        smooth=ndi.gaussian_filter(sub,4); contrast=smooth-ndi.gaussian_filter(sub,35)
        yy,xx=np.where((contrast==ndi.maximum_filter(contrast,size=61))&(contrast>=35))
        seeds=sorted([(float(contrast[y,x]),np.array([x+x0,y+y0],float)) for x,y in zip(xx,yy)],key=lambda v:-v[0])
        ref=find_reflections(red,[x0,y0,x1,y1],anchor,bank,config['reflection'])
        # Keep the four user-reviewed examples excluded even if automatic
        # template search were to miss one in a changed search crop.
        for label in config['reflection']['review_labels']:
            if label['frame_id']==fid:
                r=next(v for v in old if v['frame_id']==fid and v['spot_id']==label['spot_id'])
                q=np.array([float(r['u_raw_px']),float(r['v_raw_px'])])/2
                if not any(np.linalg.norm(q-v['q'])*2<70 for v in ref): ref.append({'q':q,'decision_source':'user_reviewed'})
        rc=[np.array(r['q'])*2 for r in ref]
        for r in ref: reflected.append({'frame_id':fid,'u_raw_px':float(r['q'][0]*2),'v_raw_px':float(r['q'][1]*2),'source':r['decision_source']})
        loc=[]
        for amp,q in seeds:
            c,bg,major,ratio=centroid(red,q,25)
            if all(np.linalg.norm(c-v['center'])*2>=75 for v in loc):
                loc.append({'center':c,'contrast':amp,'major':major,'ratio':ratio})
        centers=[v['center']*2 for v in loc]
        pool_this=[]
        for n,v in enumerate(loc,1):
            c=v['center']*2
            row={'frame_id':fid,'candidate_id':f'c{n:03d}','u_raw_px':float(c[0]),'v_raw_px':float(c[1]),
                 'voltage_V':frame_map[fid]['voltage_recorded_V'],'scan_direction':frame_map[fid]['scan_direction'],
                 'contrast_DN':v['contrast'],'major_rms_raw_px':v['major'],'axis_ratio':v['ratio'],
                 'photometry_valid':False,'purpose':'expanded_CFA_channel_response_diagnostic'}
            reasons=[]
            if any(np.linalg.norm(c-r)<150 for r in rc): reasons.append('REFLECTION_OR_NEAR_REFLECTION')
            distances=[np.linalg.norm(c-nc) for nc in centers if np.linalg.norm(c-nc)>3]
            nearest=min(distances,default=1e9); row['nearest_candidate_RAW_px']=nearest
            if nearest<150: reasons.append('NEARBY_PEAK_OVERLAP_RISK')
            if not 10<=v['major']<=65 or v['ratio']>4: reasons.append('MORPHOLOGY_REVIEW')
            if v['contrast']<60: reasons.append('R_CONTRAST_TOO_LOW')
            if not (170<c[0]<mosaic.shape[1]-170 and 170<c[1]<mosaic.shape[0]-170): reasons.append('IMAGE_EDGE')
            if not reasons:
                try:
                    row.update(measure(mosaic,black,c,centers,rc))
                    if row['R_max_raw_within81_DN']>=cap: reasons.append('R_NEAR_SATURATION')
                    if row['R_mean_r50_DN']<50: reasons.append('R_MEAN_TOO_LOW')
                    if row['R_background_change_fraction']>.10: reasons.append('R_BACKGROUND_SENSITIVE')
                except ValueError as e:
                    reasons.append('BACKGROUND_INSUFFICIENT');row['measurement_error']=str(e)
            row['eligible']=not reasons; row['rejection_reasons']=';'.join(reasons)
            candidates.append(row)
            if row['eligible']:
                pool_this.append(row)
                key=f'{fid}_{row["candidate_id"]}'
                im=Image.fromarray(image_patch(red,c,105)).resize((420,420),Image.Resampling.NEAREST).convert('RGB')
                draw=ImageDraw.Draw(im);draw.ellipse([160,160,260,260],outline='cyan',width=2)
                im.save(out/'candidate_views'/f'{key}.png')
                # Context around the main beam allows manual lattice/row review.
        if pool_this:
            cx,cy=anchor.astype(int);q=red[max(0,cy-420):min(red.shape[0],cy+420),max(0,cx-660):min(red.shape[1],cx+660)]
            disp=np.uint8(np.clip(np.arcsinh(np.maximum(q,0)/8)/np.arcsinh((16383-512)/8)*255,0,255))
            im=Image.fromarray(disp).convert('RGB');dr=ImageDraw.Draw(im)
            font=ImageFont.truetype(FONT_PATH,22)
            for r in pool_this:
                x=r['u_raw_px']/2-max(0,cx-660);y=r['v_raw_px']/2-max(0,cy-420)
                dr.ellipse([x-25,y-25,x+25,y+25],outline='cyan',width=2);dr.text((x+28,y-14),r['candidate_id'],fill='cyan',font=font)
            im.resize((990,630)).save(out/'candidate_views'/f'{fid}_context.png')
        after=digest(path);assert before==after
        summaries.append({'frame_id':fid,'candidate_count':len(loc),'eligible_count':len(pool_this),'reflections':len(rc),
                          'sha256_before':before,'sha256_after':after,'anchor_RAW_x':float(anchor[0]*2),'anchor_RAW_y':float(anchor[1]*2)})
        if int(fid)%5==0: print(f'{fid}/80 searched; eligible={sum(r["eligible"] for r in candidates)}, frames={sum(r["eligible_count"]>0 for r in summaries)}',flush=True)
    save_csv(out/'candidate_pool.csv',candidates);save_csv(out/'frame_search.csv',summaries);save_csv(out/'reflections.csv',reflected)
    save_json(out/'search_rules.json',{'R_mean_min_DN':50,'R_detection_contrast_min_DN':60,'R_background_change_max':.1,
       'R_max_within81_cap_DN':cap,'min_neighbour_distance_RAW_px':150,'morphology_major_rms_RAW_px':[10,65],
       'max_axis_ratio':4,'G_used_for_selection':False,'ratio_used_for_selection':False,
       'aperture_radii':[35,50,65],'background_annuli':[[85,120],[110,150]],
       'selection_note':'This is a candidate pool requiring visual context review, not all-frame diffraction classification.'})
    print(f'POOL: {sum(r["eligible"] for r in candidates)} eligible in {sum(r["eligible_count"]>0 for r in summaries)} frames',flush=True)


def select_points(pool,excluded_frames,excluded_keys):
    rows=[r for r in pool if r['eligible']=='True' and r['frame_id'] not in excluded_frames
          and f"{r['frame_id']}_{r['candidate_id']}" not in excluded_keys]
    # Choose one brightest R candidate per frame, then cover both scan direction
    # and the entire logarithmic R signal range. Only R determines this choice.
    top=[]
    for fid in sorted({r['frame_id'] for r in rows}):
        top.append(max([r for r in rows if r['frame_id']==fid],key=lambda r:float(r['Q_R_r50'])))
    if len(top)<=40: selected=top[:]
    else:
        up=sorted([r for r in top if r['scan_direction']=='up'],key=lambda r:float(r['Q_R_r50']))
        down=sorted([r for r in top if r['scan_direction']=='down'],key=lambda r:float(r['Q_R_r50']))
        nup=round(40*len(up)/len(top));ndown=40-nup
        selected=[v[i] for v,n in [(up,nup),(down,ndown)] for i in np.rint(np.linspace(0,len(v)-1,n)).astype(int)]
    if len(selected)<40:
        used={(r['frame_id'],r['candidate_id']) for r in selected}
        rest=sorted([r for r in rows if (r['frame_id'],r['candidate_id']) not in used],key=lambda r:float(r['Q_R_r50']))
        n=min(40-len(selected),len(rest))
        selected += [rest[i] for i in np.rint(np.linspace(0,len(rest)-1,n)).astype(int)] if n else []
    return sorted(selected,key=lambda r:(r['frame_id'],r['candidate_id']))


def fit(out,excluded_frames,excluded_keys):
    from .paths import require_writable_run
    out=require_writable_run(out)
    pool=read_csv(out/'candidate_pool.csv'); selected=select_points(pool,excluded_frames,excluded_keys)
    review=json.loads((out/'visual_review.json').read_text())
    accepted=set(review['accepted_keys'])
    assert all(f"{r['frame_id']}_{r['candidate_id']}" in accepted for r in selected), 'Visual context review required before fitting'
    os.environ['MPLCONFIGDIR']=str(out/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    for i,r in enumerate(selected,1):
        r['sample_id']=f'{i:02d}';r['visual_review']='PASS_PATTERN_CONTEXT';r['selection_G_ratio_or_fit_residual_used']=False
    converted=[{k:float(v) if k.startswith('Q_') or k.endswith('_DN') and v else v for k,v in r.items()} for r in selected]
    fits={}
    for rad in (35,50,65):
        fits[f'G_R_{rad}']=regress(converted,f'Q_R_r{rad}',f'Q_G_r{rad}')
        fits[f'G_RGB_{rad}']=regress(converted,f'Q_RGB_r{rad}',f'Q_G_r{rad}')
    old=json.loads((BASE/'outputs/green_channel_study_20261010_r2/results.json').read_text())['aperture_fits']['50']
    old_err=[];loo=[]
    for row,r in zip(selected,converted):
        pred=old['slope']*r['R_mean_r50_DN']+old['intercept_DN']
        err=(r['G_mean_r50_DN']-pred)/pred
        row['previous38_model_prediction_G_mean_DN']=pred;row['previous38_model_residual_pct']=err*100;old_err.append(err)
        train=[v for v in converted if v['frame_id']!=r['frame_id']]
        f=regress(train,'Q_R_r50','Q_G_r50'); pp=f['slope']*r['Q_R_r50']+f['intercept_DN']
        loo.append({'sample_id':row['sample_id'],'frame_id':r['frame_id'],'residual_pct':100*(r['Q_G_r50']-pp)/pp})
    rng=np.random.default_rng(20261010);unique=sorted({r['frame_id'] for r in converted});boot=[]
    for _ in range(3000):
        chosen=rng.choice(unique,len(unique),replace=True)
        rr=[v for fid in chosen for v in converted if v['frame_id']==fid]
        x=np.array([r['Q_R_r50'] for r in rr]);y=np.array([r['Q_G_r50'] for r in rr]);boot.append(np.polyfit(x,y,1)[0])
    summary={'n':len(selected),'distinct_frames':len(unique),'up_points':sum(r['scan_direction']=='up' for r in selected),
             'down_points':sum(r['scan_direction']=='down' for r in selected),'fits':fits,
             'previous38_prediction_median_abs_relative_pct':float(np.median(np.abs(old_err))*100),
             'previous38_prediction_p95_abs_relative_pct':float(np.percentile(np.abs(old_err),95)*100),
             'leave_one_frame_out_median_abs_relative_pct':float(np.median([abs(r['residual_pct']) for r in loo])),
             'leave_one_frame_out_p95_abs_relative_pct':float(np.percentile([abs(r['residual_pct']) for r in loo],95)),
             'slope_frame_cluster_bootstrap_95_interval':np.percentile(boot,[2.5,97.5]).tolist(),
             'excluded_frames':excluded_frames,'excluded_keys':excluded_keys,
             'physical_total_power_known':False,'RGB_sum_contains_G':True,
             'measurement':'Same-aperture CFA density-corrected integration; not full-peak power or watts',
             'fits_by_scan':{direction:regress([r for r in converted if r['scan_direction']==direction],'Q_R_r50','Q_G_r50') for direction in ('up','down')},
             'visual_review':'40 selected patches and associated pattern contexts reviewed before regression'}
    save_csv(out/'selected40.csv',selected);save_csv(out/'leave_one_frame_out.csv',loo);save_json(out/'fit_results.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.4),layout='constrained')
    colors={'up':'#d45340','down':'#2378a8'}
    for ax,key,fitkey,title,label in [(axes[0],'Q_R_r50','G_R_50','Independent channel check','Optical-intensity proxy: unsaturated R'),
                                     (axes[1],'Q_RGB_r50','G_RGB_50','Camera RGB sum (contains G)','Camera RGB sum: R + G + B')]:
        for direction in ('up','down'):
            pp=[r for r in converted if r['scan_direction']==direction]
            ax.scatter([r[key]/1e6 for r in pp],[r['Q_G_r50']/1e6 for r in pp],s=42,c=colors[direction],label=f'{direction} scan',zorder=3)
        f=fits[fitkey];x=np.linspace(0,max(r[key] for r in converted)*1.06,200)/1e6
        ax.plot(x,f['slope']*x+f['intercept_DN']/1e6,'k--',label=f"G = {f['slope']:.5f} x {f['intercept_DN']/1e6:+.4f}\nR² = {f['R_squared']:.6f}")
        ax.set(xlabel=label+'\n(integrated DN × RAW pixel area / 10⁶)',ylabel='Green integrated signal\n(DN × RAW pixel area / 10⁶)',title=f'{title}; n={len(selected)}')
        ax.legend(fontsize=10);ax.grid(alpha=.16)
    fig.savefig(out/'fit40.png');fig.savefig(out/'fit40.pdf');plt.close(fig)
    f=fits['G_R_50'];fig,axes=plt.subplots(1,2,figsize=(13.5,4.8),layout='constrained')
    for direction in ('up','down'):
        pp=[r for r in converted if r['scan_direction']==direction]
        pred=np.array([f['slope']*r['Q_R_r50']+f['intercept_DN'] for r in pp]);obs=np.array([r['Q_G_r50'] for r in pp]);err=100*(obs-pred)/pred
        axes[0].scatter([r['Q_R_r50']/1e6 for r in pp],err,s=38,c=colors[direction],label=direction)
        axes[1].scatter([int(r['frame_id']) for r in pp],err,s=38,c=colors[direction],label=direction)
    for ax in axes:
        ax.axhline(0,c='k',lw=1);ax.axhline(5,c='gray',ls=':',lw=1);ax.axhline(-5,c='gray',ls=':',lw=1);ax.grid(alpha=.16)
        ax.set(ylabel='Relative residual to fitted G (%)');ax.legend()
    axes[0].set(xlabel='Unsaturated R integrated signal / 10⁶',title=f"Median |residual| = {f['median_abs_relative_residual_pct']:.2f}%\n95th percentile = {f['p95_abs_relative_residual_pct']:.2f}%")
    axes[1].set(xlabel='Chronological RAW frame number',title='Residual versus frame / scan direction')
    fig.savefig(out/'residual40.png');plt.close(fig)
    # Forty selected patches and their identities for explicit visual review.
    font=ImageFont.truetype(FONT_PATH,17); sheet=Image.new('RGB',(5*260,8*298),'#10171c');dr=ImageDraw.Draw(sheet)
    for i,r in enumerate(selected):
        x=(i%5)*260;y=(i//5)*298;im=Image.open(out/'candidate_views'/f'{r["frame_id"]}_{r["candidate_id"]}.png').resize((250,250))
        sheet.paste(im,(x+5,y+43));dr.text((x+7,y+3),f'{r["sample_id"]}: {r["frame_id"]} {r["candidate_id"]}',font=font,fill='white')
        dr.text((x+7,y+22),f'R={float(r["R_mean_r50_DN"]):.1f} G={float(r["G_mean_r50_DN"]):.1f} DN',font=font,fill='#a8dfe9')
    sheet.save(out/'selected40_contact.png')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['search','fit']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--exclude-frames',default='012,014,018,026,064')
    p.add_argument('--exclude-keys',default='')
    args=p.parse_args()
    if args.mode=='search':search(args.out)
    else:fit(args.out,[x for x in args.exclude_frames.split(',') if x],[x for x in args.exclude_keys.split(',') if x])
