"""Measured mode separation, CFA background and leave-one-out brightness evidence.

These are detection diagnostics, never CFA flux or a physical-order assignment.
Coordinates in this module are R-sample centres unless explicitly labelled RAW.
"""
import numpy as np
from scipy import ndimage as ndi
from .geometry import transform


def aligned_green(mosaic, black):
    """Average the independently black-subtracted G sites at the R centres."""
    g1=mosaic[::2,1::2].astype(np.float32)-black[1]
    g2=mosaic[1::2,::2].astype(np.float32)-black[3]
    return .5*(ndi.shift(g1,(0.,.5),order=1,mode='nearest',prefilter=False)
               +ndi.shift(g2,(.5,0.),order=1,mode='nearest',prefilter=False))


def sample(field,q):
    return float(ndi.map_coordinates(field,[[q[1]],[q[0]]],order=1,mode='nearest')[0])


def hessian(field,q):
    x,y=np.rint(q).astype(int)
    if not (1<=x<field.shape[1]-1 and 1<=y<field.shape[0]-1):return None
    dxx=float(field[y,x+1]+field[y,x-1]-2*field[y,x])
    dyy=float(field[y+1,x]+field[y-1,x]-2*field[y,x])
    dxy=float(field[y+1,x+1]-field[y+1,x-1]-field[y-1,x+1]+field[y-1,x-1])/4
    return np.linalg.eigvalsh([[dxx,dxy],[dxy,dyy]])


def connected_saddle(field,a,b,sigma,valid=None):
    """Maximin 8-connected path in a declared local ROI, not just a line cut."""
    a,b=np.rint(a).astype(int),np.rint(b).astype(int)
    padding=int(np.ceil(max(3*sigma,.5*np.linalg.norm(a-b))))
    lo=np.maximum(np.minimum(a,b)-padding,0)
    hi=np.minimum(np.maximum(a,b)+padding+1,field.shape[::-1])
    roi=field[lo[1]:hi[1],lo[0]:hi[0]]
    allowed=np.isfinite(roi)
    if valid is not None:allowed=allowed & valid[lo[1]:hi[1],lo[0]:hi[0]].astype(bool)
    aa,bb=a-lo,b-lo
    if not allowed[aa[1],aa[0]] or not allowed[bb[1],bb[0]]:
        return dict(status='OUTSIDE_VALID_SCREEN',saddle_fraction=0.)
    va,vb=float(roi[aa[1],aa[0]]),float(roi[bb[1],bb[0]])
    lower=float(roi[allowed].min());upper=min(va,vb)
    for _ in range(12):
        level=(lower+upper)/2
        labels,_=ndi.label(allowed & (roi>=level),structure=np.ones((3,3),np.uint8))
        connected=labels[aa[1],aa[0]]!=0 and labels[aa[1],aa[0]]==labels[bb[1],bb[0]]
        if connected:lower=level
        else:upper=level
    return dict(status='LOCAL_2D_MAXIMIN_PATH',saddle_DN=float(lower),
        endpoint_DN=[va,vb],saddle_fraction=float((min(va,vb)-lower)/max(min(va,vb),1e-8)),
        roi_sample_bounds=[lo.tolist(),hi.tolist()],connectivity=8)


def resolved_pair(a,b,fields,trees,positions,cfg,valid=None):
    distance=float(np.linalg.norm(a-b)*2)
    record=dict(distance_raw_px=distance,scales=[])
    floor=cfg.get('resolution_floor_raw_px',3*max(fields)*2)
    if distance<floor:return False,{**record,'status':'BELOW_DECLARED_RESOLUTION_FLOOR'}
    for sigma in sorted(set((min(fields),max(fields)))):
        da,ia=trees[sigma].query(a);db,ib=trees[sigma].query(b)
        qa,qb=positions[sigma][ia],positions[sigma][ib]
        s=dict(scale_red_samples=sigma,mode_raw_separation_px=float(np.linalg.norm(qa-qb)*2),
            mode_shifts_raw_px=[float(da*2),float(db*2)])
        record['scales'].append(s)
        if ia==ib or max(da,db)*2>cfg['maximum_scale_shift_raw_px']:
            return False,{**record,'status':'NO_TWO_PERSISTENT_MODES'}
        eigen=[hessian(fields[sigma][0],q) for q in (qa,qb)]
        s['hessian_eigenvalues']=[v.tolist() if v is not None else None for v in eigen]
        if any(v is None or v[-1]>=0 for v in eigen):
            return False,{**record,'status':'NOT_TWO_2D_MODES'}
        saddle=connected_saddle(fields[sigma][0],qa,qb,sigma,valid)
        s.update(saddle)
        if (min(saddle.get('endpoint_DN',[0]))<=0
                or saddle['saddle_fraction']<cfg.get('minimum_saddle_fraction',.15)):
            return False,{**record,'status':'CONNECTED_SINGLE_ENVELOPE'}
    return True,{**record,'status':'RESOLVED_MULTISCALE_PAIR'}


def cfa_background_evidence(q,rfield,gfield,gnoise,valid,cfg):
    """Predict local broadband R texture from G outside the candidate core.

    A robust plane plus bounded R/G gain is trained on an annulus. A distinct
    red excess is protected. Weak/uncorrelated green cannot reject a candidate.
    """
    half=70;x0,y0=np.rint(q).astype(int)
    lo=np.maximum([x0-half,y0-half],0);hi=np.minimum([x0+half+1,y0+half+1],rfield.shape[::-1])
    yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
    xx=xx-q[0];yy=yy-q[1];radius=np.hypot(xx,yy)
    rv=rfield[lo[1]:hi[1],lo[0]:hi[0]].astype(float)
    gv=gfield[lo[1]:hi[1],lo[0]:hi[0]].astype(float)
    use=(radius>=25)&(radius<=70)&valid[lo[1]:hi[1],lo[0]:hi[0]].astype(bool)
    result=dict(method='LEAVE_CORE_OUT_LOCAL_CFA_BACKGROUND',R_contrast_DN=sample(rfield,q),
                G_contrast_DN=sample(gfield,q),status='INSUFFICIENT_BROADBAND_EVIDENCE')
    if use.sum()<300:return result
    gn=gnoise[lo[1]:hi[1],lo[0]:hi[0]][use]
    noise=max(.25,1.4826*float(np.median(abs(gn-np.median(gn)))))
    result['green_noise_DN']=noise
    if result['G_contrast_DN']<max(cfg.get('minimum_green_contrast_DN',2.),5*noise):return result
    X=np.c_[np.ones(use.sum()),xx[use]/70,yy[use]/70,gv[use]]
    y=rv[use];weights=np.ones(len(y));coef=np.zeros(4)
    for _ in range(5):
        sw=np.sqrt(weights)
        coef=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
        residual=y-X@coef
        scatter=max(.25,1.4826*float(np.median(abs(residual-np.median(residual)))))
        weights=np.minimum(1.,2.5*scatter/np.maximum(abs(residual),1e-9))
    gain=float(coef[3]);plane=X[:,:3]@coef[:3]
    corr=float(np.corrcoef((y-plane)[weights>.5],gv[use][weights>.5])[0,1])
    expected=float(coef[0]+gain*result['G_contrast_DN'])
    excess=result['R_contrast_DN']-expected
    result.update(annular_R_per_G=gain,annular_texture_correlation=corr,
        annular_residual_MAD_DN=scatter,predicted_broadband_R_DN=expected,
        red_excess_DN=float(excess),training_samples=int(use.sum()))
    if not np.isfinite(corr):result['annular_texture_correlation']=None;return result
    if not (.2<=gain<=cfg.get('maximum_broadband_R_per_G',3.) and corr>=.7):return result
    # A correlated annulus does not validate extrapolation into a much brighter
    # core. Clipping is one important counterexample to this linear model.
    green_range=np.percentile(gv[use],[1,99])
    result['observed_annular_G_range_DN']=green_range.tolist()
    if not green_range[0]-3*noise<=result['G_contrast_DN']<=green_range[1]+3*noise:
        result['status']='OUTSIDE_OBSERVED_BACKGROUND_RANGE';return result
    # Both absolute residual and a deliberately broad color ratio protect a red
    # spot superposed on an illuminated edge. This is no test for a virtual image.
    if (excess<=3*scatter and result['R_contrast_DN']<2.5*result['G_contrast_DN']
            and abs(excess)<=4*np.hypot(scatter,gain*noise)):
        result['status']='BROADBAND_BACKGROUND_EXPLAINS_CANDIDATE'
    else:result['status']='MEASURED_RED_EXCESS_OR_UNEXPLAINED_FEATURE'
    return result


def single_cfa_impulse_evidence(q,red,fields,valid,minimum_contrast=4.,mask_edges=False):
    """Leave one isolated native R sample out of the detection contrast only.

    A sensor impulse can look like a Gaussian spot after smoothing. Reject its
    hypothesis only when neither scale retains a detectable spatial component
    in the surrounding 40 RAW pixels. RAW samples and photometry stay untouched.
    """
    q=np.asarray(q,float);center=np.rint(q).astype(int)
    lo=np.maximum(center-6,0);hi=np.minimum(center+7,red.shape[::-1])
    cut=red[lo[1]:hi[1],lo[0]:hi[0]]
    result=dict(method='NATIVE_SINGLE_CFA_SAMPLE_ABLATION',status='NO_ISOLATED_NATIVE_IMPULSE')
    if cut.size<25:return result
    at=np.unravel_index(np.argmax(cut),cut.shape);pixel=lo+np.array(at[::-1])
    if np.linalg.norm(pixel-q)>6:return result
    a=np.maximum(pixel-3,0);b=np.minimum(pixel+4,red.shape[::-1])
    neighbourhood=red[a[1]:b[1],a[0]:b[0]].astype(float).ravel()
    ordered=np.sort(neighbourhood);median=float(np.median(ordered[:-1]));mad=max(.25,1.4826*float(np.median(abs(ordered[:-1]-median))))
    impulse=float(ordered[-1]-median);second=max(float(ordered[-2]-median),1.)
    if impulse<max(128.,16*mad) or impulse/second<10:return result
    radius=int(np.ceil(4*max(max(35.,4*s) for s in fields)))+18
    a=np.maximum(center-radius,0);b=np.minimum(center+radius+1,red.shape[::-1])
    impulse_image=np.zeros((b[1]-a[1],b[0]-a[0]),float)
    impulse_image[pixel[1]-a[1],pixel[0]-a[0]]=impulse
    weight=valid[a[1]:b[1],a[0]:b[0]].astype(float);cache={}
    def response(sigma):
        if sigma not in cache:
            cache[sigma]=ndi.gaussian_filter(impulse_image,sigma)
            if mask_edges:cache[sigma]/=np.maximum(ndi.gaussian_filter(weight,sigma),1e-6)
        return cache[sigma]
    p=np.maximum(center-10,0);end=np.minimum(center+11,red.shape[::-1]);residual=[]
    ys=slice(p[1]-a[1],end[1]-a[1]);xs=slice(p[0]-a[0],end[0]-a[0])
    allowed=valid[p[1]:end[1],p[0]:end[0]].astype(bool)
    if not allowed.any():return result
    for sigma,(contrast,_) in fields.items():
        contribution=response(sigma)-response(max(35.,4*sigma))
        remaining=contrast[p[1]:end[1],p[0]:end[0]]-contribution[ys,xs]
        residual.append(dict(scale_red_samples=sigma,maximum_remaining_contrast_DN=float(remaining[allowed].max())))
    return dict(method=result['method'],impulse_pixel_sample_xy=pixel.tolist(),native_excess_DN=impulse,
        native_second_excess_DN=second,native_local_MAD_DN=mad,scales=residual,
        minimum_contrast_DN=minimum_contrast,raw_samples_modified=False,
        status=('SINGLE_CFA_IMPULSE_WITHOUT_SPATIAL_SPOT_SUPPORT'
                if all(r['maximum_remaining_contrast_DN']<minimum_contrast for r in residual)
                else 'NATIVE_IMPULSE_WITH_REMAINING_SPATIAL_SIGNAL_RETAINED'))


def boundary_ridge_evidence(q,coarse,corners,maximum_distance_raw_px=80.,minimum_prominence=.15):
    """Test a weak near-edge hypothesis for a resolved along-edge maximum.

    Sampling is in R CFA coordinates. This does not exclude an entire edge band;
    a localized weak spot can survive arbitrarily close to the accepted margin.
    """
    q=np.asarray(q,float);corners=np.asarray(corners,float);distances=[]
    for a,b in zip(corners,np.roll(corners,-1,axis=0)):
        v=b-a;t=np.clip((q-a)@v/(v@v),0,1);distances.append(float(np.linalg.norm(q-(a+t*v))*2))
    side=int(np.argmin(distances));result=dict(method='COARSE_TANGENTIAL_EDGE_PROFILE',
        side=side,edge_distance_raw_px=distances[side],status='NOT_NEAR_BOUNDARY')
    if distances[side]>maximum_distance_raw_px:return result
    tangent=corners[(side+1)%4]-corners[side];tangent/=np.linalg.norm(tangent)
    steps=np.arange(-40,41,dtype=float);points=q+steps[:,None]*tangent
    values=ndi.map_coordinates(coarse,[points[:,1],points[:,0]],order=1,mode='nearest')
    peak=float(values[40]);result['coarse_contrast_DN']=peak
    if peak<=0:result['status']='NO_POSITIVE_COARSE_CONTRAST';return result
    left=float(values[(steps<=-5)&(steps>=-40)].min());right=float(values[(steps>=5)&(steps<=40)].min())
    prominence=float((peak-max(left,right))/peak)
    result.update(tangential_prominence=prominence,minimum_prominence=minimum_prominence,
        profile_offsets_raw_px=(steps*2).tolist(),profile_contrast_DN=values.tolist(),
        status='RESOLVED_TANGENTIAL_MAXIMUM' if prominence>=minimum_prominence else 'SCREEN_EDGE_RIDGE_WITHOUT_RESOLVED_TANGENTIAL_PEAK')
    return result


def contextual_brightness(raw,amplitudes,H,eligible=None,bilateral_support=False):
    """Leave-one-out local directional context with a robust log interval.

    The interval describes observed image contrast, not a monotonic diffraction
    law. An outlier is isolated from training; its measured signal is retained.
    """
    raw=np.asarray(raw,float);amps=np.asarray(amplitudes,float)
    if len(raw)==0:return []
    xy=transform(raw,H);anchor=int(np.argmax(amps))
    relative=xy-xy[anchor];radius=np.linalg.norm(relative,axis=1)
    pair=np.linalg.norm(xy[:,None]-xy,axis=2);np.fill_diagonal(pair,np.inf)
    spacing=float(np.median(pair.min(axis=1))) if len(raw)>1 else 1.
    eligible=np.ones(len(raw),bool) if eligible is None else np.asarray(eligible,bool)
    eligible=eligible & np.isfinite(amps)&(amps>0);eligible[anchor]=False
    predictions=[]
    for i in range(len(raw)):
        result=dict(method='LEAVE_ONE_OUT_DIRECTIONAL_REGION_LOG_MEDIAN',
            status='INSUFFICIENT_COMPARABLE_OBSERVATIONS',radius_mm=float(radius[i]),
            observed_contrast_DN=float(amps[i]),interpretation='regional image evidence; not an order-intensity law')
        if i==anchor:result['status']='CENTRAL_REFERENCE_NOT_PREDICTED';predictions.append(result);continue
        if not np.isfinite(amps[i]) or amps[i]<=0:
            result['status']='NONPOSITIVE_MEASUREMENT_NOT_LOG_PREDICTED';predictions.append(result);continue
        use=eligible.copy();use[i]=False
        cosine=np.divide(relative@relative[i],radius*max(radius[i],1e-9),out=np.zeros(len(raw)),where=radius>0)
        use &= (cosine>=np.cos(np.pi/6)) & (radius>=.7*radius[i]) & (radius<=radius[i]+max(1.5*spacing,2.))
        j=np.flatnonzero(use);result['peer_count']=len(j)
        result['peer_raw_xy']=raw[j].tolist()
        if len(j)>=3:
            logs=np.log(amps[j]);median=float(np.median(logs));scatter=float(1.4826*np.median(abs(logs-median)))
            width=max(np.log(8.),3*scatter);expected=float(np.exp(median));upper=float(np.exp(median+width))
            result.update(predicted_contrast_DN=expected,lower_contrast_DN=float(np.exp(median-width)),
                upper_contrast_DN=upper,log_MAD=scatter,peer_contrasts_DN=amps[j].tolist(),
                observed_to_predicted_ratio=float(amps[i]/expected),
                status='REGIONAL_BRIGHTNESS_OUTLIER' if amps[i]>upper else 'WITHIN_BROAD_REGIONAL_RANGE')
            opposite=np.linalg.norm(relative+relative[i],axis=1);opposite[~eligible]=np.inf;opposite[i]=np.inf;opposite[anchor]=np.inf
            k=int(np.argmin(opposite))
            if (opposite[k]<=max(.3*spacing,1.) and eligible[k]
                    and 1/16<=amps[k]/amps[i]<=16):
                result['opposite_measured_raw_xy']=raw[k].tolist()
                result['opposite_position_residual_mm']=float(opposite[k])
                if result['status']=='REGIONAL_BRIGHTNESS_OUTLIER':result['status']='BRIGHT_REGION_WITH_OPPOSITE_MEASURED_SUPPORT'
            if bilateral_support and result['status']=='REGIONAL_BRIGHTNESS_OUTLIER':
                # A different optical family can be brighter than the nearby
                # small lattice. Permit modest bending of opposite branches;
                # one isotropic distance bound otherwise mislabels real rows.
                asymmetry=abs(radius-radius[i])/np.maximum((radius+radius[i])/2,1e-9)
                opposite_cosine=np.divide(-relative@relative[i],radius*max(radius[i],1e-9),
                    out=np.zeros(len(raw)),where=radius>0)
                ratio=amps/max(amps[i],1e-9)
                support=eligible & (np.arange(len(raw))!=i) & (asymmetry<=.1)
                support &= (opposite_cosine>=np.cos(np.pi/18)) & (ratio>=1/16) & (ratio<=16)
                candidates=np.flatnonzero(support)
                if len(candidates):
                    k=int(candidates[np.argmin(asymmetry[candidates]+1-opposite_cosine[candidates])])
                    result.update(status='BRIGHT_REGION_WITH_BILATERAL_POSITIONAL_SUPPORT',
                        bilateral_support_raw_xy=raw[k].tolist(),bilateral_radius_asymmetry=float(asymmetry[k]),
                        bilateral_opposite_angle_deviation_deg=float(np.degrees(np.arccos(np.clip(opposite_cosine[k],-1,1)))),
                        bilateral_contrast_ratio=float(ratio[k]),
                        support_interpretation='measured opposite branch; regional brightness prior may mix optical families, not proof of diffraction identity')
        predictions.append(result)
    return predictions
