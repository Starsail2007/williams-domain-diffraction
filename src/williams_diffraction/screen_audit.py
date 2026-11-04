"""Evidence-gated screen refinement. Display pixels, never diffraction peaks.

Unlike the legacy dark-inward fitter, side edges can have either polarity.
Unsupported edges retain their explicitly reviewed prior and are marked as such.
"""
import numpy as np
from scipy import ndimage as ndi
import cv2

from .screen_validation_trial import fit_line, intersection


def fit_screen_edge(gray, a, b, center, side, half_width=15.):
    a, b = np.asarray(a, float), np.asarray(b, float)
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])
    if normal @ (center - (a + b) / 2) < 0:
        normal = -normal
    offsets = np.arange(-half_width, half_width + .25, .25)
    samples = []
    for i, t in enumerate(np.linspace(.06, .94, 89)):
        if side == 2 and .3 < t < .7:
            continue  # The hand occludes the middle; both ends must still support a fit.
        p = (1 - t) * a + t * b
        loc = p + offsets[:, None] * normal
        values = ndi.map_coordinates(gray, [loc[:, 1], loc[:, 0]], order=1, mode='nearest')
        gradient = np.gradient(ndi.gaussian_filter1d(values, 2), offsets)
        # The top has a bright physical lip: use its inward dark transition,
        # not alternating sides of the thin bright rim. Side/background polarity
        # may reverse and therefore remains bipolar.
        strength = np.maximum(-gradient, 0) if side == 0 else np.abs(gradient)
        merit = strength * np.exp(-.5 * (offsets / max(half_width, 1)) ** 2)
        k = int(merit.argmax())
        valid = 5 < k < len(offsets) - 6 and strength[k] >= .8
        samples.append(dict(t=float(t), x=float(loc[k, 0]), y=float(loc[k, 1]),
                            offset_px=float(offsets[k]), gradient=float(gradient[k]),
                            eligible=bool(valid), split='held_out' if i % 3 == 1 else 'train'))
    train = [s for s in samples if s['eligible'] and s['split'] == 'train']
    held = [s for s in samples if s['eligible'] and s['split'] == 'held_out']
    record = dict(side=side, status='REVIEWED_PRIOR_INSUFFICIENT_EDGE_SUPPORT',
                  samples=samples, supported=False, independent_metric_validation=False)
    prior = ((a + b) / 2, normal)
    if len(train) < 8 or len(held) < 5:
        return *prior, record
    points = np.array([[s['x'], s['y']] for s in train])
    # A deterministic consensus rejects crossing background edges and hand contours.
    best = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            v = points[j] - points[i]
            length = np.linalg.norm(v)
            if length < .3 * np.linalg.norm(b - a):
                continue
            v /= length
            if abs(v @ tangent) < .998:
                continue
            n = np.array([-v[1], v[0]])
            keep = np.abs((points - points[i]) @ n) < 1.8
            score = (int(keep.sum()), -float(np.median(np.abs((points[keep] - points[i]) @ n))))
            if best is None or score > best[0]:
                best = score, keep
    if best is None or best[1].sum() < 8:
        return *prior, record
    try:
        c, n, keep = fit_line(points[best[1]])
    except ValueError as error:
        record.update(status='REVIEWED_PRIOR_UNSTABLE_EDGE_FIT', fit_failure=str(error))
        return *prior, record
    selected = np.flatnonzero(best[1])[keep]
    ts = np.array([train[i]['t'] for i in selected])
    hp = np.array([[s['x'], s['y']] for s in held])
    hr = np.abs((hp - c) @ n)
    hkeep = hr < 2.5
    ht = np.array([s['t'] for s in held])[hkeep]
    count = len(selected) + int(hkeep.sum())
    coverage = float(np.ptp(ts))
    residual = float(np.percentile(hr[hkeep], 95)) if hkeep.any() else None
    record.update(accepted_train=len(selected), accepted_held_out=int(hkeep.sum()),
                  sampled_held_out=len(held), held_out_inlier_fraction=float(hkeep.mean()),
                  held_out_p95_display_px=residual, supported_t_span=coverage,
                  supported_t_min=float(ts.min()), supported_t_max=float(ts.max()),
                  endpoint_displacement_px=[float(abs((q-c)@n)) for q in (a,b)])
    required = 16 if side == 2 else 35
    valid = (count >= required and coverage >= .65 and ts.min() <= .25 and ts.max() >= .75
             and len(ht) >= 5 and ht.min() <= .3 and ht.max() >= .7
             and hkeep.mean() >= .7 and residual <= 1.8
             and max(record['endpoint_displacement_px']) <= half_width)
    if not valid:
        return *prior, record
    record.update(status='BIPOLAR_EDGE_WITH_HELD_OUT_SUPPORT', supported=True)
    return c, n, record


def fit_screen_edge_multipeak(gray, a, b, center, side, half_width=15.):
    """Fit a common physical edge from several gradients per held-out profile.

    Tangential averaging reduces sensor noise. Consensus chooses one gradient
    per cross-section, allowing the background polarity to change and rejecting
    the curved outline of the hand. No diffraction positions enter this fit.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    tangent = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-tangent[1], tangent[0]])
    if normal @ (center-(a+b)/2) < 0: normal = -normal
    midpoint = (a+b)/2
    offsets = np.arange(-half_width, half_width+.25, .25)
    profiles = []
    for i,t in enumerate(np.linspace(.04,.96,101)):
        p = (1-t)*a+t*b
        locations = p+offsets[:,None]*normal
        values = np.mean([ndi.map_coordinates(gray,
            [(locations+dt*tangent)[:,1], (locations+dt*tangent)[:,0]],
            order=1,mode='nearest') for dt in (-3.,-1.5,0.,1.5,3.)],axis=0)
        gradient = np.gradient(ndi.gaussian_filter1d(values,1.5),offsets)
        strength = np.maximum(-gradient,0) if side==0 else np.abs(gradient)
        maxima = np.flatnonzero((strength==ndi.maximum_filter1d(strength,9)) & (strength>=.45))
        maxima = [int(k) for k in maxima if 5<k<len(offsets)-6]
        maxima.sort(key=lambda k:-strength[k])
        chosen=[]
        for k in maxima:
            if all(abs(offsets[k]-offsets[j])>=2 for j in chosen): chosen.append(k)
            if len(chosen)==4: break
        profiles.append(dict(t=float(t),split='held_out' if i%3==1 else 'train',
            candidates=[dict(x=float(locations[k,0]),y=float(locations[k,1]),
                offset_px=float(offsets[k]),gradient=float(gradient[k])) for k in chosen]))
    prior = (midpoint,normal)
    record=dict(side=side,status='REVIEWED_PRIOR_INSUFFICIENT_EDGE_SUPPORT',
        samples=profiles,supported=False,independent_metric_validation=False,
        method='TANGENTIAL_AVERAGE_MULTIPLE_GRADIENT_CONSENSUS')
    train=[s for s in profiles if s['split']=='train' and s['candidates']]
    held=[s for s in profiles if s['split']=='held_out' and s['candidates']]
    if len(train)<12 or len(held)<6: return *prior,record
    points=np.array([[v['x'],v['y']] for s in train for v in s['candidates']])
    weights=np.array([min(abs(v['gradient']),3.) for s in train for v in s['candidates']])
    seeds=[]
    # Bounded Hough search; only training profiles vote for the line.
    for angle in np.linspace(-.04,.04,65):
        n=normal*np.cos(angle)+tangent*np.sin(angle)
        intercepts=(points-midpoint)@n
        bins=np.arange(-half_width-2,half_width+2.5,.5)
        hist,_=np.histogram(intercepts,bins=bins,weights=weights)
        votes=ndi.convolve1d(hist,np.ones(5),mode='constant')
        for k in np.argsort(-votes)[:2]:
            seeds.append((float(votes[k]),midpoint+n*(bins[k]+.25),n))
    def nearest(profiles,c,n):
        selected=[]
        for s in profiles:
            candidate=min(s['candidates'],key=lambda v:abs((np.array([v['x'],v['y']])-c)@n))
            residual=float(abs((np.array([candidate['x'],candidate['y']])-c)@n))
            selected.append({**s,**candidate,'residual_px':residual})
        return selected
    best=None
    for _,c,n in sorted(seeds,key=lambda s:-s[0])[:32]:
        selected=nearest(train,c,n)
        good=[s for s in selected if s['residual_px']<1.6]
        if len(good)<12: continue
        for _ in range(2):
            q=np.array([[s['x'],s['y']] for s in good])
            c=q.mean(0);_,_,vt=np.linalg.svd(q-c,full_matrices=False);n=vt[-1]
            selected=nearest(train,c,n);good=[s for s in selected if s['residual_px']<1.6]
            if len(good)<12:break
        if len(good)<12:continue
        ts=np.array([s['t'] for s in good])
        displacement=max(abs((p-c)@n) for p in (a,b))
        if displacement>half_width:continue
        score=(len(good),float(np.ptp(ts)),-float(np.median([s['residual_px'] for s in good])),
               -float(abs((midpoint-c)@n)))
        if best is None or score>best[0]:best=(score,c,n,good)
    if best is None:return *prior,record
    _,c,n,good=best
    validation=nearest(held,c,n);hgood=[s for s in validation if s['residual_px']<2.5]
    ts=np.array([s['t'] for s in good]);ht=np.array([s['t'] for s in hgood])
    p95=float(np.percentile([s['residual_px'] for s in hgood],95)) if hgood else None
    fraction=len(hgood)/len(held)
    endpoint_support=[]
    for lower,upper in ((.04,.25),(.75,.96)):
        selected=[s for s in validation if lower<=s['t']<=upper]
        eligible=[s for s in profiles if s['split']=='held_out' and lower<=s['t']<=upper]
        inliers=sum(s['residual_px']<2.5 for s in selected)
        residuals=[s['residual_px'] for s in selected if s['residual_px']<2.5]
        endpoint_support.append(dict(sampled=len(eligible),accepted=inliers,
            inlier_fraction=inliers/len(eligible) if eligible else 0.,
            p95_display_px=float(np.percentile(residuals,95)) if residuals else None))
    record['held_out_endpoint_support']=endpoint_support
    ends_supported=all(e['accepted']>=5 and e['inlier_fraction']>=.7
                       and e['p95_display_px']<=1.8 for e in endpoint_support)
    record.update(accepted_train=len(good),accepted_held_out=len(hgood),sampled_held_out=len(held),
        held_out_inlier_fraction=fraction,held_out_p95_display_px=p95,
        supported_t_span=float(np.ptp(ts)),supported_t_min=float(ts.min()),supported_t_max=float(ts.max()),
        endpoint_displacement_px=[float(abs((p-c)@n)) for p in (a,b)],
        selected_train=good,selected_held_out=validation)
    # The bottom's middle is predeclared as occluded, never evidence for the
    # physical boundary. Validate both visible endpoint strips independently;
    # retain the all-profile residual above as an occlusion diagnostic.
    record['validation_domain']=('BOTH_PREDECLARED_VISIBLE_ENDPOINT_STRIPS' if side==2
                                  else 'FULL_EDGE_WITH_BOTH_ENDPOINT_STRIPS')
    valid=(len(good)+len(hgood)>=(24 if side==2 else 45) and np.ptp(ts)>=.7
        and ts.min()<=.22 and ts.max()>=.78 and len(ht)>=8 and ht.min()<=.25 and ht.max()>=.75
        and ends_supported and (side==2 or (fraction>=.7 and p95<=1.8)))
    if not valid:return *prior,record
    record.update(status='MULTIPEAK_EDGE_WITH_HELD_OUT_SUPPORT',supported=True)
    return c,n,record


def refine_screen(gray, seed, half_widths=(15., 15., 8., 15.), method='legacy',
                  edge_reviews=None,corner_constraints=None):
    seed = np.asarray(seed, float)
    gray = ndi.gaussian_filter(np.asarray(gray, float), 1.)
    lines, diagnostics = [], []
    for side in range(4):
        fitter=fit_screen_edge_multipeak if method=='multipeak' else fit_screen_edge
        c, n, record = fitter(gray, seed[side], seed[(side+1)%4],
                                       seed.mean(0), side, half_widths[side])
        record['gradient_fit_supported']=record['supported']
        if edge_reviews is not None:
            review=edge_reviews[side]
            if review['side']!=side:raise ValueError('Physical edge review order differs')
            record['physical_visibility_review']=review
            endpoints=review['endpoint_visibility']
            visible=(review['physical_edge_identity_visible'] is True
                     and len(endpoints)==2 and all(v is True for v in endpoints.values()))
            record['physical_endpoints_visible']=visible
            if not visible or review.get('coordinate_repair_required',False):
                record.update(supported=False,status='CONDITIONAL_PHYSICAL_ENDPOINT_GEOMETRY')
            if review['physical_edge_identity_visible'] is False:
                tangent=seed[(side+1)%4]-seed[side]
                c=(seed[side]+seed[(side+1)%4])/2
                n=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
                record['geometry_source']='REVIEWED_PRIOR_FALSE_EDGE_IDENTITY'
            else:record['geometry_source']='GRADIENT_FIT_OR_REVIEWED_PRIOR_WITH_PHYSICAL_VISIBILITY_REVIEW'
        lines.append((c, n)); diagnostics.append(record)
    corners = np.array([intersection(lines[3], lines[0]), intersection(lines[0], lines[1]),
                        intersection(lines[1], lines[2]), intersection(lines[2], lines[3])])
    for label,constraint in (corner_constraints or {}).items():
        index=['TL','TR','BR','BL'].index(label)
        interval=constraint['proposed_display_xy_interval']
        xy=corners[index]
        inside=interval['x'][0]<=xy[0]<=interval['x'][1] and interval['y'][0]<=xy[1]<=interval['y'][1]
        proposed=xy.copy()
        if not inside:corners[index]=seed[index]
        for side in ((index-1)%4,index):
            diagnostics[side].update(supported=False,status='CONDITIONAL_REVIEWED_VISIBLE_CORNER',
                geometry_source='REVIEWED_VISIBLE_CORNER_AND_CONDITIONAL_OTHER_ENDPOINT')
            diagnostics[side].setdefault('visible_corner_constraints',[]).append(dict(
                corner=label,automatic_proposal_display_xy=proposed.tolist(),
                automatic_inside_review_interval=bool(inside),selected_display_xy=corners[index].tolist(),
                constraint=constraint))
    for side,record in enumerate(diagnostics):
        record['selected_geometry_endpoints_display']=[corners[side].tolist(),corners[(side+1)%4].tolist()]
    if (not cv2.isContourConvex(corners.astype(np.float32))
            or cv2.contourArea(corners.astype(np.float32), oriented=True) <= 0
            or np.linalg.norm(corners-seed, axis=1).max() > 25):
        raise ValueError('Refined screen escaped its reviewed corner prior')
    return corners, diagnostics
