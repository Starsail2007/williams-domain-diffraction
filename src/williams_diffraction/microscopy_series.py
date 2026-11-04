"""Array-only descriptive texture families for a complete video series.

These labels describe image spectra, not director phases or optical powers.
The period/quality gates must be recorded by the calling study.
"""
import numpy as np
from .microscopy import spectrum_peaks,detrend_plane,axial_frequency_distance


def acute_axis_angle(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)),0,1))))


def texture_measurement(image,sx_um,sy_um,*,minimum_std=1.,minimum_band_fraction=.3,
                        minimum_peak_ratio=50.,family_period_band_um=(8.,35.),minimum_base_power_fraction=.12):
    result,power,fx,fy=spectrum_peaks(image,sx_um,sy_um,max_peaks=16)
    a=image[:,:,1] if image.ndim==3 else image
    residual=detrend_plane(a);std=float(np.std(residual))
    xx,yy=np.meshgrid(fx,fy);radius=np.hypot(xx,yy)
    band=(radius>=1/80)&(radius<=1/4)
    fraction=float(power[band].sum()/max(power.sum(),1e-30))
    peaks=result['peaks'];strongest=peaks[0] if peaks else None
    valid=bool(strongest and std>=minimum_std and fraction>=minimum_band_fraction and strongest['power_to_band_median']>=minimum_peak_ratio)
    fields={'status':'RESOLVED_CANDIDATES' if valid else 'LOW_OR_NONPERIODIC','texture_std_green':std,
            'band_power_fraction':fraction,'strongest_period_um':strongest['period_um'] if strongest else None,
            'strongest_normal_deg':strongest['normal_angle_deg'] if strongest else None,'morphology':'unresolved',
            'long_period_um':None,'long_normal_deg':None,'short_period_um':None,'harmonic_visible':False,
            'pair_period_1_um':None,'pair_period_2_um':None,'pair_angle_deg':None,'pair_length_ratio':None,
            'selected_ranks':[]}
    if not valid:return fields,peaks,np.zeros(128),np.zeros(90)
    families=[]
    for peak in peaks:
        if not family_period_band_um[0]<=peak['period_um']<=family_period_band_um[1]:continue
        if peak['power']<minimum_base_power_fraction*strongest['power']:continue
        q=np.array([peak['fx_cycles_um'],peak['fy_cycles_um']]);members=[]
        for order in (1,2,3):
            candidates=[p for p in peaks if axial_frequency_distance([p['fx_cycles_um'],p['fy_cycles_um']],order*q)/(order*np.linalg.norm(q))<.10]
            if candidates:members.append((order,max(candidates,key=lambda p:p['power'])))
        families.append({'base':peak,'members':members,'score':sum(p['power'] for _,p in members)})
    families.sort(key=lambda f:f['score'],reverse=True)
    if families:
        best=families[0];base=best['base'];fields.update(long_period_um=base['period_um'],long_normal_deg=base['normal_angle_deg'],
            harmonic_visible=any(n>1 for n,p in best['members']),selected_ranks=[base['rank']])
        higher=[p for n,p in best['members'] if n==2]
        if higher:fields['short_period_um']=higher[0]['period_um']
        angle=np.degrees(np.arctan2(yy,xx))%180
        delta=abs(angle-base['normal_angle_deg']);delta=np.minimum(delta,180-delta)
        axial_fraction=float(power[band&(delta<=12)].sum()/max(power[band].sum(),1e-30))
        fields['dominant_axis_fraction']=axial_fraction
        fields['morphology']='stripe' if axial_fraction>=.62 else 'mixed'
        # Pick low-frequency non-collinear candidates without an equal-length constraint.
        bases=[f['base'] for f in families if f['base']['power']>=.20*strongest['power']]
        pairs=[]
        for i,p in enumerate(bases):
            for r in bases[i+1:]:
                qa=[p['fx_cycles_um'],p['fy_cycles_um']];qb=[r['fx_cycles_um'],r['fy_cycles_um']]
                theta=acute_axis_angle(qa,qb)
                if theta>=35:
                    pairs.append((p['period_um']*r['period_um'],p,r,theta))
        if pairs and axial_fraction<.8:
            _,p,r,theta=max(pairs,key=lambda x:x[0])
            fields.update(morphology='dual_direction',pair_period_1_um=p['period_um'],pair_period_2_um=r['period_um'],
                          pair_angle_deg=theta,pair_length_ratio=max(p['period_um'],r['period_um'])/min(p['period_um'],r['period_um']),
                          pair_normal_1_deg=p['normal_angle_deg'],pair_normal_2_deg=r['normal_angle_deg'],selected_ranks=[p['rank'],r['rank']])
    radial,_=np.histogram(radius[band],bins=np.linspace(1/80,1/4,129),weights=power[band])
    angles=np.degrees(np.arctan2(yy,xx))%180
    angular,_=np.histogram(angles[band],bins=np.linspace(0,180,91),weights=power[band])
    return fields,peaks,radial/max(radial.sum(),1e-30),angular/max(angular.sum(),1e-30)
