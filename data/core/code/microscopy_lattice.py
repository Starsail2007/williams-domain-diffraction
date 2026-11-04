"""Unconstrained reciprocal-basis candidates for image texture spectra."""
import numpy as np

def select_texture_family(peaks,minimum_base_power_fraction=.25,period_band=(8.,35.)):
    """A supported image-frequency family, never a certified physical base."""
    if not peaks:return None
    maximum=max(p['power'] for p in peaks);families=[]
    for base in peaks:
        if not period_band[0]<=base['period_um']<=period_band[1] or base['power']<maximum*minimum_base_power_fraction:continue
        q=np.array([base['fx_cycles_um'],base['fy_cycles_um']]);members=[]
        for n in (1,2,3):
            matches=[]
            for p in peaks:
                v=np.array([p['fx_cycles_um'],p['fy_cycles_um']]);distance=min(np.linalg.norm(v-n*q),np.linalg.norm(v+n*q))/(n*np.linalg.norm(q))
                if distance<.10:matches.append(p)
            if matches:members.append((n,max(matches,key=lambda p:p['power'])))
        families.append({'base':base,'members':members,'score':sum(p['power'] for _,p in members)})
    return max(families,key=lambda f:f['score']) if families else None

def select_texture_pair(peaks,*,period_band=(8.,35.),minimum_relative_power=.20,
                        maximum_integer=3,maximum_relative_residual=.15):
    if not peaks:return None
    maximum=max(p['power'] for p in peaks)
    bases=[p for p in peaks if period_band[0]<=p['period_um']<=period_band[1] and p['power']>=maximum*minimum_relative_power]
    candidates=[];total=sum(p['power'] for p in peaks)
    for i,a in enumerate(bases):
        for b in bases[i+1:]:
            qa=np.array([a['fx_cycles_um'],a['fy_cycles_um']]);qb=np.array([b['fx_cycles_um'],b['fy_cycles_um']])
            angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(qa,qb))/(np.linalg.norm(qa)*np.linalg.norm(qb)),0,1))))
            if angle<35:continue
            B=np.column_stack([qa,qb]);scale=min(np.linalg.norm(qa),np.linalg.norm(qb));covered=0.;members=[]
            for p in peaks:
                q=np.array([p['fx_cycles_um'],p['fy_cycles_um']]);coef=np.linalg.solve(B,q);ab=np.rint(coef).astype(int)
                error=float(np.linalg.norm(B@ab-q)/scale)
                if np.max(abs(ab))<=maximum_integer and np.any(ab) and error<=maximum_relative_residual:
                    covered+=p['power']*np.exp(-.5*(error/maximum_relative_residual)**2)
                    members.append({'rank':p['rank'],'integer_coefficients':ab.tolist(),'relative_residual':error})
            candidates.append({'first':a,'second':b,'angle_deg':angle,'coverage':covered/max(total,1e-30),
                               'basis_frequency_sum':float(np.linalg.norm(qa)+np.linalg.norm(qb)),'members':members})
    if not candidates:return None
    best_coverage=max(x['coverage'] for x in candidates)
    # Equivalent integer bases may explain the same lattice. Prefer a reduced
    # low-frequency basis only among candidates with comparable coverage.
    close=[x for x in candidates if x['coverage']>=best_coverage-.02]
    chosen=min(close,key=lambda x:x['basis_frequency_sum'])
    return chosen
