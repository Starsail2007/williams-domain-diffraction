"""The established bright-node voltage closure, with explicit experiment windows.

Phase uses equal-frame normalized residuals and a profiled training-frame scale.
The fixed-six signal law supplies the voltage-only scale at evaluation time.
"""
import numpy as np
from scipy.optimize import least_squares
from .diffraction_models import weights, profile_scale

PAIRS={'A':{(0,1),(0,-1)},'B':{(1,0),(-1,0)},'C':{(-1,1),(1,-1)}}
SIX=set().union(*PAIRS.values())


def phase_at(parameters, voltage, window):
    low,high=window
    if high<=low:raise ValueError('Voltage window must increase')
    p=np.asarray(parameters,float);z=(np.asarray(voltage)-low)/(high-low)
    return np.array([p[0]*(1-z)+p[1]*z,p[2]*(1-z)+p[3]*z])


def fit_signal_law(groups, family='linear', pivot=6.):
    """Fit fixed-six camera signal only; never use a held frame as an anchor."""
    if family not in ('linear','log_quadratic'):raise ValueError('Unknown signal law')
    degree=1 if family=='linear' else 2
    if len({g['U'] for g in groups})<degree+1:raise ValueError('Insufficient distinct training voltages')
    q=np.array([g['Q6'] for g in groups],float)
    if not np.isfinite(q).all() or np.any(q<=0):raise ValueError('Positive finite Q6 required')
    x=np.array([g['U']-pivot for g in groups]);design=np.vander(x,degree+1,increasing=True)
    return np.linalg.lstsq(design,np.log(q) if family=='log_quadratic' else q,rcond=None)[0]


def signal_at(law, voltage, family='linear', pivot=6.):
    if family not in ('linear','log_quadratic'):raise ValueError('Unknown signal law')
    value=np.polynomial.polynomial.polyval(np.asarray(voltage)-pivot,law)
    value=np.exp(value) if family=='log_quadratic' else value
    if not np.isfinite(value).all() or np.any(value<=0):raise ValueError('Nonpositive predicted fixed-six scale')
    return value


def fit_voltage_spectrum(groups, window, pivot=6.):
    if len(groups)<3 or len({g['U'] for g in groups})<3:raise ValueError('At least three distinct voltages required')
    if any(not np.isfinite(g['Q6']) or g['Q6']<=0 for g in groups):raise ValueError('Complete positive fixed-six measurements required')
    def residual(p):
        terms=[]
        for g in groups:
            q=np.asarray(g['q']);w=weights(g['labels'],phase_at(p,g['U'],window),'bessel_low')
            terms.extend((profile_scale(q,w)*w-q)/np.linalg.norm(q))
        return np.array(terms)
    attempts=[];good=[]
    for start in [[1.3,1.3,1.5,1.7],[2.5]*4,[.7,1.5,1.,2.]]:
        fit=least_squares(residual,start,bounds=([.05]*4,[3.75]*4),max_nfev=500,
                          ftol=1e-11,xtol=1e-11,gtol=1e-11)
        attempts.append({'start':start,'success':bool(fit.success),'RSS':float(fit.fun@fit.fun),'endpoints':fit.x.tolist()})
        if fit.success:good.append(fit)
    if not good:raise RuntimeError('Shared bright spectrum fit failed')
    p=min(good,key=lambda r:r.fun@r.fun).x
    law=fit_signal_law(groups,pivot=pivot)
    return p,law,attempts


def predict_voltage_spectrum(labels, voltage, parameters, law, window, pivot=6., family='linear'):
    beta=phase_at(parameters,voltage,window);q6=float(signal_at(law,voltage,family,pivot))
    w6=float(weights(np.array(sorted(SIX)),beta,'bessel_low').sum())
    if q6<=0 or w6<=0:raise ValueError('Nonpositive predicted fixed-six scale')
    return q6/w6*weights(np.asarray(labels),beta,'bessel_low')
