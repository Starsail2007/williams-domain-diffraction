"""Batch-independent effective optical models, retaining the established definitions.

Geometry uses direction cosines, integer labels remain conditional optical labels.
Signal fits use signed CFA integrals, not radiometric power.
Extracted unchanged from the active 2026-10-12/13 studies; frozen runs stay intact.
"""
import itertools
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.special import jv
from .runtime_trace import enabled as trace_enabled, log_values


def fit_geometry(labels, positions, model):
    labels, positions = np.asarray(labels, float), np.asarray(positions, float)
    X = np.c_[np.ones(len(labels)), labels]
    if np.linalg.matrix_rank(X) < 3:
        raise ValueError('Geometry not identified')
    affine = np.linalg.lstsq(X, positions, rcond=None)[0]
    log_values('geometry_inputs',model=model,labels=labels,positions=positions,design_matrix=X,affine_start=affine)
    if model == 'affine':
        return affine, X @ affine
    h, v = np.linalg.norm(affine[1]), np.linalg.norm(affine[2])
    angle = math.atan2(affine[1, 1], affine[1, 0])
    if model == 'square':
        start = [*affine[0], math.log((h + v) / 2), angle]
    else:
        start = [*affine[0], math.log(h), math.log(v), angle]
    def make(p):
        h = math.exp(p[2]); v = h if model == 'square' else math.exp(p[3]); angle = p[-1]
        return np.array([p[:2], [h * math.cos(angle), h * math.sin(angle)], [-v * math.sin(angle), v * math.cos(angle)]])
    result = least_squares(lambda p: (X @ make(p) - positions).ravel(), start, max_nfev=300,
                           ftol=1e-11, xtol=1e-11, gtol=1e-11,verbose=2 if trace_enabled() else 0)
    if not result.success:
        raise RuntimeError('Geometry fit failed: ' + model)
    coeff = make(result.x)
    log_values('geometry_solution',parameters=result.x,coefficients=coeff,residual=result.fun,nfev=result.nfev,cost=result.cost,optimality=result.optimality)
    return coeff, X @ coeff


def metric(q, pred):
    q, pred = np.asarray(q), np.asarray(pred)
    error = q - pred
    denominator = float(q @ q)
    if denominator <= 0:
        raise ValueError('No measurable signal norm')
    variance = float(np.sum((q - q.mean()) ** 2))
    positive = (q > 0) & (pred > 0)
    logerr = np.log(q[positive] / pred[positive])
    return {'NRMSE_L2': float(np.linalg.norm(error) / np.linalg.norm(q)),
        'R2': float(1 - error @ error / variance) if variance > 0 else None,
        'positive_log_RMSE': float(np.sqrt(np.mean(logerr ** 2))) if positive.any() else None,
        'n_log_cells': int(positive.sum()), 'n_positive_invalid_prediction': int(((q > 0) & (pred <= 0)).sum()),
        'positive_median_factor_error': float(np.exp(np.median(np.abs(logerr)))) if positive.any() else None}


def phase_weights(labels, beta1, beta2, gamma, size=64):
    theta = 2 * np.pi * np.arange(size) / size
    u, v = theta[:, None], theta[None, :]
    phase = beta1 * np.cos(u) + beta2 * np.cos(v) + gamma * np.cos(u + v)
    C = np.fft.fft2(np.exp(1j * phase)) / size ** 2
    labels = np.asarray(labels, int)
    return np.abs(C[labels[:, 0] % size, labels[:, 1] % size]) ** 2


def weights(labels, p, model):
    a, b = np.asarray(labels).T
    if model == 'gaussian_axis':
        return np.exp(-p[0] * a * a - p[1] * b * b)
    if model == 'gaussian_rotated':
        return np.exp(-p[0] * a * a - p[1] * b * b - 2 * p[2] * np.sqrt(p[0] * p[1]) * a * b)
    if model.startswith('bessel'):
        return jv(a, p[0]) ** 2 * jv(b, p[1]) ** 2
    return phase_weights(labels, *p)


def profile_scale(q, w):
    norm = float(w @ w)
    if norm <= 0:
        raise ValueError('Zero predicted weights')
    return max(0., float(q @ w / norm))


def spectrum_spec(model):
    if model == 'gaussian_axis':
        return [1e-4, 1e-4], [8., 8.], [[x, y] for x, y in ((.3, .3), (.8, .8), (1.5, .5), (.5, 1.5))]
    if model == 'gaussian_rotated':
        return [1e-4, 1e-4, -.95], [8., 8., .95], [[x, y, r] for x, y, r in ((.5, .5, 0), (1, 1, 0), (1, .5, .4), (.5, 1, -.4))]
    if model == 'phase_sum':
        return [.05, .05, 0.], [3.75, 3.75, 4.], [[1.3, 1.6, x] for x in (0., .2, .8, 1.6)] + [[2.5, 2.5, .5]]
    bound = 3.75 if model == 'bessel_low' else 12.
    seeds = [.7, 1.3, 2.5, 3.4] if model == 'bessel_low' else [1.3, 2.5, 4.8, 6.8, 9.5, 11.]
    return [.05, .05], [bound, bound], [list(x) for x in itertools.product(seeds, repeat=2)]


def fit_spectrum(labels, q, model, seeds=None):
    lower, upper, defaults = spectrum_spec(model)
    q = np.asarray(q, float); norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError('No response norm')
    attempts, good = [], []
    for seed in defaults if seeds is None else seeds:
        start = np.clip(seed, np.array(lower) + 1e-8, np.array(upper) - 1e-8)
        log_values('spectrum_start',model=model,labels=np.asarray(labels),observed=q,norm=norm,start=start,lower=lower,upper=upper)
        evaluations=0
        def residual(p):
            nonlocal evaluations
            evaluations+=1
            w = weights(labels, p, model)
            values=(profile_scale(q, w) * w - q) / norm
            if trace_enabled() and (evaluations<=2 or evaluations%20==0):
                log_values('spectrum_residual_evaluation',evaluation=evaluations,parameters=p,weights=w,residual=values)
            return values
        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=300,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9,verbose=2 if trace_enabled() else 0)
        log_values('spectrum_attempt',parameters=result.x,residual=result.fun,nfev=result.nfev,cost=result.cost,success=bool(result.success),optimality=result.optimality)
        attempts.append({'start': start.tolist(), 'success': bool(result.success), 'nfev': result.nfev,
                         'RSS_normalized': float(result.fun @ result.fun), 'p': result.x.tolist()})
        if result.success:
            good.append(result)
    if not good:
        raise RuntimeError('All declared starts failed: ' + model)
    best = min(good, key=lambda r: r.fun @ r.fun)
    w = weights(labels, best.x, model); scale = profile_scale(q, w)
    log_values('spectrum_best',parameters=best.x,scale=scale,weights=w,predicted=scale*w,observed=q)
    return best.x, scale, scale * w, attempts


def inversion_key(label):
    a, b = map(int, label)
    return min((a, b), (-a, -b))


def one_d_prediction(p, z):
    beta = p[0] * (1 - z) + p[1] * z
    u = np.arange(512) * 2 * np.pi / 512
    field = np.exp(1j * (beta[:, None] * np.cos(u) + p[2] * beta[:, None] ** 2 * np.cos(2 * u)))
    w = np.abs(np.fft.fft(field, axis=1)[:, 1:4] / 512) ** 2
    return np.log(np.maximum(w[:, 1:], 1e-24) / np.maximum(w[:, :1], 1e-24))


def fit_one_d(z, y, single=False):
    def pred(p, zz):
        if not single:
            return one_d_prediction(p, zz)
        w = jv(np.arange(1, 4)[None, :], (p[0] * (1 - zz) + p[1] * zz)[:, None]) ** 2
        return np.log(np.maximum(w[:, 1:], 1e-24) / np.maximum(w[:, :1], 1e-24))
    starts = [(a, b) if single else (a, b, c) for a, b in ((1.5, 2.5), (2.5, 1.5), (2, 2)) for c in ((0,) if single else (.1, .3, .6, 1.2))]
    lower, upper = ([.05, .05], [3.75, 3.75]) if single else ([.05, .05, 0], [3.75, 3.75, 2])
    good = []
    for start in starts:
        log_values('one_d_start',single_sine=single,z=np.asarray(z),observed_log_ratios=np.asarray(y),start=start,lower=lower,upper=upper)
        evaluations=0
        def residual(p):
            nonlocal evaluations
            evaluations+=1
            values=(pred(p,z)-y).ravel()
            if trace_enabled() and (evaluations<=2 or evaluations%20==0):
                log_values('one_d_residual_evaluation',evaluation=evaluations,parameters=p,residual=values)
            return values
        r = least_squares(residual, start, bounds=(lower, upper), max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10,verbose=2 if trace_enabled() else 0)
        log_values('one_d_attempt',parameters=r.x,residual=r.fun,nfev=r.nfev,cost=r.cost,success=bool(r.success),optimality=r.optimality)
        if r.success:
            good.append(r)
    if not good:
        raise RuntimeError('All 1D starts failed')
    best = min(good, key=lambda r: r.fun @ r.fun)
    log_values('one_d_best',single_sine=single,parameters=best.x,predicted_log_ratios=pred(best.x,z),cost=best.cost,nfev=best.nfev)
    return best.x, pred(best.x, z), pred


def project(raw, H):
    points = np.atleast_2d(raw)
    p = np.c_[points, np.ones(len(points))] @ H.T
    return p[:, :2] / p[:, 2:]


def lab_change(B, origin, H):
    center = project(origin, H)[0]
    choices = []
    for i, j in itertools.product(range(-3, 4), repeat=2):
        if (i, j) == (0, 0) or math.gcd(abs(i), abs(j)) != 1:
            continue
        vec = project(origin + B @ [i, j], H)[0] - center
        choices.append((np.linalg.norm(vec), np.array([i, j]), vec))
    h = min((v for v in choices if v[2][0] > 0 and abs(v[2][1]) / v[0] < .25), key=lambda v: v[0])
    v = min((v for v in choices if v[2][1] > 0 and abs(v[2][0]) / v[0] < .25), key=lambda v: v[0])
    change = np.column_stack((h[1], v[1]))
    if abs(round(np.linalg.det(change))) != 2:
        raise ValueError('Unexpected supercell; do not impose parity labels')
    return change
