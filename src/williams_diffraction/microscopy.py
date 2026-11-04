"""Array-only microscopy texture measurements (not optical-field inversion).

Coordinates: image x right/y down; returned physical frequency x right/y up.
Frequencies are cycles/um, periods are um, axial orientations are modulo 180 deg.
Cropping never changes the original pixel scale. No imports read data or write.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, map_coordinates
from scipy.signal import fftconvolve, find_peaks
from .guide_events import emit


def pixel_scale(field_width_um: float, reference_width_px: int,
                pixel_aspect_y_over_x: float = 1.) -> tuple[float, float]:
    """Field width is edge-to-edge; square object-plane sampling is explicit."""
    if reference_width_px < 2 or not np.isfinite(field_width_um) or field_width_um <= 0:
        raise ValueError('Positive physical width and at least two reference pixels required')
    if not np.isfinite(pixel_aspect_y_over_x) or pixel_aspect_y_over_x <= 0:
        raise ValueError('Positive pixel aspect ratio required')
    sx = float(field_width_um / reference_width_px)
    return sx, sx * float(pixel_aspect_y_over_x)


def gray_array(image, channel=1):
    """Use a declared decoded color channel, without nonlinear contrast edits."""
    a = np.asarray(image, dtype=float)
    if a.ndim == 3:
        if a.shape[2] not in (3, 4) or channel not in (0, 1, 2):
            raise ValueError('Three/four-channel image and explicit RGB/BGR channel required')
        a = a[:, :, channel]
    if a.ndim != 2 or min(a.shape) < 32 or not np.isfinite(a).all():
        raise ValueError('Finite 2D image, at least 32 pixels per side, required')
    return a


def detrend_plane(a):
    a = gray_array(a)
    y = np.linspace(-1, 1, a.shape[0])[:, None]
    x = np.linspace(-1, 1, a.shape[1])[None, :]
    ax = float(np.sum(a*x) / (a.shape[0]*np.sum(x*x)))
    ay = float(np.sum(a*y) / (a.shape[1]*np.sum(y*y)))
    return a - a.mean() - ax*x - ay*y


def axial_frequency_distance(q, target):
    """Distance between unoriented +/- frequency pairs, in the input units."""
    q, target = np.asarray(q, float), np.asarray(target, float)
    if q.shape != (2,) or target.shape != (2,) or not np.isfinite([q, target]).all():
        raise ValueError('Two finite 2D frequency vectors required')
    return float(min(np.linalg.norm(q-target), np.linalg.norm(q+target)))


def _quadratic_peak(log_power, iy, ix):
    if min(iy, ix) < 1 or iy+1 >= log_power.shape[0] or ix+1 >= log_power.shape[1]:
        return 0., 0., False
    yy, xx = np.mgrid[-1:2, -1:2]
    design = np.column_stack((xx.ravel()**2, yy.ravel()**2, (xx*yy).ravel(),
                              xx.ravel(), yy.ravel(), np.ones(9)))
    p = np.linalg.lstsq(design, log_power[iy-1:iy+2, ix-1:ix+2].ravel(), rcond=None)[0]
    H = np.array([[2*p[0], p[2]], [p[2], 2*p[1]]])
    if np.linalg.eigvalsh(H).max() >= -1e-10:
        return 0., 0., False
    delta = -np.linalg.solve(H, p[3:5])
    if np.max(np.abs(delta)) > .75:
        return 0., 0., False
    return float(delta[0]), float(delta[1]), True


def spectrum_peaks(image, sx_um, sy_um, *, min_period_um=4., max_period_um=80.,
                   max_peaks=12, minimum_peak_separation_bins=3, channel=1):
    """Return multiple texture peaks and the unpadded 2D power spectrum.

    Conjugate +/- peaks count once. Harmonics and combination peaks remain;
    the function deliberately does not certify a fundamental or lattice basis.
    Quadratic localization is interpolation, not a statistical error estimate.
    """
    a = gray_array(image, channel)
    if not np.isfinite([sx_um, sy_um]).all() or min(sx_um, sy_um) <= 0:
        raise ValueError('Positive, finite x/y sampling required')
    if not 0 < min_period_um < max_period_um or max_peaks < 1:
        raise ValueError('Invalid period search band or peak count')
    ny, nx = a.shape
    window = np.outer(np.hanning(ny), np.hanning(nx))
    detrended = detrend_plane(a)
    emit('texture.detrend', array=detrended)
    windowed = detrended*window
    emit('texture.window', array=windowed)
    transformed = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(transformed)**2 / np.sum(window**2)
    fx = np.fft.fftshift(np.fft.fftfreq(nx, sx_um))
    fy_down = np.fft.fftshift(np.fft.fftfreq(ny, sy_um))
    xx, yy = np.meshgrid(fx, fy_down)
    radius = np.hypot(xx, yy)
    half_plane = (xx > 0) | ((xx == 0) & (yy < 0))
    band = (radius >= 1/max_period_um) & (radius <= 1/min_period_um)
    allowed = band & half_plane
    metadata = {'shape': [ny, nx], 'sx_um': float(sx_um), 'sy_um': float(sy_um),
                'frequency_bin_x_cycles_um': 1/(nx*sx_um),
                'frequency_bin_y_cycles_um': 1/(ny*sy_um),
                'channel': channel, 'preprocess': 'plane removal, separable Hann, no zero padding',
                'period_band_um': [min_period_um, max_period_um]}
    if not allowed.any() or float(power[band].max()) <= 1e-20:
        emit('texture.spectrum', power=power, fx=fx, fy=-fy_down, peaks=[])
        return {**metadata, 'peaks': [], 'status': 'NO_RESOLVED_TEXTURE'}, power, fx, -fy_down
    size = 2*minimum_peak_separation_bins+1
    maxima = allowed & (power == maximum_filter(power, size=size, mode='constant'))
    candidates = np.argwhere(maxima)
    candidates = sorted(candidates, key=lambda q: power[tuple(q)], reverse=True)
    median = max(float(np.median(power[band])), 1e-30)
    log_power = np.log(np.maximum(power, 1e-30))
    peaks = []
    for iy, ix in candidates[:max_peaks]:
        dx, dy, refined = _quadratic_peak(log_power, int(iy), int(ix))
        px = float(fx[ix] + dx/(nx*sx_um))
        py = -float(fy_down[iy] + dy/(ny*sy_um))
        f = float(np.hypot(px, py))
        if f == 0:
            continue
        peaks.append({'rank': len(peaks)+1, 'fx_cycles_um': px, 'fy_cycles_um': py,
                      'period_um': 1/f, 'normal_angle_deg': float(np.degrees(np.arctan2(py, px)) % 180),
                      'stripe_angle_deg': float((np.degrees(np.arctan2(py, px))+90) % 180),
                      'power': float(power[iy, ix]), 'power_to_band_median': float(power[iy, ix]/median),
                      'refined': refined, 'fft_index_x': int(ix), 'fft_index_y': int(iy),
                      'delta_bin_x': dx, 'delta_bin_y': dy})
    emit('texture.spectrum', power=power, fx=fx, fy=-fy_down, peaks=peaks)
    return {**metadata, 'peaks': peaks, 'status': 'CANDIDATES_ONLY'}, power, fx, -fy_down


def normal_profile(image, sx_um, sy_um, peak, *, channel=1, strip_width_um=0.):
    """Central normal line/strip and linear (non-wrapping) autocorrelation.

    ACF peak near the selected FFT period is a consistency check using the same
    image, not independent evidence or a guaranteed complete physical period.
    """
    a = gray_array(image, channel)
    ny, nx = a.shape
    f = np.array([peak['fx_cycles_um'], peak['fy_cycles_um']], float)
    n = f/np.linalg.norm(f)
    d_pixels = np.array([n[0]/sx_um, -n[1]/sy_um])
    center = np.array([(nx-1)/2, (ny-1)/2])
    bounds = [(center[i]-2)/abs(d_pixels[i]) for i in (0, 1) if abs(d_pixels[i]) > 1e-12]
    if not bounds or min(bounds) <= 0:
        raise ValueError('Profile cannot fit inside ROI')
    extent = min(bounds)
    step = min(sx_um, sy_um)/2
    distance = np.arange(-extent, extent+step*.5, step)
    tangent = np.array([-n[1]/sx_um, -n[0]/sy_um])
    offsets = [0.] if strip_width_um <= 0 else np.linspace(-strip_width_um/2, strip_width_um/2, 5)
    profiles = []
    for offset in offsets:
        q = center[:, None] + d_pixels[:, None]*distance + tangent[:, None]*offset
        if (q[0] < 0).any() or (q[0] > nx-1).any() or (q[1] < 0).any() or (q[1] > ny-1).any():
            continue
        profiles.append(map_coordinates(a, [q[1], q[0]], order=1, mode='nearest'))
    if not profiles:
        raise ValueError('No valid normal profiles')
    values = np.mean(profiles, axis=0)
    residual = values - np.polyval(np.polyfit(distance, values, 1), distance)
    acf = fftconvolve(residual, residual[::-1], mode='full')[len(residual)-1:]
    acf = acf/np.arange(len(residual), 0, -1)
    if acf[0] > 0:
        acf /= acf[0]
    lags = np.arange(len(acf))*step
    loc, _ = find_peaks(acf)
    candidates = [int(i) for i in loc if .7*peak['period_um'] <= lags[i] <= 1.3*peak['period_um']
                  and i < len(acf)/3]
    index = max(candidates, key=lambda i: acf[i]) if candidates else None
    period = None
    if index is not None:
        denom = acf[index-1]-2*acf[index]+acf[index+1]
        shift = .5*(acf[index-1]-acf[index+1])/denom if denom < 0 else 0.
        period = float((index+np.clip(shift, -.5, .5))*step)
    return {'distance_um': distance, 'values': values, 'lag_um': lags, 'acf': acf,
            'acf_period_um': period, 'line_center_px': center.tolist(),
            'line_direction_px_per_um': d_pixels.tolist(),
            'note': 'ACF searched near chosen texture frequency; p/2 and 2p ambiguity not removed'}


def map_video_times(micro_pts_s, phone_pts_s, phone_minus_micro_s):
    """Nearest phone presentation frame, with explicit uncovered time ranges."""
    a, b = np.asarray(micro_pts_s, float), np.asarray(phone_pts_s, float)
    if (a.ndim != 1 or b.ndim != 1 or len(b) < 2 or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(phone_minus_micro_s)
            or np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0)):
        raise ValueError('Finite strictly increasing presentation timestamps required')
    rows = []
    for i, t in enumerate(a):
        target = float(t+phone_minus_micro_s)
        if target < b[0] or target > b[-1]:
            rows.append({'micro_frame_index': i, 'micro_pts_s': float(t), 'phone_target_s': target,
                         'phone_frame_index': None, 'phone_pts_s': None, 'residual_s': None,
                         'status': 'OUTSIDE_PHONE_RECORDING'})
            continue
        j = int(np.searchsorted(b, target))
        options = [k for k in (j-1, j) if 0 <= k < len(b)]
        k = min(options, key=lambda k: abs(b[k]-target))
        rows.append({'micro_frame_index': i, 'micro_pts_s': float(t), 'phone_target_s': target,
                     'phone_frame_index': k, 'phone_pts_s': float(b[k]),
                     'residual_s': float(b[k]-target), 'status': 'TIME_MATCH_ONLY'})
    return rows
