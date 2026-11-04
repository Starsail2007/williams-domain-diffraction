"""Print actual intermediate values without changing calculation or random state."""
from contextlib import contextmanager
from contextvars import ContextVar
import sys
import time
import numpy as np

_trace = ContextVar('williams_numeric_trace', default=None)


def enabled():
    return _trace.get() is not None


@contextmanager
def console_values():
    if enabled():
        yield
        return
    token = _trace.set(dict(start=time.monotonic(), sequence=0))
    try:
        yield
    finally:
        _trace.reset(token)


def describe(name, value, depth=0):
    indent = '  ' * (depth + 1)
    if isinstance(value, np.ndarray):
        print(f'{indent}{name}: shape={value.shape}, dtype={value.dtype}, size={value.size}', flush=True)
        if value.dtype.kind in 'buifc' and value.size:
            finite = np.isfinite(value)
            numbers = np.abs(value[finite]) if np.iscomplexobj(value) else value[finite]
            if numbers.size:
                print(f'{indent}  finite={int(finite.sum())}/{value.size}; '
                      f'{"abs " if np.iscomplexobj(value) else ""}min={numbers.min():.9g}, '
                      f'max={numbers.max():.9g}, mean={numbers.mean():.9g}, std={numbers.std():.9g}', flush=True)
            sample = value if value.size <= 36 else value.ravel()[:12]
            print(f'{indent}  values={np.array2string(sample, precision=8, threshold=36, max_line_width=120)}', flush=True)
            if value.ndim == 2 and value.size > 36:
                y, x = np.array(value.shape) // 2
                patch = value[max(0,y-1):y+2,max(0,x-2):x+3]
                print(f'{indent}  centre_3x5={np.array2string(patch, precision=6, max_line_width=120)}', flush=True)
    elif isinstance(value, dict) and depth < 5:
        print(f'{indent}{name}: {len(value)} fields', flush=True)
        for key, item in value.items():
            describe(str(key), item, depth+1)
    elif isinstance(value, (list, tuple)) and depth < 5:
        if len(value) <= 16 and all(isinstance(x,(str,int,float,bool,np.generic,type(None))) for x in value):
            print(f'{indent}{name}={value}', flush=True)
        else:
            print(f'{indent}{name}: length={len(value)}', flush=True)
            for i, item in enumerate(value[:60]):
                describe(f'[{i}]', item, depth+1)
            if len(value) > 60:
                print(f'{indent}  ... {len(value)-60} further entries; complete evidence is saved in the run files', flush=True)
    else:
        text = str(value)
        print(f'{indent}{name}={text[:1200]}', flush=True)


def log_values(name, **values):
    state = _trace.get()
    if state is None:
        return
    frame = sys._getframe(1)
    origin = frame.f_code.co_filename.replace('\\','/').rsplit('/',1)[-1]
    event_values(name, values, f'{origin}:{frame.f_code.co_name}:{frame.f_lineno}')


def event_values(name, values, origin):
    state = _trace.get()
    if state is None:
        return
    state['sequence'] += 1
    print(f"\n[VAR {state['sequence']:05d} +{time.monotonic()-state['start']:.3f}s] {origin} | {name}", flush=True)
    for key, value in values.items():
        describe(key, value)
