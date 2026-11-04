"""Optional, read-only observations of real calculation stages.

No observer means no copying or file IO. An observer receives a deep copy so it
cannot change the scientific arrays, candidates, parameters, or random state.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import sys
from .runtime_trace import event_values, enabled

_observer = ContextVar('williams_stage_observer', default=None)


@contextmanager
def observe(callback, accepts=lambda name, data: True):
    token = _observer.set((callback, accepts))
    try:
        yield
    finally:
        _observer.reset(token)


def emit(name, **data):
    if enabled():
        frame=sys._getframe(1)
        origin=frame.f_code.co_filename.replace('\\','/').rsplit('/',1)[-1]
        event_values(name,data,f'{origin}:{frame.f_code.co_name}:{frame.f_lineno}')
    current = _observer.get()
    # Filtering only needs immutable scalar tags (e.g. peak UID / channel).
    # Never lend the filter mutable scientific arrays or candidate lists.
    metadata={k:v for k,v in data.items() if isinstance(v,(str,int,float,bool,type(None)))} if current is not None else {}
    if current is not None and current[1](name, metadata):
        current[0](name, deepcopy(data))
