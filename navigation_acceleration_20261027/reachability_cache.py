"""Bounded exact-input memoization of the original temp-goal FMM result."""
from collections import OrderedDict
import inspect,time
import numpy as np

class ExactReachabilityCache:
    def __init__(self, original, max_bytes=64*1024**2, max_entries=128):
        self.original=original;self.signature=inspect.signature(original)
        self.max_bytes=max_bytes;self.max_entries=max_entries;self.entries=OrderedDict()
        self.bytes=0;self.calls=0;self.hits=0;self.key_seconds=0.;self.compute_seconds=0.
    def __call__(self,*args,**kwargs):
        self.calls+=1;t=time.perf_counter()
        bound=self.signature.bind(*args,**kwargs);bound.apply_defaults()
        parts=[];cost=0
        for name,value in bound.arguments.items():
            if value is None:
                part=None
            elif name in ('grid','goal_map','visited','collision_map','obstacle_selem'):
                a=np.asarray(value)
                if a.dtype.hasobject:
                    return self.original(*args,**kwargs)
                data=a.tobytes(order='C');cost+=len(data)
                part=(a.shape,a.dtype.str,data)
            else:
                # These inputs are numeric scalars or short start/window lists.
                a=np.asarray(value)
                if a.dtype.hasobject:return self.original(*args,**kwargs)
                data=a.tobytes(order='C');cost+=len(data)
                part=(a.shape,a.dtype.str,data)
            parts.append((name,part))
        key=tuple(parts);self.key_seconds+=time.perf_counter()-t
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key)
            return self.entries[key][0]
        t=time.perf_counter();value=self.original(*args,**kwargs);self.compute_seconds+=time.perf_counter()-t
        if cost<=self.max_bytes and self.max_entries>0:
            while self.entries and (len(self.entries)>=self.max_entries or self.bytes+cost>self.max_bytes):
                _,old=self.entries.popitem(last=False);self.bytes-=old[1]
            self.entries[key]=(value,cost);self.bytes+=cost
        return value
    def snapshot(self):
        return dict(calls=self.calls,hits=self.hits,entries=len(self.entries),
                    payload_bytes=self.bytes,max_payload_bytes=self.max_bytes,
                    key_seconds=self.key_seconds,compute_seconds=self.compute_seconds)

def install():
    from envs.utils import ieve_reachability
    cache=ExactReachabilityCache(ieve_reachability.compute_temp_goal_distance_ieve)
    ieve_reachability.compute_temp_goal_distance_ieve=cache
    return cache
