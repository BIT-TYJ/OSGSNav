"""Read-only observers for isolated repair verification; no full map copies."""
import json,os,resource,time
from pathlib import Path

def emit(event, **data):
    root=os.environ.get('OSGS_MEMORY_VERIFY_DIR')
    if not root:return
    try:
        from five_goal_benchmark_20261026.runtime import instrumentation
        target=instrumentation.CURRENT_TARGET
    except ImportError:target=None
    with open('/proc/self/status') as f:
        rss=next(int(s.split()[1])*1024 for s in f if s.startswith('VmRSS:'))
    row=dict(event=event,time=time.time(),pid=os.getpid(),target_index=target,rss_bytes=rss,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**data)
    with (Path(root)/'telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

def install(base):
    class ObservedDBSCAN(base):
        def fit(self,x,y=None,sample_weight=None):
            t=time.monotonic();emit('cluster_start',points=len(x),eps=self.eps,min_samples=self.min_samples)
            try:
                result=super().fit(x,y,sample_weight)
            except BaseException as exc:
                emit('cluster_error',seconds=time.monotonic()-t,error=repr(exc));raise
            emit('cluster_end',points=len(x),seconds=time.monotonic()-t,clusters=len(set(self.labels_)-{-1}))
            return result
    from gaussian_mapping.mapper.realtime_3dgs import RealtimeSLAM
    original=RealtimeSLAM.add_frame
    counts={}
    def observed(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        key=id(self);counts[key]=counts.get(key,0)+1
        if counts[key]==1 or counts[key]%25==0:
            emit('mapping',observed_frames=counts[key],gaussians=int(self.gaussians.get_xyz.shape[0]),keyframes=len(getattr(self,'keyframes',{})))
        return result
    RealtimeSLAM.add_frame=observed
    return ObservedDBSCAN
