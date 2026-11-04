"""V2 statistics and bounded input capture for future exact slow-call replay."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from memory_safe_clustering.telemetry import install as install_base,emit


def install(base):
    observed=install_base(base)
    captured=set();used=[0]
    class ObservedV2(observed):
        def fit(self,x,y=None,sample_weight=None):
            t=time.monotonic();result=super().fit(x,y,sample_weight)
            elapsed=time.monotonic()-t
            emit('cluster_acceleration',points=len(x),seconds=elapsed,
                 cache_hit=bool(getattr(self,'cache_hit_',False)))
            # V1 did not save slow inputs. V2 saves selected natural inputs,
            # irrespective of success, bounded to 64 MiB per episode.
            if len(x)>=250000 or elapsed>=5:
                a=np.ascontiguousarray(x)
                key=hashlib.sha256(memoryview(a).cast('B')).hexdigest()
                if key not in captured and used[0]+a.nbytes<=64*1024**2:
                    capture_start=time.monotonic()
                    directory=Path(os.environ['OSGS_MEMORY_VERIFY_DIR'])/'cluster_inputs';directory.mkdir(exist_ok=True)
                    path=directory/(key+'.npy');np.save(path,a)
                    np.savez(path.with_suffix('.labels.npz'),labels=self.labels_,core=self.core_sample_indices_)
                    meta=dict(points=len(a),eps=self.eps,min_samples=self.min_samples,
                              seconds=elapsed,cache_hit=self.cache_hit_,input_array_sha256=key,
                              input_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    path.with_suffix('.json').write_text(json.dumps(meta,indent=2))
                    captured.add(key);used[0]+=a.nbytes
                    emit('cluster_input_capture',path=str(path),seconds=time.monotonic()-capture_start)
            return result
    return ObservedV2
