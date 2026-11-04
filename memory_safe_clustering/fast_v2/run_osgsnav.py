"""Opt-in grid-accelerated exact DBSCAN entry; frozen V1 code is untouched."""
import os,runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from memory_safe_clustering.fast_v2.cached import MemorySafeDBSCAN
import sklearn.cluster
if os.environ.get('OSGS_MEMORY_VERIFY_DIR'):
    from memory_safe_clustering.fast_v2.telemetry import install
    MemorySafeDBSCAN=install(MemorySafeDBSCAN)
sklearn.cluster.DBSCAN=MemorySafeDBSCAN
print('[MEMORY_SAFE_DBSCAN_V2] exact clique-cell connectivity; 64MiB exact-input cache; pid=%d'%os.getpid(),flush=True)
entry=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(entry,run_name='__main__')
