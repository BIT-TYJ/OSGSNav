"""V3: validated V2 clustering + exact-input FMM cache + detailed wall timers."""
import os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navigation_acceleration_20261027.reachability_cache import install
cache=install()
root=os.environ.get('OSGS_MEMORY_VERIFY_DIR') or os.environ.get('OSGS_SUPP_RUN')
if root:
    from navigation_acceleration_20261027.timing import install as install_timing
    ledger=install_timing(Path(root)/'detailed_timing.json',cache)
print('[NAVIGATION_ACCELERATION_V3] exact-input reachability memo + V2 clustering; no iteration/action budget change',flush=True)
entry=str(ROOT/'memory_safe_clustering/fast_v2/run_osgsnav.py')
sys.argv=[entry,*sys.argv[1:]]
runpy.run_path(entry,run_name='__main__')
