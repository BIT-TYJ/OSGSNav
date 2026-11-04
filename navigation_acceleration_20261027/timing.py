"""Low-frequency inclusive/exclusive wall timers; no CUDA synchronization added."""
import atexit,functools,json,os,threading,time
from collections import defaultdict
from pathlib import Path

class TimingLedger:
    def __init__(self,output,cache):
        self.output=Path(output);self.cache=cache;self.data=defaultdict(lambda:[0,0.,0.,0.]);self.local=threading.local();self.last=0.;self.started=time.time()
        atexit.register(self.flush)
    def wrap(self,owner,name,label=None):
        original=getattr(owner,name,None)
        if original is None or not callable(original):return
        label=label or '%s.%s'%(getattr(owner,'__name__',type(owner).__name__),name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            stack=getattr(self.local,'stack',None)
            if stack is None:stack=self.local.stack=[]
            state=[time.perf_counter(),0.];stack.append(state)
            try:return original(*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-state[0];stack.pop()
                if stack:stack[-1][1]+=elapsed
                row=self.data[label];row[0]+=1;row[1]+=elapsed;row[2]+=max(0.,elapsed-state[1]);row[3]=max(row[3],elapsed)
                if time.monotonic()-self.last>30:self.flush()
        setattr(owner,name,measured)
    def flush(self):
        self.last=time.monotonic()
        rows=[dict(function=k,calls=v[0],inclusive_s=v[1],exclusive_s=v[2],max_s=v[3]) for k,v in self.data.items()]
        rows.sort(key=lambda r:r['exclusive_s'],reverse=True)
        data=dict(updated=time.time(),started=self.started,pid=os.getpid(),functions=rows,reachability_cache=self.cache.snapshot(),note='Main process wall timers, nested inclusive/exclusive. GPU launches are asynchronous; no added synchronization; child simulator internals not attributed.')
        self.output.parent.mkdir(parents=True,exist_ok=True);temp=self.output.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,self.output)

def install(output,cache):
    ledger=TimingLedger(output,cache)
    from gaussian_mapping.mapper import realtime_3dgs
    from gaussian_mapping.gaussian_splatting.scene.old_gaussian_model import GaussianModel
    from ieve_minav.manager import MinAVConfirmationLocalizationManager
    from envs.utils import ieve_reachability
    from envs.utils.fmm_planner import FMMPlanner
    from five_goal_benchmark_20261026.runtime.multi_env import MultiVecPyTorch
    from multi_iin_v0.runtime.gaussian_semantic_labels import GaussianSemanticInstanceTrainer
    from multi_iin_v0.runtime.gaussian_capacity_manager import GaussianCapacityManager
    for name in ['add_frame','create_camera','_add_keyframe_gaussians','_train_mapping','_update_gaussmi_reliability','refresh_training_focus']:
        ledger.wrap(realtime_3dgs.RealtimeSLAM,name)
    ledger.wrap(realtime_3dgs,'render','mapping.render_forward_enqueue')
    ledger.wrap(realtime_3dgs,'get_loss_mapping','mapping.loss_forward_enqueue')
    import torch
    ledger.wrap(torch.Tensor,'backward','torch.backward_wall')
    ledger.wrap(torch.optim.Adam,'step','torch.Adam.step_wall')
    for name in ['update_target_semantic_from_mask','densify_and_prune','prune_points','add_densification_stats','project_mask_to_gaussian_indices']:
        ledger.wrap(GaussianModel,name)
    for name in ['update_frame','get_goal_override','_prepare_buffer_goal','_ieve_temp_goal_reachability','run_target_confirmation','run_close_localization','_run_full_close_localization','_refresh_slam_training_before_confirmation','_refresh_slam_training_before_localization','_cluster_target_confirm_indices','_select_candidate_clusters','_targeted_quick_lock_candidates']:
        ledger.wrap(MinAVConfirmationLocalizationManager,name)
    ledger.wrap(ieve_reachability,'compute_temp_goal_distance_ieve')
    ledger.wrap(FMMPlanner,'set_multi_goal');ledger.wrap(FMMPlanner,'set_goal')
    ledger.wrap(MultiVecPyTorch,'plan_act_and_preprocess');ledger.wrap(MultiVecPyTorch,'get_frontier_map')
    ledger.wrap(GaussianSemanticInstanceTrainer,'observe');ledger.wrap(GaussianCapacityManager,'maybe_prune')
    return ledger
