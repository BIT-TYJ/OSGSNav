"""Read-only experiment observers. No reset of semantic values or policy changes."""
import hashlib, json, os, random, time
from pathlib import Path
CURRENT_TARGET=0
def safe(x):
    if hasattr(x,'tolist'): return x.tolist()
    return str(x)
def write_event(kind, data):
    root=os.environ.get('OSGS_SUPP_RUN')
    if root:
        with (Path(root)/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=kind,time=time.time(),target_index=CURRENT_TARGET,data=data),default=safe)+'\n')
def record_retrieval(manager,event,data):
    write_event('retrieval',dict(name=event,**data))
def audit_extra(mapper,manager,info):
    global CURRENT_TARGET
    import torch
    CURRENT_TARGET=int(info.get('multi_target_index',0))
    slam=getattr(mapper,'slam',None); g=getattr(slam,'gaussians',None)
    out=dict(target_index=CURRENT_TARGET,goal_name=info.get('goal_name'),goal_cat_id=info.get('goal_cat_id'),
        cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved())
    try:
        import psutil
        out['process_rss_bytes']=psutil.Process().memory_info().rss
    except ImportError: pass
    if g is not None:
        for key in ['_semantic_target','_multi_semantic_evidence','_multi_semantic_observations','_confirmed_target_indices']:
            v=getattr(g,key,None)
            if torch.is_tensor(v):
                a=v.detach().cpu().numpy()
                out[key]=dict(shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest(),positive_count=int((a>0).sum()))
        sem=getattr(g,'_semantic_target',None); ev=getattr(g,'_multi_semantic_evidence',None)
        cid=info.get('goal_cat_id')
        if sem is not None and ev is not None and ev.ndim==2 and len(sem)==len(ev) and len(sem)>0 and cid is not None:
            old=sem>.4; known=ev.float().sum(1)>0
            out['target_scalar_count']=int(old.sum())
            out['target_scalar_known_other_class_count']=int((old & known & (ev.argmax(1)!=int(cid))).sum())
        metadata=getattr(slam,'kf_metadata',{})
        out['keyframes_with_target_mask']=sum(int(m.get('target_loss_mask_pixels',0) or 0)>0 for m in metadata.values())
    return out
def install_capture():
    from gaussian_mapping.mapper.realtime_3dgs import RealtimeSLAM
    if getattr(RealtimeSLAM,'_supp_capture_installed',False):return
    RealtimeSLAM._supp_capture_installed=True
    root=os.environ.get('OSGS_SUPP_RUN')
    if not root or os.environ.get('OSGS_SUPP_CAPTURE')!='1':return
    import numpy as np
    import torch
    root=Path(root); frames=root/'rgbd_sequence'; frames.mkdir(exist_ok=True)
    original=RealtimeSLAM.add_frame
    count=[0]; states=[0]
    def add_frame(self,rgb,depth,position,rotation,target_mask=None,region_type='target',training_context=None):
        if CURRENT_TARGET==0 and count[0]<120:
            count[0]+=1
            np.savez_compressed(frames/f'frame_{count[0]:04d}.npz',rgb=rgb,depth=depth,position=position,rotation=rotation,
                target_mask=np.asarray(target_mask) if target_mask is not None else np.empty((0,),np.uint8))
            with (frames/'frames.jsonl').open('a') as f:
                f.write(json.dumps(dict(frame=count[0],mapper_frame_before=self.frame_count,region_type=region_type,training_context=training_context),default=safe)+'\n')
            if count[0]==1:
                (frames/'config.json').write_text(json.dumps(dict(config=self.config,camera_params=self.camera_params,initial_gaussians=int(self.gaussians.get_xyz.shape[0])),default=safe,indent=2))
        return original(self,rgb,depth,position,rotation,target_mask=target_mask,region_type=region_type,training_context=training_context)
    RealtimeSLAM.add_frame=add_frame
    refresh=RealtimeSLAM.refresh_training_focus
    def refresh_focus(self,*args,**kwargs):
        if CURRENT_TARGET==0 and count[0]<=120:
            with (frames/'focus_events.jsonl').open('a') as f:
                f.write(json.dumps(dict(after_input_frame=count[0],args=args,kwargs=kwargs),default=safe)+'\n')
        return refresh(self,*args,**kwargs)
    RealtimeSLAM.refresh_training_focus=refresh_focus
    # Record actual stage outcomes and durations, never call a new policy stage.
    from ieve_minav.localizer.localization_system import LocalizationSystem
    for stage_name in ['_execute_fem','_execute_pre','_execute_ref','_execute_mr']:
        stage_original=getattr(LocalizationSystem,stage_name)
        def stage(self,*args,_original=stage_original,_name=stage_name,**kwargs):
            torch.cuda.synchronize();started=time.perf_counter()
            try:
                result=_original(self,*args,**kwargs)
            except Exception as exc:
                write_event('localization_stage_error',dict(stage=_name,error=repr(exc)))
                raise
            torch.cuda.synchronize()
            write_event('localization_stage',dict(stage=_name,seconds=time.perf_counter()-started,
                localization_object_id=id(self),result=result,pose=getattr(self,'best_pose',None),
                fem_score=getattr(self,'fem_best_score',None),pre_score=getattr(self,'pre_best_score',None),
                ref_score=getattr(self,'ref_best_score',None),current_object=getattr(self,'current_object',None),
                correct_instance_verified=None,verification_note='Internal stage result is not ground-truth instance verification'))
            return result
        setattr(LocalizationSystem,stage_name,stage)
    score=RealtimeSLAM.score_phase1_gaussmi_yaws
    def scorer(self,*args,**kwargs):
        result=score(self,*args,**kwargs)
        if states[0]>=10 or not result.get('ready'):return result
        states[0]+=1; out=root/'swmi_states'/f'state_{states[0]:03d}';out.mkdir(parents=True,exist_ok=False)
        t=time.perf_counter(); py_rng=random.getstate(); np_rng=np.random.get_state(); cpu_rng=torch.get_rng_state(); gpu_rng=torch.cuda.get_rng_state_all()
        try:
            model=self.gaussians; xyz=model.get_xyz.detach().cpu().numpy().copy()
            weights,debug=self._phase1_gaussmi_semantic_weights(target_weight=kwargs.get('target_weight',2.),nearby_weight=kwargs.get('nearby_weight',1.5),nearby_radius_m=kwargs.get('nearby_radius_m',2.))
            model.save_ply(str(out/'full_gaussians.ply'),include_color=True,highlight_confirmed=False)
            np.savez_compressed(out/'gaussian_state.npz',xyz=xyz,reliability=model.get_reliability.detach().cpu().numpy(),semantic_target=model.get_semantic_target.detach().cpu().numpy(),semantic_weights=weights.detach().cpu().numpy())
            position=args[0] if args else kwargs['current_position']; rotation=args[1] if len(args)>1 else kwargs['current_rotation']; offsets=args[2] if len(args)>2 else kwargs['yaw_offsets_deg']
            candidates=[]
            for offset in offsets:
                w2c=self._quat_to_matrix(position,self._relative_habitat_yaw_quaternion(rotation,float(offset)))
                penalty=1-kwargs.get('angle_penalty_k',.5)*(abs(float(offset))/kwargs.get('theta_max_deg',90.))**2
                candidates.append(dict(offset_deg=float(offset),valid=True,world_to_camera=w2c,angle_penalty=penalty))
            (out/'capture.json').write_text(json.dumps(dict(target_index=CURRENT_TARGET,mapping_frame=self.frame_count,xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),camera_parameters=self.camera_params,pipeline=vars(self.pipeline_params),diagnostic_candidates=candidates,online_semantic_counts=debug,controller_result=result,selection='first ten naturally ready scoring calls per preselected Shared sequence; no outcome filtering'),default=safe,indent=2))
        finally:
            random.setstate(py_rng);np.random.set_state(np_rng);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
            write_event('capture_overhead',dict(seconds=time.perf_counter()-t,state=states[0]))
        return result
    RealtimeSLAM.score_phase1_gaussmi_yaws=scorer
