#!/usr/bin/env python3
"""Portable, single-GPU launch/preflight for the historical Habitat adapter."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--unigoal-root',type=Path,default=os.environ.get('UNIGOAL_ROOT'))
 p.add_argument('--goals',type=int,choices=[2,5],default=5)
 p.add_argument('--gpu',type=int,default=0)
 p.add_argument('--episode',type=int,default=0)
 p.add_argument('--assets-root',type=Path,default=REPO)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--execute',action='store_true',help='Without this flag, only validate source and print the command')
 a=p.parse_args()
 if a.unigoal_root is None:p.error('Provide --unigoal-root or UNIGOAL_ROOT')
 u=a.unigoal_root.expanduser().resolve();lock=json.loads((HERE/'upstream.lock.json').read_text())
 try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=u,text=True).strip()
 except (OSError,subprocess.CalledProcessError):p.error('UniGoal root must be a checkout of the pinned dependency')
 if commit!=lock['commit']:p.error('UniGoal revision differs from upstream.lock.json; preserve it as a separate version')
 bad=[x['path'] for x in lock['files'] if not (u/x['path']).is_file() or hashlib.sha256((u/x['path']).read_bytes()).hexdigest()!=x['sha256']]
 if bad:p.error('Dependency source hash mismatch: '+', '.join(bad[:8]))
 if not 0<=a.episode<100:p.error('--episode must be 0--99')
 out=a.output.resolve();manifest=REPO/('multi_iin_v0/manifests/miin_v0_val_k2.json' if a.goals==2 else 'five_goal_benchmark_20261026/configs/five_goal_100.json')
 cmd=[sys.executable,'-u',str(HERE/f'k{a.goals}/main.py'),'--config-file',str(HERE/f'configs/k{a.goals}.yaml'),'--unigoal_root',str(u),'--assets_root',str(a.assets_root.resolve()),'--multi_manifest',str(manifest),'--multi_episode_start',str(a.episode),'--multi_num_episodes','1','--multi_goal_step_budget','1000']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',UNIGOAL_ROOT=str(u),UNIGOAL_DUMP_LOCATION=str(out),UNIGOAL_EXPERIMENT_ID='ug',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
 env['PYTHONPATH']=os.pathsep.join([str(REPO),str(u),str(u/'third_party/Grounded-Segment-Anything'),str(u/'third_party/Grounded-Segment-Anything/segment-anything')])
 plan=dict(argv=cmd,cwd=str(u),upstream_commit=commit,checked_source_files=len(lock['files']),backend='habitat',env={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','UNIGOAL_ROOT','UNIGOAL_DUMP_LOCATION','UNIGOAL_EXPERIMENT_ID','PYTHONPATH']})
 print(json.dumps(plan,indent=2),flush=True)
 if not a.execute:return
 if out.exists() and any(out.iterdir()):p.error('Use a fresh output directory')
 out.mkdir(parents=True,exist_ok=True);(out/'launch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 raise SystemExit(subprocess.call(cmd,cwd=u,env=env))
if __name__=='__main__':main()
