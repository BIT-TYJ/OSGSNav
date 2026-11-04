#!/usr/bin/env python3
"""Relocatable launcher for the frozen five-goal V5 simulation runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0, help='Physical CUDA device; only this device is exposed')
    parser.add_argument('--episode', type=int, default=0, help='Manifest sequence index, 0--99')
    parser.add_argument('--variant', choices=['Continuous-Shared', 'Continuous-Clear'], default='Continuous-Shared')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--execute', action='store_true', help='Run simulation; otherwise print the resolved command')
    args = parser.parse_args()
    if not 0 <= args.episode < 100:
        parser.error('--episode must be 0--99')
    output = args.output.resolve()
    values = dict(repo=str(ROOT), output=str(output), episode=args.episode)
    argv = [value.format(**values) for value in json.loads((ROOT/'configs/multi_goal_v5_argv.json').read_text())]
    argv[argv.index('--multi_variant') + 1] = args.variant
    cmd = [sys.executable, '-u', str(ROOT/'navigation_acceleration_20261027/run_osgsnav.py'),
           str(ROOT/'five_goal_benchmark_20261026/runtime/main_multi_iin_frontier_v5.py'), *argv]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               OSGS_SUPP_RUN=str(output), OSGS_SUPP_CAPTURE='0', OSGS_MEMORY_VERIFY_DIR=str(output))
    plan = dict(argv=cmd, cwd=str(ROOT), variant=args.variant, episode=args.episode,
                env={key: env[key] for key in ['CUDA_VISIBLE_DEVICES','OSGS_SUPP_RUN','OSGS_SUPP_CAPTURE','OSGS_MEMORY_VERIFY_DIR']})
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be fresh; existing results are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    (output/'launch_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    # Preserve source CLI; startup diagnostics and errors remain visible.
    raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))

if __name__ == '__main__':
    main()
