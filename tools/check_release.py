#!/usr/bin/env python3
"""CPU-only syntax, exported-source integrity and local import checks."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    errors = []
    checked = 0
    for p in ROOT.rglob('*.py'):
        if '.git' in p.parts:
            continue
        checked += 1
        try:
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p.relative_to(ROOT)))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(str(exc))
    manifest = json.loads((ROOT/'release/source_manifest.json').read_text())
    for item in manifest['files']:
        rel = item['path']
        p = ROOT/('docs/IEVE_UPSTREAM_README.md' if rel == 'README.md' else rel)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            errors.append('Exported source differs: '+rel)
    # Check project imports without importing Habitat, torch or CUDA extensions.
    prefixes = {'baselines','agents','algo','envs','utils','gaussian_mapping','ieve_minav','multi_iin_v0',
                'five_goal_benchmark_20261026','memory_safe_clustering','navigation_acceleration_20261027'}
    for p in ROOT.rglob('*.py'):
        if '.git' in p.parts or '3rdparty' in p.parts or 'submodules' in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text())
        except (SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod = node.module
                if mod.split('.')[0] not in prefixes:
                    continue
                q = ROOT.joinpath(*mod.split('.'))
                if not q.is_dir() and not q.with_suffix('.py').is_file():
                    errors.append(str(p.relative_to(ROOT))+': missing local import '+mod)
    print(json.dumps(dict(python_files_checked=checked, source_files_verified=len(manifest['files']),
                          errors=sorted(set(errors)), simulation_executed=False,
                          hardware_executed=False), indent=2))
    return bool(errors)

if __name__ == '__main__':
    sys.exit(main())
