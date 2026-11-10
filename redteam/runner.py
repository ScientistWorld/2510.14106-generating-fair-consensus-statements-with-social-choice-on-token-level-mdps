"""Install one candidate, run visible train and restored hidden test with no args.

All evaluated candidates exit before evaluation. There are no persistent
processes, evaluator edits, environment overrides, or interpreter modifications.
Only the candidate's ordinary on-disk artifacts and (for one shortcut) data
generator differ when the fresh evaluator process starts.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/user')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run_logged(command, logfile):
    with logfile.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def main(name):
    folder = ROOT / 'redteam' / name
    state = ROOT / 'redteam/.state' / name
    if state.exists():
        raise RuntimeError(f'Unrestored state exists: {state}')
    state.mkdir(parents=True)
    targets = [ROOT / p for p in ('checkpoints/core_test', 'scoring/scores_train.json',
               'scoring/scores_test.json', 'data/core_synthetic/generate.py')]
    saved = []
    protected = [ROOT / 'scripts/evaluate_test.sh', *sorted((ROOT / 'eval').rglob('*.py'))]
    protected_hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    started = time.time()
    summary = {'name': name, 'test_command': ['bash', 'scripts/evaluate_test.sh'],
               'protected_sha256_before': protected_hashes}
    try:
        for i, target in enumerate(targets):
            backup = state / str(i)
            exists = target.exists()
            saved.append((target, backup, exists))
            if exists:
                if target.is_dir():
                    shutil.copytree(target, backup)
                else:
                    shutil.copy2(target, backup)
        if targets[0].exists():
            shutil.rmtree(targets[0])
        targets[2].unlink(missing_ok=True)
        summary['method_exit'] = run_logged(
            ['python', '-m', f'redteam.{name}.method'], folder / 'method.log')
        if summary['method_exit'] != 0:
            raise RuntimeError(f'{name}: method failed; see method.log')
        summary['train_exit'] = run_logged(
            ['bash', 'scripts/evaluate_train.sh'], folder / 'train.log')
        if (ROOT / 'scoring/scores_train.json').exists():
            shutil.copy2(ROOT / 'scoring/scores_train.json', folder / 'scores_train_before_hook.json')
        summary['post_train_exit'] = run_logged(
            ['python', '-m', f'redteam.{name}.method', 'post_train'], folder / 'post_train.log')
        if targets[1].exists():
            shutil.copy2(targets[1], folder / 'scores_train.json')
        before = targets[2].stat().st_mtime_ns if targets[2].exists() else None
        if targets[2].exists():
            shutil.copy2(targets[2], folder / 'scores_before_test.json')
        summary['test_exit'] = run_logged(
            ['bash', 'scripts/evaluate_test.sh'], folder / 'test.log')
        after = targets[2].stat().st_mtime_ns if targets[2].exists() else None
        summary['scores_written_by_test'] = after is not None and before != after
        if targets[2].exists():
            shutil.copy2(targets[2], folder / 'scores_test.json')
        # Compact on-disk coverage evidence, without committing generated artifacts.
        manifest = ROOT / 'checkpoints/core_test/manifest.json'
        if manifest.exists():
            coverage = {}
            payload = json.loads(manifest.read_text())
            for path in payload['artifacts']:
                source = Path(path)
                if not source.is_absolute() and not source.exists():
                    source = ROOT / 'checkpoints/core_test' / source
                record = json.loads(source.read_text())
                for policy in record['policies']:
                    coverage.setdefault(policy['name'], []).append([record['rho'], record['seed']])
            summary['disk_policy_coverage'] = coverage
    except Exception as exc:
        summary['harness_error'] = repr(exc)
    finally:
        for target, backup, existed in reversed(saved):
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            if existed:
                if backup.is_dir():
                    shutil.copytree(backup, target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, target)
        summary['protected_sha256_after'] = {str(p.relative_to(ROOT)): digest(p) for p in protected}
        summary['protected_files_unchanged'] = summary['protected_sha256_before'] == summary['protected_sha256_after']
        summary['elapsed_seconds'] = time.time() - started
        (folder / 'run_summary.json').write_text(json.dumps(summary, indent=2))
        shutil.rmtree(state)
    print(json.dumps({k: summary.get(k) for k in
                      ('name', 'method_exit', 'train_exit', 'test_exit', 'scores_written_by_test', 'harness_error')}), flush=True)
    if summary.get('harness_error'):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
