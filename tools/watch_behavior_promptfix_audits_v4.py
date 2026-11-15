"""Audit each newly finished DEV quarter once while training continues.

CPU only. Never trains, selects, generates, judges, restarts or signals a
model worker. Existing admitted quarters are reused without repeating audits.
"""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from mine_behavior_promptfix_v4 import BASE, read, pin


def main():
    from evidence_dial.io import write_json
    from evidence_dial.continuation_resources_v1 import CPUObserverResources
    import audit_behavior_promptfix_quarter_v4 as observer
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', default='first')
    args = ap.parse_args()
    assert re.fullmatch('[a-z0-9-]+', args.attempt)
    folder = BASE / 'quarter-audit-watcher'
    folder.mkdir(parents=True, exist_ok=True)
    lease = (folder / '.owned.lock').open('a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = folder / (args.attempt + '-process.json')
    assert not receipt.exists()
    declaration_path = BASE / 'training-declaration-first.json'
    declaration = read(declaration_path)
    parent = os.getppid()
    assert ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0 and os.getppid() == parent
    def stop(sig, frame):
        raise KeyboardInterrupt('Owned CPU audit watcher stopped; model jobs remain independently owned')
    signal.signal(signal.SIGTERM, stop)
    write_json(receipt, {'time': time.time(), 'pid': os.getpid(),
        'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'argv': [x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        'declaration': pin(declaration_path), 'source': pin(__file__), 'auditor': pin(observer.__file__),
        'new_model_calls': 0, 'new_Jev_calls': 0, 'new_updates': 0})
    jobs = [(cohort, step) for step in [256, 512, 768, 1024] for cohort in declaration['cohorts']]
    try:
        while True:
            pending = 0
            for cohort, step in jobs:
                plan_path = Path(cohort['plan']['path'])
                assert pin(plan_path) == cohort['plan']
                output = BASE / f'GPU{cohort["gpu"]}-quarter{step}-independent-admission-first.json'
                if output.exists():
                    saved = read(output)
                    assert saved['state'] == 'declared-complete-behavior-quarter-independently-readmitted'
                    assert saved['plan'] == cohort['plan'] and saved['step'] == step
                    continue
                pending += 1
                candidate = plan_path.parent
                assert not (candidate / 'controller-failed.json').exists(), 'Model controller failed; review its exact saved state'
                marker = candidate / f'quarter-{step // 256:02d}-finished-first.json'
                if not marker.exists():
                    continue
                status = folder / f'GPU{cohort["gpu"]}-step{step}-{args.attempt}-resources.json'
                assert not status.exists()
                with CPUObserverResources(status) as resource:
                    resource.check()
                    result = observer.audit(plan_path, step)
                    resource.check()
                    assert not output.exists()
                    write_json(output, {'time': time.time(), 'source': pin(observer.__file__),
                        'execution_wrapper': pin(__file__), **result})
                print(json.dumps({'audited_gpu': cohort['gpu'], 'step': step,
                    'output': pin(output), 'arms': result['arms'], 'new_model_or_Jev_calls': 0}), flush=True)
            if not pending:
                break
            # This is an owned background process; no model work runs here.
            time.sleep(30)
        write_json(folder / (args.attempt + '-completion.json'), {'time': time.time(),
            'complete': True, 'quarter_admissions': 8, 'new_model_or_Jev_calls': 0, 'new_updates': 0})
    except BaseException as error:
        write_json(folder / (args.attempt + '-failure.json'), {'time': time.time(),
            'error': repr(error), 'all_prior_admissions_preserved': True, 'model_jobs_altered': False})
        raise


if __name__ == '__main__':
    main()
