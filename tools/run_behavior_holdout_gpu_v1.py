"""Own a finite locked inference shard on one GPU; no test-based decisions."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--gpu', type=int, choices=[0,1], required=True); args = ap.parse_args()
    from evidence_dial.behavior_confirmation_v1 import pin, read, validate_plan
    from evidence_dial.io import write_json
    plan = validate_plan(read(args.plan)); folder = args.plan.resolve().parent / f'GPU{args.gpu}'
    folder.mkdir(exist_ok=True); first = folder / 'controller-first.json'; assert not first.exists()
    lock = (folder / '.controller.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = os.getppid(); assert ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0 and os.getppid() == parent
    def stop(sig, frame): raise KeyboardInterrupt('Owned confirmation shard interrupted; retain actual firsts')
    signal.signal(signal.SIGTERM, stop)
    write_json(first, {'time': time.time(), 'pid': os.getpid(),
        'birth': Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'argv': [x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        'plan': pin(args.plan), 'gpu': args.gpu})
    arms = [a for a in plan['arms'] if a['gpu'] == args.gpu]
    try:
        for index, arm in enumerate(arms,1):
            receipt = folder / 'executions' / f'arm-{index:03d}.json'; assert not receipt.exists()
            command = [str(ROOT / '.venv/bin/python'), 'tools/measure_behavior_holdout_arm_v1.py',
                '--plan', str(args.plan.resolve()), '--name', arm['name']]
            receipt.parent.mkdir(exist_ok=True); process = None
            with receipt.with_suffix('.log').open('x') as log:
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
                            'TOKENIZERS_PARALLELISM':'false', 'PYTHONDONTWRITEBYTECODE':'1'})
                    write_json(receipt, {'time': time.time(), 'pid': process.pid, 'command': command, 'name': arm['name']})
                    code = process.wait()
                finally:
                    if process is not None:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try: process.wait(timeout=20)
                            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
                        write_json(receipt.with_suffix('.terminal.json'), {'time': time.time(), 'returncode': process.returncode, 'reaped': True})
            assert code == 0, 'Preserve failed/unknown firsts; no automatic new generation attempt'
            write_json(folder / f'arm-{index:03d}-finished-first.json', {'time': time.time(), 'name': arm['name'],
                'completed_arms': index, 'total_arms': len(arms), 'partial_scores_read': False})
        validate_plan(read(args.plan))
        write_json(folder / 'completion-first.json', {'time': time.time(), 'complete': True,
            'gpu': args.gpu, 'arms': [a['name'] for a in arms], 'cases_each': 32, 'selection_performed': False})
        print(json.dumps({'complete': True, 'gpu': args.gpu, 'arms': len(arms)}))
    except BaseException as error:
        write_json(folder / 'controller-failed-first.json', {'time': time.time(), 'error': repr(error), 'firsts_preserved': True})
        raise


if __name__ == '__main__': main()
