"""Continue saved quarter256 after the original controller's missing-wrapper error.

An execution-only derivative: immutable original plans, phase workers, health
observer, checkpoints, optimizer state, data and all scientific code are reused.
Initialization and the completed first256 updates are never replayed.
"""
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args()
    from evidence_dial.benchmark import file_sha
    from evidence_dial.io import write_json
    from evidence_dial.behavior_fullanswer_health_v3 import passing_guard
    from particle_sliders.lifecycle import read_packet
    def pin(p):
        p = Path(p).resolve()
        return {'path': str(p), 'sha256': file_sha(p)}
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    folder = plan_path.parent
    out = folder / 'run'
    for record in plan['inputs'] + plan['sources']:
        assert pin(record['path']) == record
    assert plan['checkpoints'] == [256, 512, 768, 1024]
    assert plan['session_options']['steps'] == 1024 and plan['ema_decay'] is None
    assert not (out / 'candidate-failed.json').exists()
    assert not list(out.glob('candidate-confirmed-failed-*.json'))
    identity = json.loads((folder / 'controller-first.json').read_text())
    terminal_path = folder / 'actual-tool-terminal-receipt-first.json'
    terminal = json.loads(terminal_path.read_text())
    assert terminal['actual_tool_terminal_collected'] and terminal['exit_code'] == 1
    assert terminal['pid'] == identity['pid'] and terminal['birth'] == identity['birth']
    assert terminal['reaped'] and terminal['all_children_reaped'] and terminal['private_leases_free']
    assert not Path(f'/proc/{identity["pid"]}').exists()
    original_phases = 4 if plan['gpu'] == 0 else 3
    bad = folder / 'controller-executions' / f'phase-{original_phases:03d}'
    command = json.loads(bad.with_suffix('.json').read_text())['command']
    assert command == [str(ROOT / '.venv/bin/python'), 'tools/check_behavior_fullanswer_health_v3.py',
        '--candidate', str(out), '--step', '256']
    error = bad.with_suffix('.log').read_text()
    assert "can't open file" in error and 'No such file or directory' in error
    assert 'check_behavior_fullanswer_health_v3.py' in error
    for index in range(1, original_phases + 1):
        phase = folder / 'controller-executions' / f'phase-{index:03d}.json'
        execution = json.loads(phase.read_text())
        end = json.loads(phase.with_suffix('.terminal.json').read_text())
        assert end['reaped'] and not Path(f'/proc/{execution["pid"]}').exists()
        assert end['returncode'] == (2 if index == original_phases else 0)
    assert (out / 'RAW-000256.pt').is_file() and (out / 'NATIVE-000256.pt').is_file()
    assert (out / 'probes-000256.json').is_file() and not (out / 'FULL-000257.pt').exists()
    packet = read_packet(out / 'FULL-000256.pt')
    assert packet['payload']['game']['completed_steps'] == 256
    assert packet['payload']['ema'] == {'decay': None, 'values': None}
    del packet
    resource = json.loads((out / 'train-000256-first-resources.completed.json').read_text())
    assert resource['state'] == 'complete' and resource['failure'] is None
    health_wrapper = ROOT / 'tools/check_behavior_fullanswer_health_v2.py'
    assert pin(health_wrapper) in plan['sources']
    lock = (folder / '.controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    first = folder / 'continuation-controller-first.json'
    assert not first.exists()
    parent = os.getppid()
    assert ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0 and os.getppid() == parent
    def stop(sig, frame):
        raise KeyboardInterrupt('Owned continuation interrupted; preserve saved state')
    signal.signal(signal.SIGTERM, stop)
    write_json(first, {'time': time.time(), 'pid': os.getpid(),
        'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'argv': [x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        'plan': pin(plan_path), 'source': pin(__file__), 'corrected_health_wrapper': pin(health_wrapper),
        'original_root_terminal': pin(terminal_path), 'restored_FULL256': pin(out / 'FULL-000256.pt'),
        'completed_prefix_replayed': False, 'new_initialization': False,
        'scope': 'Correct only health checker executable filename; all scientific source/configuration unchanged'})
    executions = folder / 'continuation-controller-executions'
    executions.mkdir(exist_ok=False)
    counter = 0
    def child(command):
        nonlocal counter
        counter += 1
        receipt = executions / f'phase-{counter:03d}.json'
        assert not receipt.exists()
        process = None
        with receipt.with_suffix('.log').open('x') as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                    'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'})
                write_json(receipt, {'time': time.time(), 'pid': process.pid, 'command': command})
                code = process.wait()
            finally:
                if process is not None:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    write_json(receipt.with_suffix('.terminal.json'), {
                        'time': time.time(), 'returncode': process.returncode, 'reaped': True})
        assert code == 0, f'Continuation phase failed{code}; preserve all state before review'
    def phase(name, step):
        child([str(ROOT / '.venv/bin/python'), 'tools/behavior_expanded_phase_worker_v2.py',
            '--plan', str(plan_path), '--phase', name, '--stop', str(step), '--attempt', 'first'])
    try:
        for quarter, step in enumerate(plan['checkpoints'], 1):
            if step > 256:
                phase('train', step)
            child([str(ROOT / '.venv/bin/python'), 'tools/check_behavior_fullanswer_health_v2.py',
                '--candidate', str(out), '--step', str(step)])
            assert passing_guard(out, step, out / f'FULL-{step:06d}.pt') is not None
            phase('evaluate', step)
            marker = folder / f'quarter-{quarter:02d}-finished-first.json'
            assert not marker.exists()
            write_json(marker, {'time': time.time(), 'step': step, 'budget': 1024,
                'finished_DEV_strengths': plan['development_strengths'], 'partial_scores_reported': False,
                'execution_derivative': pin(__file__)})
        for record in plan['inputs'] + plan['sources']:
            assert pin(record['path']) == record
        write_json(folder / 'completion-first.json', {'time': time.time(), 'complete': True,
            'steps': 1024, 'all4quarters_and3strengths_finished': True, 'weight_EMA': False,
            'selection_performed': False, 'holdout_access': False,
            'operational_filename_failure_preserved': pin(folder / 'controller-failed.json'),
            'continuation_controller': pin(first)})
        print(json.dumps({'complete': True, 'steps': 1024, 'gpu': plan['gpu'], 'replayed_updates': 0}))
    except BaseException as error:
        write_json(folder / 'continuation-controller-failed.json', {
            'time': time.time(), 'error': repr(error), 'completed_work_preserved': True})
        raise


if __name__ == '__main__':
    main()
