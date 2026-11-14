"""Close the declared two-rate study and retain all 24-cell DEV maxima.

Consumes independent quarter admissions and actual root terminal receipts.
No models, new judgments, updates, or holdout rows are accessed.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/behavior-dial-20261114-v1'
sys.path.insert(0, str(ROOT / 'src'))


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verified(record):
    assert pin(record['path']) == record
    return Path(record['path'])


def read(path):
    return json.loads(Path(path).read_text())


def not_live(pid, birth):
    path = Path('/proc') / str(pid) / 'stat'
    return not path.exists() or path.read_text().rsplit(')', 1)[1].split()[19] != birth


def close():
    from evidence_dial.io import read_jsonl
    declaration = read(BASE / 'training-declaration-first.json')
    assert declaration['all24_candidate_cells_required'] and declaration['declared_before_first_update']
    assert [(c['gpu'], c['g_lr']) for c in declaration['cohorts']] == [(0, .001), (1, .00025)]
    controls_path = BASE / 'DEV-controls-GPU0-v1/independent-completion-audit-first.json'
    controls = read(controls_path)
    assert controls['state'] == 'all32-native-and-NanoGPT-Jev-control-records-readmitted'
    off = read_jsonl(verified(controls['metrics']['off']['predictions']))
    prompt = read_jsonl(verified(controls['metrics']['prompt-only']['predictions']))
    assert len(off) == len(prompt) == 16 and [r['id'] for r in off] == [r['id'] for r in prompt]
    arms, cohorts = [], []
    for cohort in declaration['cohorts']:
        plan_path = verified(cohort['plan']); plan = read(plan_path); folder = plan_path.parent; out = folder / 'run'
        assert plan['session_options']['steps'] == 128 and plan['checkpoints'] == [32, 64, 96, 128]
        assert plan['development_strengths'] == [.5, 1., 2.] and plan['ema_decay'] is None
        for source in plan['inputs'] + plan['sources']:
            verified(source)
        identity = read(folder / 'controller-first.json')
        assert identity['plan_sha256'] == pin(plan_path)['sha256']
        terminal_path = folder / 'actual-tool-terminal-receipt-first.json'; terminal = read(terminal_path)
        assert terminal['actual_tool_terminal_collected'] and terminal['actual_tool_terminal_chunk']
        assert terminal['exit_code'] == 0 and terminal['reaped']
        assert terminal['pid'] == identity['pid'] and terminal['birth'] == identity['birth']
        assert terminal['command'] == identity['argv'] and not_live(identity['pid'], identity['birth'])
        assert not (folder / 'controller-failed.json').exists() and not (out / 'candidate-failed.json').exists()
        assert not list(out.glob('candidate-confirmed-failed-*.json'))
        completion = read(folder / 'completion-first.json')
        assert completion['complete'] and completion['steps'] == 128 and completion['all4quarters_and3strengths_finished']
        assert completion['weight_EMA'] is False and not completion['holdout_access'] and not completion['selection_performed']
        for index in range(1, 14):
            phase = folder / 'controller-executions' / f'phase-{index:03d}.json'
            execution = read(phase); end = read(phase.with_suffix('.terminal.json'))
            assert end['reaped'] and end['returncode'] == 0 and not Path(f'/proc/{execution["pid"]}').exists()
        for phase, steps in [('initialize', [0]), ('train', [32,64,96,128]), ('evaluate', [32,64,96,128])]:
            for step in steps:
                resource = read(out / f'{phase}-{step:06d}-first-resources.completed.json')
                assert resource['state'] == 'complete' and resource['failure'] is None
        with (out / '.shared-gpu-job.lock').open('r+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lease, fcntl.LOCK_UN)
        audits = []
        for step in [32, 64, 96, 128]:
            path = BASE / f'GPU{cohort["gpu"]}-quarter{step}-independent-admission-first.json'; audit = read(path)
            assert audit['state'] == 'declared-complete-behavior-quarter-independently-readmitted'
            assert audit['plan'] == cohort['plan'] and audit['step'] == step and audit['budget'] == 128
            assert audit['all48_native_inputs_outputs_typed_Jev_scores_verified'] and audit['no_weight_EMA']
            assert audit['selection_performed'] is False and audit['holdout_rows_read'] == 0
            verified(audit['source']); verified(audit['FULL']); verified(audit['RAW']); verified(audit['NATIVE'])
            verified(audit['qualified_checkpoint_guard'])
            assert audit['control_admission'] == pin(controls_path)
            assert [a['strength'] for a in audit['arms']] == [.5, 1., 2.]
            for value in audit['arms']:
                assert value['cases'] == 16 and value['artifact'] == audit['NATIVE'] and value['FULL'] == audit['FULL']
                rows = read_jsonl(verified(value['predictions'])); verified(value['summary'])
                assert [r['id'] for r in rows] == [r['id'] for r in off]
                gain = sum(r['effect'] - b['effect'] for r, b in zip(rows, off)) / 16
                assert math.isfinite(gain) and gain == value['effect_ON_minus_OFF']
                arms.append({**value, 'gpu': cohort['gpu'], 'g_lr': cohort['g_lr'],
                    'name': f'PG-GPU{cohort["gpu"]}-step{step}-alpha{value["strength"]:g}',
                    'quarter_admission': pin(path), 'plan': cohort['plan'],
                    'effect_ON_minus_prompt_only': sum(r['effect'] - b['effect'] for r, b in zip(rows, prompt)) / 16})
            audits.append(pin(path))
        cohorts.append({'plan': cohort['plan'], 'actual_terminal': pin(terminal_path),
            'completion': pin(folder / 'completion-first.json'), 'independent_quarters': audits})
    assert len(arms) == 24 and len({a['name'] for a in arms}) == 24
    maximum = max(a['effect_ON_minus_OFF'] for a in arms)
    selected = [a for a in arms if a['effect_ON_minus_OFF'] == maximum]
    return {'state': 'declared-two-rate-24-cell-DEV-search-complete', 'cohorts': cohorts,
        'all24_arms': arms, 'selector': 'equal-case mean intended effect ON minus matched OFF only',
        'maximum_DEV_effect_gain': maximum, 'all_maximum_ties': selected,
        'display_checkpoint_rule': 'First maximum in declared GPU0 then GPU1, step32/64/96/128, strength0.5/1/2 order; a presentation identity, not a superior tie',
        'display_checkpoint_representative': selected[0], 'exact_strength_health_still_required': True,
        'holdout32_still_sealed': True, 'holdout_based_reselection': False,
        'new_model_calls': 0, 'new_Jev_calls': 0, 'new_updates': 0, 'holdout_rows_read': 0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    assert not args.output.exists(); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from evidence_dial.io import write_json
    result = close()
    write_json(args.output, {'time': time.time(), 'source': pin(__file__), **result})
    print(json.dumps({'closure': pin(args.output), 'maximum_DEV_effect_gain': result['maximum_DEV_effect_gain'],
        'maximum_ties': [a['name'] for a in result['all_maximum_ties']], 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
