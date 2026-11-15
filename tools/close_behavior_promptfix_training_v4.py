"""Select from all declared, independently audited DEV cells; never read test answers."""
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from mine_behavior_promptfix_v4 import BASE, DIAG, read, pin


def absent(pid, birth):
    path = Path('/proc') / str(pid) / 'stat'
    return not path.exists() or path.read_text().rsplit(')', 1)[1].split()[19] != birth


def close():
    from evidence_dial.io import read_jsonl, write_json
    declaration = read(BASE / 'training-declaration-first.json')
    assert declaration['all24_DEV_cells_required'] and declaration['no_weight_EMA']
    assert [(c['gpu'], c['g_lr']) for c in declaration['cohorts']] == [(0, .00025), (1, .0000625)]
    diagnostic = read(DIAG / 'independent-audit-first.json')
    metrics = {a['name']: a for a in diagnostic['arms']}
    controls = {name: read_jsonl(metrics[arm]['predictions']['path'])
        for name, arm in [('off', 'off'), ('prompt-only', 'target0-unrestricted')]}
    for arm in ['off', 'target0-unrestricted']:
        assert pin(metrics[arm]['predictions']['path']) == metrics[arm]['predictions']
    assert len(controls['off']) == len(controls['prompt-only']) == 16
    arms, cohorts = [], []
    for cohort in declaration['cohorts']:
        assert pin(cohort['plan']['path']) == cohort['plan']
        plan_path = Path(cohort['plan']['path'])
        plan = read(plan_path)
        folder, out = plan_path.parent, plan_path.parent / 'run'
        for item in plan['sources'] + plan['inputs']:
            assert pin(item['path']) == item, item['path']
        identity = read(folder / 'controller-first.json')
        terminal_path = folder / 'actual-tool-terminal-receipt-first.json'
        terminal = read(terminal_path)
        assert terminal['actual_tool_terminal_collected'] and terminal['exit_code'] == 0
        assert terminal['actual_tool_terminal_chunk'] and terminal['reaped'] and terminal['all_children_reaped']
        assert terminal['pid'] == identity['pid'] and terminal['birth'] == identity['birth']
        assert terminal['command'] == identity['argv'] and absent(identity['pid'], identity['birth'])
        completion = read(folder / 'completion-first.json')
        assert completion['complete'] and completion['steps'] == 1024
        assert completion['all4quarters_and3strengths_finished'] and not completion['weight_EMA']
        assert not completion['holdout_access'] and not completion['selection_performed']
        assert not (folder / 'controller-failed.json').exists() and not (out / 'candidate-failed.json').exists()
        executions = sorted((folder / 'controller-executions').glob('phase-*.json'))
        executions = [p for p in executions if '.terminal.' not in p.name]
        assert len(executions) == 13
        for path in executions:
            child, end = read(path), read(path.with_suffix('.terminal.json'))
            assert end['returncode'] == 0 and end['reaped']
            assert not Path(f'/proc/{child["pid"]}').exists()
        for phase, steps in [('initialize', [0]), ('train', [256, 512, 768, 1024]), ('evaluate', [256, 512, 768, 1024])]:
            for step in steps:
                end = read(out / f'{phase}-{step:06d}-first-resources.completed.json')
                assert end['state'] == 'complete' and end['failure'] is None
        with (out / '.shared-gpu-job.lock').open('r+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lease, fcntl.LOCK_UN)
        admissions = []
        for step in [256, 512, 768, 1024]:
            audit_path = BASE / f'GPU{cohort["gpu"]}-quarter{step}-independent-admission-first.json'
            audit = read(audit_path)
            assert audit['state'] == 'declared-complete-behavior-quarter-independently-readmitted'
            assert audit['step'] == step and audit['plan'] == cohort['plan']
            assert audit['all48_answers_native_EOS'] and audit['all48_native_inputs_outputs_typed_Jev_scores_verified']
            assert audit['no_weight_EMA'] and audit['RAW_FULL_NATIVE_network_equality']
            assert audit['training_pairs'] == declaration['training_pairs']
            assert [a['strength'] for a in audit['arms']] == [.25, .5, 1.]
            for key in ['source', 'FULL', 'RAW', 'NATIVE', 'qualified_checkpoint_guard']:
                assert pin(audit[key]['path']) == audit[key]
            for value in audit['arms']:
                assert pin(value['predictions']['path']) == value['predictions']
                assert pin(value['summary']['path']) == value['summary']
                rows = read_jsonl(value['predictions']['path'])
                assert [r['id'] for r in rows] == [r['id'] for r in controls['off']]
                mean = sum(Decimal(str(r['scores']['effect'])) for r in rows) / 64
                gains = {name: sum((Decimal(str(r['scores']['effect'])) - Decimal(str(b['scores']['effect']))
                    for r, b in zip(rows, baseline)), Decimal(0)) / 64 for name, baseline in controls.items()}
                arms.append({**value, 'name': f'PG-GPU{cohort["gpu"]}-step{step}-alpha{value["strength"]:g}',
                    'gpu': cohort['gpu'], 'g_lr': cohort['g_lr'], 'plan': cohort['plan'],
                    'mean_effect': float(mean), 'mean_effect_exact_decimal': str(mean),
                    'effect_ON_minus_OFF': float(gains['off']), 'effect_ON_minus_OFF_exact_decimal': str(gains['off']),
                    'effect_ON_minus_prompt_only': float(gains['prompt-only']), 'quarter_admission': pin(audit_path)})
            admissions.append(pin(audit_path))
        cohorts.append({'plan': cohort['plan'], 'actual_terminal': pin(terminal_path),
            'completion': pin(folder / 'completion-first.json'), 'quarters': admissions})
    assert len(arms) == 24 and len({a['name'] for a in arms}) == 24
    maximum = max(Decimal(a['effect_ON_minus_OFF_exact_decimal']) for a in arms)
    ties = [a for a in arms if Decimal(a['effect_ON_minus_OFF_exact_decimal']) == maximum]
    report = {'time': time.time(), 'source': pin(__file__), 'state': 'promptfix-two-rate-24-cell-DEV-search-complete',
        'declaration': pin(BASE / 'training-declaration-first.json'), 'cohorts': cohorts, 'all24_arms': arms,
        'all_maximum_ties': ties, 'maximum_DEV_effect_gain': float(maximum),
        'selector': 'equal-question mean full finished intended effect ON minus matched OFF only',
        'display_checkpoint_representative': ties[0],
        'display_rule': 'First in declared GPU, step, strength order; no superior tie claim',
        'exact_strength_health_still_required': True, 'training_pairs': declaration['training_pairs'],
        'fresh64_still_sealed': True, 'fresh_holdout_manifest': pin(DIAG / 'new-confirmation-data/manifest-first.json'),
        'holdout_rows_read': 0, 'new_model_calls': 0, 'new_Jev_calls': 0, 'new_updates': 0,
        'controls_reused_from_audited_diagnostic': pin(DIAG / 'independent-audit-first.json')}
    output = BASE / 'development-search-closure-first.json'
    assert not output.exists()
    write_json(output, report)
    print(json.dumps({'closure': pin(output), 'maximum_DEV_gain': float(maximum),
        'ties': [a['name'] for a in ties]}), flush=True)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    close()
