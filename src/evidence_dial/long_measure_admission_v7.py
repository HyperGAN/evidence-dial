"""Exact saved-checkpoint admission for future DEV-locked confirmation.

Validation is CPU-only. No evaluation rows or model are loaded here; the
worker must validate the complete selection lock before reading test rows.
"""
import importlib
import json
import math
from pathlib import Path
import re

from .benchmark import file_sha

ROOT = Path(__file__).resolve().parents[2]
LOCKED_DATA_SHA = '77a30c471b825ba8c7f47db344b5c8d4aa4a1e031baa8afc97fa2284e81e2d09'
BACKENDS = {
    'curated256': 'evidence_dial.curated_first_answer_binding_v1',
    'curated512': 'evidence_dial.curated512_native_v1',
    'sparse256': 'evidence_dial.sparse_nonterminal_native_v1',
    'first-answer': 'evidence_dial.first_answer_binding_v2',
    'canonical_stop256': 'evidence_dial.canonical_stop_prefix_native_v1',
    'canonical_stop512': 'evidence_dial.canonical_stop512_native_v1',
}


def pin(path):
    path = (ROOT / path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def verified(value):
    assert set(value) == {'path', 'sha256'}
    path = (ROOT / value['path']).resolve()
    assert file_sha(path) == value['sha256'], path
    return path


def backend_for(arm):
    assert arm['backend'] in BACKENDS
    return importlib.import_module(BACKENDS[arm['backend']]).inference_backend('first-answer')


def finished_PG_cells(audit_record):
    """Read a verified independent audit's complete effect cells, without forwards."""
    audit = json.loads(verified(audit_record).read_text())
    allowed = ['audit_pg_declared_quarter_v1.py', 'audit_pg_continuation_quarter_v1.py',
        'audit_pg_continue512_quarter_v1.py', 'audit_saved_pg_alpha_local_v1.py', 'audit_saved_pg_alpha_local_v2.py']
    assert audit['auditor_source_sha256'] in {file_sha(ROOT / 'tools' / name) for name in allowed}
    assert audit.get('no_weight_EMA') is True or audit.get('weight_EMA') is False
    if audit.get('native_artifact'):
        assert audit['all_declared_DEV_strengths_finished'] and audit['qualified_guard']
        verified(audit['qualified_guard'])
        candidates = {audit['milestone']: {'artifact': audit['native_artifact'], 'FULL': audit['FULL']}}
    else:
        assert audit['complete']
        alpha_plan = json.loads(verified(audit['plan']).read_text())
        candidates = {candidate['step']: candidate for candidate in alpha_plan['candidates']}
    cells = []
    for arm in audit['DEV_arms']:
        step = arm.get('step', audit.get('milestone'))
        candidate = candidates[step]
        for item in [candidate['artifact'], candidate['FULL'], arm['predictions'], arm['summary']]:
            verified(item)
        assert arm['all128_native_ordinary_prompt_input_output_tokens_verified']
        assert arm['cases'] == 128 and arm['OFF_correct'] == 67
        assert arm['effect_ON_minus_matched_OFF'] == (arm['correct'] - 67) / 128
        cells.append({'artifact': candidate['artifact'], 'step': step, 'strength': arm['strength'],
            'correct': arm['correct'], 'cases': 128, 'OFF_correct': 67,
            'effect_ON_minus_OFF': (arm['correct'] - 67) / 128, 'predictions': arm['predictions']})
    return cells


def validate_completed_study(study, required):
    """Check every declared PG checkpoint and actual root closure, not a flag."""
    assert study['study_id'] == required['study_id']
    assert study['recipe'] == required['recipe']
    recipe_path = verified(study['recipe'])
    recipe = json.loads(recipe_path.read_text())
    for item in recipe['frozen_inputs'] + recipe['sources']:
        verified(item)
    assert recipe['training']['weight_ema'] is None
    assert recipe['training']['steps'] == required['budget']
    assert recipe['development_checkpoints'] == required['development_checkpoints']
    closure = json.loads(verified(study['actual_root_closure']).read_text())
    assert closure['actual_tool_terminal_collected'] and closure['reaped'] and closure['exit_code'] == 0
    assert closure['parent_recipe_sha256'] == study['recipe']['sha256']
    declaration = json.loads(verified(study['declaration']).read_text())
    assert str(verified(study['declaration'])) in closure['command']
    for item in declaration['inputs'] + declaration['sources']:
        verified(item)
    completion = json.loads(verified(study['completion_state']).read_text())
    assert completion['state'] == 'training-and-development-complete'
    assert [a['step'] for a in study['finished_DEV_audits']] == recipe['development_checkpoints']
    allowed_auditors = [ROOT / 'tools' / name for name in ['audit_pg_declared_quarter_v1.py',
        'audit_pg_continuation_quarter_v1.py', 'audit_pg_continue512_quarter_v1.py']]
    cells = []
    for record in study['finished_DEV_audits']:
        audit = json.loads(verified(record['audit']).read_text())
        assert audit['milestone'] == record['step'] and audit['budget'] == required['budget']
        assert audit['recipe'] == study['recipe'] and audit['no_weight_EMA']
        assert audit['all_declared_DEV_strengths_finished'] and audit['qualified_guard']
        assert audit['auditor_source_sha256'] in {file_sha(p) for p in allowed_auditors}
        assert [a['strength'] for a in audit['DEV_arms']] == recipe['new_model_development_strengths']
        for key in ['FULL', 'raw_export', 'native_artifact', 'qualified_guard']:
            verified(audit[key])
        for arm in audit['DEV_arms']:
            assert arm['cases'] == 128 and arm['OFF_correct'] == 67
            assert arm['effect_ON_minus_matched_OFF'] == (arm['correct'] - 67) / 128
            for key in ['predictions', 'summary']:
                verified(arm[key])
        cells += finished_PG_cells(record['audit'])
    assert study['finished_DEV_audits'][-1]['step'] == required['budget']
    return cells


def validate_arm(arm, *, require_strength_guard=True):
    """Admit actual RAW PG or SFT weights at the explicitly named saved step."""
    import torch
    from .qualified_text_guard_admission_v1 import passing_guard
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', arm['name'])
    assert arm['kind'] in ['off', 'particle', 'sft']
    if arm['kind'] == 'off':
        assert set(arm) == {'name', 'kind'} and arm['name'] == 'off'
        return {'kind': 'off', 'weights': 'raw-pretrained', 'new_model_calls': 0}
    step = arm['step']
    assert type(step) is int and step > 0
    artifact = verified(arm['artifact'])
    candidate = (ROOT / arm['candidate']).resolve()
    guard = verified(arm['checkpoint_guard'])
    audit_path = verified(arm['development_audit'])
    audit = json.loads(audit_path.read_text())
    predictions = verified(arm['development_predictions'])
    assert audit.get('no_weight_EMA') is True or audit.get('weight_EMA') is False
    if arm['kind'] == 'particle':
        from particle_sliders.lifecycle import read_packet
        from particle_sliders.inference import read_inference
        from .train import tensor_equal
        full = verified(arm['FULL'])
        assert artifact.parent == full.parent == candidate
        assert passing_guard(candidate, step, full) == guard
        packet, native = read_packet(full), read_inference(artifact)
        game = packet['payload']['game']
        assert game['completed_steps'] == native['origin']['completed_steps'] == step
        assert game['config']['recipe']['ema_decay'] == 0.
        assert packet['payload']['ema'] == {'decay': None, 'values': None}
        assert native['selection'] == 'raw'
        assert tensor_equal(native['network'], game['network'])
        factory = backend_for(arm)
        assert native['specification']['backend'] == factory.name
        assert native['specification']['configuration'] == {'mode': 'first-answer', 'rank': 8,
            'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None}
        factory.validate({}, __import__('particle_sliders.inference', fromlist=['InferenceSpec']).InferenceSpec(native['specification']))
        strength = arm['strength']
        assert isinstance(strength, (int, float)) and math.isfinite(strength) and strength > 0
        audit_arms = audit.get('DEV_arms', audit.get('new_strength_arms', []))
        eligible = [a for a in audit_arms if a['strength'] == strength and a['predictions'] == arm['development_predictions']
            and a.get('step', audit.get('milestone')) == step]
        assert len(eligible) == 1
        assert eligible[0]['cases'] == 128 and eligible[0]['OFF_correct'] == 67
        if audit.get('native_artifact'):
            assert audit['native_artifact'] == arm['artifact'] and audit['FULL'] == arm['FULL']
        else:
            plan = json.loads(verified(audit['plan']).read_text())
            chosen = next(c for c in plan['candidates'] if c['step'] == step)
            assert chosen['artifact'] == arm['artifact'] and chosen['FULL'] == arm['FULL']
        if require_strength_guard:
            selected = verified(arm['selected_strength_guard'])
            assert passing_guard(selected.parent, step, full) == selected
            session = json.loads((selected.parent / 'session.json').read_text())
            assert session['artifact'] == arm['artifact'] and session['FULL'] == arm['FULL']
            assert session['step'] == step and session['strength'] == strength
            assert session['development_predictions'] == arm['development_predictions']
            from .io import read_jsonl
            records = read_jsonl(predictions)
            assert len(records) == len({r['id'] for r in records}) == 128
            probes = json.loads((selected.parent / f'probes-{step:06d}.json').read_text())['probes']
            expected = [next(r for r in records if r['case_type'] == kind)
                for kind in ['answerable', 'missing', 'conflict', 'altered']]
            assert probes == expected
        result = {'kind': 'particle', 'step': step, 'strength': strength,
            'correct_DEV': eligible[0]['correct'], 'weights': 'raw', 'FULL_native_network_exact': True}
    else:
        from safetensors.torch import load_file
        assert artifact.name == 'adapter_model.safetensors'
        assert artifact.parent.name == f'adapter-{step:06d}' and artifact.parent.parent == candidate
        assert passing_guard(candidate, step, artifact) == guard
        assert audit['milestone'] == step and audit['artifact'] == arm['artifact']
        assert audit['saved_raw_adapter_matches_checkpoint_bitwise']
        assert audit['DEV']['predictions'] == arm['development_predictions']
        assert audit['DEV']['cases'] == 128 and audit['DEV']['OFF_correct'] == 67
        checkpoint = candidate / f'checkpoint-{step}'
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        assert state['global_step'] == step
        actual, original = load_file(str(artifact)), load_file(str(checkpoint / 'adapter_model.safetensors'))
        assert actual.keys() == original.keys() and all(torch.equal(actual[k], original[k]) for k in actual)
        config = json.loads(verified(arm['adapter_config']).read_text())
        assert Path(arm['adapter_config']['path']).parent == artifact.parent
        assert config['r'] == config['lora_alpha'] == 8 and config['inference_mode'] is True
        assert set(config['target_modules']) == {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
        assert not config.get('use_dora') and not config.get('use_rslora')
        session = json.loads((candidate / 'session.json').read_text())
        assert session['ema_decay'] is None
        result = {'kind': 'sft', 'step': step, 'correct_DEV': audit['DEV']['correct'],
            'weights': 'raw', 'adapter_exact_saved_checkpoint': True}
    assert not torch.cuda.is_initialized(), 'Admission must not initialize CUDA'
    return {**result, 'new_model_calls': 0, 'heldout_rows_read': 0}


def validate_confirmation_plan(path):
    """Verify a complete, immutable DEV lock before any test rows are opened."""
    from .protocol import digest
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    assert plan['schema'] == 'long-checkpoint-locked-confirmation-v1'
    assert plan['cases'] == 256 and plan['split'] == 'locked-confirmation'
    assert plan['ema_decay'] is None and plan['elapsed_work_deadline'] is None
    for item in plan['inputs'] + plan['sources']:
        verified(item)
    assert verified(plan['data']).is_file() and plan['data']['sha256'] == LOCKED_DATA_SHA
    lock = json.loads(verified(plan['selection_lock']).read_text())
    assert lock['state'] == 'all-declared-studies-complete-DEV-only-selection-locked'
    assert lock['holdout_generated_before_freeze'] is False
    assert lock['data'] == plan['data']
    assert lock['arm_descriptor_digest'] == digest(plan['arms'])
    assert lock['selector'] == 'equal-case-finished-DEV-ON-minus-matched-OFF-all-ties'
    assert len({a['name'] for a in plan['arms']}) == len(plan['arms'])
    assert [a for a in plan['arms'] if a['kind'] == 'off'] == [{'name': 'off', 'kind': 'off'}]
    # No unfinished declared budget can be silently omitted by a worker.
    registry = json.loads(verified(lock['required_studies_registry']).read_text())
    required = {r['study_id']: r for r in registry['required_studies']}
    assert len(required) == len(registry['required_studies'])
    assert set(required) == {r['study_id'] for r in lock['completed_studies']}
    assert len(lock['completed_studies']) == len(required)
    eligible_cells = []
    for study in lock['completed_studies']:
        eligible_cells += validate_completed_study(study, required[study['study_id']])
    for record in registry['retained_finished_PG_audits']:
        eligible_cells += finished_PG_cells(record)
    from .finite_alpha_admission_v1 import completed_auxiliary_cells
    eligible_cells += completed_auxiliary_cells(lock)
    expected_cells = {}
    for cell in eligible_cells:
        key = (cell['artifact']['sha256'], cell['strength'])
        if key in expected_cells:
            assert expected_cells[key] == cell
        expected_cells[key] = cell
    locked_cells = {(cell['artifact']['sha256'], cell['strength']): cell for cell in lock['particle_finished_DEV_scores']}
    assert len(locked_cells) == len(lock['particle_finished_DEV_scores'])
    assert locked_cells.keys() == expected_cells.keys(), 'Every eligible finished PG cell must remain in the lock'
    for key, cell in expected_cells.items():
        assert {k: locked_cells[key][k] for k in cell} == cell
    # The lock names every exact maximum; the worker has no reselection path.
    for kind, key in [('particle', 'selected_PG_ties'), ('sft', 'selected_SFT_ties')]:
        assert {a['name'] for a in plan['arms'] if a['kind'] == kind} == set(lock[key])
        scores = lock[kind + '_finished_DEV_scores']
        assert scores and len({a['name'] for a in scores}) == len(scores)
        assert all(a['cases'] == 128 and a['OFF_correct'] == 67 and
            a['effect_ON_minus_OFF'] == (a['correct'] - 67) / 128 for a in scores)
        best = max(a['effect_ON_minus_OFF'] for a in scores)
        assert set(lock[key]) == {a['name'] for a in scores if a['effect_ON_minus_OFF'] == best}
    admitted = {a['name']: validate_arm(a) for a in plan['arms']}
    for kind in ['particle', 'sft']:
        scores = {a['name']: a for a in lock[kind + '_finished_DEV_scores']}
        for arm in plan['arms']:
            if arm['kind'] == kind:
                assert admitted[arm['name']]['correct_DEV'] == scores[arm['name']]['correct']
    controls = json.loads(verified(registry['mandatory_SFT_maximum_registry']).read_text())
    assert controls['actual_controller_exit_code'] == 0
    assert controls['all_declared_quarters_measured_and_independently_audited']
    mandatory = controls['selected_DEV_ties']
    best_control = max(a['correct'] for a in lock['sft_finished_DEV_scores'])
    assert best_control >= max(a['correct'] for a in mandatory), 'Do not weaken the strongest completed SFT'
    if best_control == max(a['correct'] for a in mandatory):
        required_weights = {a['raw_artifact']['sha256'] for a in mandatory}
        assert required_weights <= {a['artifact']['sha256'] for a in plan['arms'] if a['kind'] == 'sft'}
    physical = {0: 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47', 1: 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'}
    assert plan['gpu'] in physical and plan['physical_gpu_uuid'] == physical[plan['gpu']]
    assert plan['new_training_updates'] == plan['teacher_calls'] == 0
    return plan, admitted
