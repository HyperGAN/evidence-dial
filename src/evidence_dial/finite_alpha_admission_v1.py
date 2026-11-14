"""Require the complete declared finite slider studies before a test lock.

This reads retained evidence only. It cannot generate, train, choose per-case
answers, or weaken the original training or same-data SFT requirements.
"""
import json
from pathlib import Path

from .benchmark import file_sha

ROOT = Path(__file__).resolve().parents[2]
AUXILIARY_REGISTRY = ROOT / 'artifacts/fixed-formulation-pursuit-20261112-v1/long-checkpoint-confirmation-execution-v1/required-finite-alpha-studies-registry-first.json'


def pin(path):
    path = (ROOT / path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def verified(item):
    assert set(item) == {'path', 'sha256'}
    path = (ROOT / item['path']).resolve()
    assert pin(path) == item
    return path


def completed_auxiliary_records():
    registry = json.loads(AUXILIARY_REGISTRY.read_text())
    assert registry['state'] == 'two-finite-alpha-studies-required-before-final-DEV-lock'
    assert len(registry['required_auxiliary_studies']) == 2
    records = []
    for required in registry['required_auxiliary_studies']:
        inventory = json.loads(verified(required).read_text())
        plan = verified(inventory['plan'])
        path = plan.parent / 'completed-auxiliary-study-first.json'
        value = json.loads(path.read_text())
        assert value['auxiliary_inventory'] == required
        records.append(value)
    assert len({r['study_id'] for r in records}) == 2
    return records


def validate_completed_auxiliary_study(study, required):
    from .long_measure_admission_v5 import finished_PG_cells
    inventory = json.loads(verified(required).read_text())
    assert inventory['state'] == 'required-auxiliary-development-study-declared-before-new-generations'
    assert inventory['no_weight_EMA'] is True
    assert inventory['new_training_updates'] == inventory['teacher_calls'] == inventory['test_rows_parsed'] == 0
    assert study['auxiliary_inventory'] == required
    assert study['study_id'] == inventory['study_id']
    assert study['state'] == 'complete-RAW-checkpoint-finite-alpha-DEV-study'
    assert study['plan'] == inventory['plan']
    plan_path = verified(study['plan'])
    plan = json.loads(plan_path.read_text())
    for item in plan['inputs'] + plan['sources']:
        verified(item)
    preflight = json.loads(verified(inventory['CPU_preflight']).read_text())
    assert preflight['new_model_calls'] == preflight['new_training_updates'] == preflight['test_rows_parsed'] == 0
    assert preflight['CUDA_initialized'] is False
    assert preflight['source'] in plan['sources'] and preflight['worker'] in plan['sources']
    assert plan['ema_decay'] is None and plan['elapsed_work_deadline'] is None
    assert plan['cases'] == 128 and plan['strengths'] == inventory['strengths']
    assert plan['new_training_updates'] == plan['teacher_calls'] == plan['test_rows_read'] == 0
    assert plan['new_native_generation_budget'] == inventory['new_native_generation_budget'] == 1024
    assert len(plan['candidates']) == 1 and len(plan['strengths']) == 8
    assert len(set(plan['strengths'])) == 8
    candidate = plan['candidates'][0]
    assert len(candidate['coarse_arms']) == 10
    assert set(plan['strengths']).isdisjoint(a['strength'] for a in candidate['coarse_arms'])
    identity = json.loads(verified(study['actual_live_binding']).read_text())
    assert identity['actual_OS_root_and_child_relationship_verified']
    assert identity['plan'] == study['plan'] and identity['auxiliary_inventory'] == required
    closure = json.loads(verified(study['actual_root_closure']).read_text())
    assert closure['actual_tool_terminal_collected'] and closure['reaped'] and closure['exit_code'] == 0
    assert closure['pid'] == identity['pid'] and closure['birth'] == identity['birth']
    assert closure['command'] == identity['argv'] and closure['tool_handle'] == identity['tool_handle']
    assert closure['parent_plan_sha256'] == study['plan']['sha256']
    assert closure['private_controller_and_job_leases_released']
    command = identity['argv']
    assert command[-2:] == ['--phase', 'controller']
    assert (ROOT / command[command.index('--plan') + 1]).resolve() == plan_path
    assert pin(command[1]) in plan['sources']
    state = json.loads(verified(study['completion_state']).read_text())
    assert state['state'] == 'all-declared-DEV-arms-finished'
    assert state['new_training_updates'] == state['test_generations'] == 0
    assert plan['quarter_completed_arms'] == inventory['quarter_completed_arms'] == [2, 4, 6, 8]
    assert len(study['quarter_markers']) == 4
    for quarter, item in enumerate(study['quarter_markers'], 1):
        marker = json.loads(verified(item).read_text())
        assert marker['plan'] == study['plan']
        assert marker['completed_declared_arms'] == 2 * quarter and marker['fraction'] == quarter / 4
        assert marker['new_training_updates'] == marker['test_generations'] == 0
    audit_path = verified(study['independent_completed_DEV_audit'])
    audit = json.loads(audit_path.read_text())
    assert audit['complete'] and audit['weight_EMA'] is False
    assert audit['plan'] == study['plan']
    assert audit['auditor_source_sha256'] == file_sha(ROOT / 'tools/audit_saved_pg_alpha_local_v2.py')
    assert audit['new_model_calls_for_audit'] == audit['new_training_updates'] == audit['test_generations'] == 0
    new = [a for a in audit['DEV_arms'] if a['reused_completed_native_measurement'] is False]
    assert [a['strength'] for a in new] == plan['strengths']
    assert all(a['step'] == candidate['step'] for a in audit['DEV_arms'])
    cells = finished_PG_cells(study['independent_completed_DEV_audit'])
    assert len(cells) == 18 and len({(c['artifact']['sha256'], c['strength']) for c in cells}) == 18
    return cells


def completed_auxiliary_cells(lock):
    assert lock['required_auxiliary_registry'] == pin(AUXILIARY_REGISTRY)
    registry = json.loads(AUXILIARY_REGISTRY.read_text())
    training = json.loads(verified(lock['required_studies_registry']).read_text())
    # The alpha studies were declared against the preserved first five studies.
    # A later exact continuation must retain that entire inventory unchanged.
    original_pin = registry['required_training_registry']
    if lock['required_studies_registry'] != original_pin:
        assert training['parent_registry'] == original_pin
        parent = json.loads(verified(original_pin).read_text())
        assert training['required_studies'][:len(parent['required_studies'])] == parent['required_studies']
        assert training['mandatory_SFT_maximum_registry'] == parent['mandatory_SFT_maximum_registry']
        assert training['conditional_TRAIN512_SFT_control'] == parent['conditional_TRAIN512_SFT_control']
    required = registry['required_auxiliary_studies']
    assert len(required) == 2
    records = lock['completed_auxiliary_development_studies']
    assert len(records) == 2 and len({r['study_id'] for r in records}) == 2
    expected = {json.loads(verified(item).read_text())['study_id']: item for item in required}
    assert {r['study_id'] for r in records} == set(expected)
    cells = []
    for record in records:
        inventory = json.loads(verified(expected[record['study_id']]).read_text())
        assert inventory['required_five_training_studies_registry'] == original_pin
        cells += validate_completed_auxiliary_study(record, expected[record['study_id']])
    return cells


def auxiliary_input_pins():
    values = [pin(AUXILIARY_REGISTRY)]
    registry = json.loads(AUXILIARY_REGISTRY.read_text())
    for required in registry['required_auxiliary_studies']:
        inventory = json.loads(verified(required).read_text())
        folder = verified(inventory['plan']).parent
        values += [required, pin(folder / 'completed-auxiliary-study-first.json')]
    return values
