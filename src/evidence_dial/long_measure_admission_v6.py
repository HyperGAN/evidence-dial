"""Execution gates layered over the CPU-qualified long-checkpoint admission.

Match actual root identities and require a completed same-data SFT control
before confirming a PG model trained on the newly expanded TRAIN512 pool.
No model, prompt, scorer, decoding, or training formulation changes.
"""
import json

from .long_measure_admission_v5 import (
    ROOT, LOCKED_DATA_SHA, BACKENDS, pin, verified, backend_for, validate_arm,
    finished_PG_cells, validate_completed_study,
    validate_confirmation_plan as validate_confirmation_plan_v1,
)


def require_root_identity(closure, expected):
    assert closure['actual_tool_terminal_collected'] and closure['reaped'] and closure['exit_code'] == 0
    assert closure['pid'] == expected['pid']
    assert closure['birth'] == expected['birth']
    assert closure['command'] == expected['argv']
    assert closure['tool_handle'] == expected['tool_handle']


def require_matched_data_control(plan, registry):
    if not any(arm['kind'] == 'particle' and arm['backend'] == 'curated512' for arm in plan['arms']):
        return False
    rule = registry['conditional_TRAIN512_SFT_control']
    from .curated512_native_v1 import TRAIN_SHA
    assert rule['training_data']['sha256'] == TRAIN_SHA
    control = json.loads(verified(plan['matched_data_SFT_control']).read_text())
    assert control['state'] == 'complete-RAW-SFT512-control-and-DEV-selection'
    assert control['training_data'] == rule['training_data']
    recipe = json.loads(verified(control['recipe']).read_text())
    assert {key: recipe[key] for key in ['steps', 'batch', 'lr', 'weight_ema', 'scheduler']} == rule['training']
    assert recipe['train_cases'] == 512
    closure = json.loads(verified(control['actual_root_closure']).read_text())
    require_root_identity(closure, control['actual_identity_at_launch'])
    for item in control['finished_DEV_audits']:
        verified(item)
    assert len(control['finished_DEV_audits']) == 4
    selection = json.loads(verified(control['completed_DEV_selection']).read_text())
    assert selection['training_budget_completed'] == rule['training']['steps']
    assert selection['all_declared_quarters_measured_and_independently_audited']
    assert selection['actual_controller_exit_code'] == 0
    strongest = selection['selected_DEV_ties']
    assert strongest and all(a['cases'] == 128 for a in strongest)
    lock = json.loads(verified(plan['selection_lock']).read_text())
    best = max(a['correct'] for a in lock['sft_finished_DEV_scores'])
    assert best >= max(a['correct'] for a in strongest)
    if best == max(a['correct'] for a in strongest):
        assert {a['raw_artifact']['sha256'] for a in strongest} <= {
            a['artifact']['sha256'] for a in plan['arms'] if a['kind'] == 'sft'}
    return True


def validate_confirmation_plan(path):
    plan, admitted = validate_confirmation_plan_v1(path)
    lock = json.loads(verified(plan['selection_lock']).read_text())
    registry = json.loads(verified(lock['required_studies_registry']).read_text())
    required = {r['study_id']: r for r in registry['required_studies']}
    for study in lock['completed_studies']:
        closure = json.loads(verified(study['actual_root_closure']).read_text())
        require_root_identity(closure, required[study['study_id']]['actual_identity_at_registry_freeze'])
    require_matched_data_control(plan, registry)
    return plan, admitted
