"""Admit the explicitly declared lower-LR fork only after actual import."""
import json
from pathlib import Path

from .benchmark import file_sha

ROOT = Path(__file__).resolve().parents[2]
PARENT_REGISTRY = ROOT / 'artifacts/fixed-formulation-pursuit-20261112-v1/long-checkpoint-confirmation-execution-v1/required-seven-studies-retained-stop512-alpha-registry-first.json'


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def verified(item):
    assert pin(item['path']) == item
    return Path(item['path'])


def verify_execution_derivative(folder):
    folder = Path(folder).resolve()
    value = json.loads((folder / 'training-execution-derivative-v1-first.json').read_text())
    assert value['state'] == 'qualified-native-LR-fork-execution-before-first-new-update'
    assert value['recipe'] == pin(folder / 'recipe.json')
    for item in value['sources']:
        verified(item)
    admission = json.loads(verified(value['actual_native_import_qualification']).read_text())
    assert admission['state'] == 'actual-native-LR-fork-import-qualified-before-first-new-SGD'
    assert admission['recipe'] == value['recipe'] and admission['inherited_updates'] == 256
    assert admission['new_training_updates'] == admission['new_native_forwards_in_import'] == 0
    assert admission['all_weights_optimizer_history_scalar_statistics_RNG_streams_exposures_and_metrics_preserved']
    assert admission['only_declared_current_LRs_and_future_budget_metadata_changed']
    assert admission['qualified_parent256_guard_transfer_verified']
    for key in ['FULL256', 'raw256', 'native256', 'initialization', 'native_proof', 'actual_import']:
        verified(admission[key])
    assert value['original_native_import_recipe_and_session_preserved']
    assert value['no_initialization_or_completed_update_replay']
    return value


def verify_sgd_admission(folder, root_pid):
    folder = Path(folder).resolve()
    derivative = verify_execution_derivative(folder)
    token = json.loads((folder / 'SGD-admission-first.json').read_text())
    assert token['state'] == 'qualified-LR-fork-eight-study-inventory-before-update257'
    assert token['recipe'] == derivative['recipe']
    assert token['declaration'] == pin(folder / 'queue-declaration-shared-v1-first.json')
    assert token['execution_derivative'] == pin(folder / 'training-execution-derivative-v1-first.json')
    owned = json.loads(verified(token['actual_owned_process']).read_text())
    assert owned['pid'] == root_pid
    proc = Path(f'/proc/{root_pid}')
    assert proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == owned['birth']
    assert proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1] == owned['argv']
    parent = json.loads(PARENT_REGISTRY.read_text())
    registry = json.loads(verified(token['required_registry']).read_text())
    assert registry['training_admission_parent_registry_eight'] == pin(PARENT_REGISTRY)
    assert registry['required_studies'][:7] == parent['required_studies'] and len(registry['required_studies']) == 8
    for key in ['retained_finished_PG_audits', 'mandatory_SFT_maximum_registry', 'conditional_TRAIN512_SFT_control',
                'required_auxiliary_studies_registry', 'retained_complete_stop_alpha_study', 'retained_complete_stop512_alpha_study']:
        assert registry[key] == parent[key]
    assert registry['eight_study_confirmation_execution_qualification_required']
    study = registry['required_studies'][7]
    assert study['recipe'] == token['recipe'] and study['declaration'] == token['declaration']
    assert study['budget'] == 384 and study['exact_inherited_updates'] == 256 and study['declared_new_updates'] == 128
    assert study['development_checkpoints'] == [288, 384] and study['backend'] == 'canonical_stop256'
    assert study['actual_native_import_qualification'] == derivative['actual_native_import_qualification']
    assert study['execution_derivative'] == token['execution_derivative']
    actual = study['actual_identity_at_registry_freeze']
    assert actual['pid'] == owned['pid'] and actual['birth'] == owned['birth'] and actual['argv'] == owned['argv']
    assert actual['tool_handle'] == token['actual_root_tool_handle']
    assert token['new_training_updates_before_gate'] == token['test_rows_parsed'] == 0
    return registry
