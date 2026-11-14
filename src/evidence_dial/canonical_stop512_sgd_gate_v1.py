"""Admit TRAIN512 stopping-data SGD after actual native0 and seven-study freeze."""
import json
from pathlib import Path

from .benchmark import file_sha

ROOT = Path(__file__).resolve().parents[2]
PARENT_REGISTRY = ROOT / 'artifacts/fixed-formulation-pursuit-20261112-v1/long-checkpoint-confirmation-execution-v1/required-six-studies-retained-stop256-alpha-registry-first.json'


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def verified(item):
    assert pin(item['path']) == item
    return Path(item['path'])


def verify_execution_derivative(folder):
    folder = Path(folder).resolve()
    path = folder / 'training-execution-derivative-v1-first.json'
    derivative = json.loads(path.read_text())
    assert derivative['state'] == 'native0-qualified-execution-only-training-derivative'
    assert derivative['recipe'] == pin(folder / 'recipe.json')
    for item in derivative['sources']:
        verified(item)
    native0 = json.loads(verified(derivative['native0_qualification']).read_text())
    assert native0['state'] == 'actual-native0-qualified-before-first-SGD'
    assert native0['recipe'] == derivative['recipe']
    for key in ['FULL0', 'raw0', 'native0_artifact', 'native_proof', 'unchanged_formulation_proof', 'initialization']:
        verified(native0[key])
    assert native0['ordinary_late_complete_answer_ON_equals_OFF_cache']
    assert native0['all_three_contexts_2593_connected_finite_gradient_tensors'] and native0['exact_FULL_restore']
    assert native0['new_training_updates'] == 0 and native0['weight_EMA'] is False
    assert derivative['original_native0_recipe_and_session_config_preserved']
    assert derivative['no_fresh_initialization_or_native0_replay']
    return derivative


def verify_sgd_admission(folder, root_pid):
    folder = Path(folder).resolve()
    derivative = verify_execution_derivative(folder)
    token = json.loads((folder / 'SGD-admission-first.json').read_text())
    assert token['state'] == 'native0-qualified-seven-study-inventory-frozen-before-SGD'
    assert token['recipe'] == pin(folder / 'recipe.json')
    assert token['declaration'] == pin(folder / 'queue-declaration-shared-v1-first.json')
    assert token['execution_derivative'] == pin(folder / 'training-execution-derivative-v1-first.json')
    owned = json.loads(verified(token['actual_owned_process']).read_text())
    assert owned['pid'] == root_pid
    proc = Path('/proc') / str(root_pid)
    assert proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == owned['birth']
    assert [x.decode() for x in proc.joinpath('cmdline').read_bytes().split(b'\0') if x] == owned['argv']
    parent = json.loads(PARENT_REGISTRY.read_text())
    registry = json.loads(verified(token['required_registry']).read_text())
    assert registry['training_admission_parent_registry'] == pin(PARENT_REGISTRY)
    assert registry['required_studies'][:6] == parent['required_studies']
    assert len(registry['required_studies']) == 7
    for key in ['retained_finished_PG_audits', 'mandatory_SFT_maximum_registry',
                'conditional_TRAIN512_SFT_control', 'required_auxiliary_studies_registry',
                'retained_complete_stop_alpha_study']:
        assert registry[key] == parent[key], key
    assert registry['conditional_TRAIN512_backend_names'] == ['curated512', 'canonical_stop512']
    assert registry['confirmation_requires_seven_study_execution_qualification']
    assert registry['old_six_study_confirmation_profile_not_current']
    study = registry['required_studies'][6]
    assert study['recipe'] == token['recipe'] and study['declaration'] == token['declaration']
    assert study['budget'] == 1024 and study['development_checkpoints'] == [256, 512, 768, 1024]
    assert study['backend'] == 'canonical_stop512' and study['training_data'] == registry['conditional_TRAIN512_SFT_control']['training_data']
    assert study['native0_qualification'] == token['native0_qualification'] == derivative['native0_qualification']
    assert study['execution_derivative'] == token['execution_derivative']
    actual = study['actual_identity_at_registry_freeze']
    assert actual['pid'] == owned['pid'] and actual['birth'] == owned['birth'] and actual['argv'] == owned['argv']
    assert actual['tool_handle'] == token['actual_root_tool_handle']
    assert token['new_training_updates_before_gate'] == token['test_rows_parsed'] == 0
    return registry
