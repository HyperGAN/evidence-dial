"""Admit a fresh stopping-data study only after actual native0 qualification."""
import json
import os
from pathlib import Path

from .benchmark import file_sha

ROOT = Path(__file__).resolve().parents[2]
PARENT_REGISTRY = ROOT / 'artifacts/fixed-formulation-pursuit-20261112-v1/long-checkpoint-confirmation-execution-v1/required-studies-and-controls-registry-first.json'


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def verified(item):
    assert pin(item['path']) == item
    return Path(item['path'])


def verify_sgd_admission(folder, root_pid):
    folder = Path(folder).resolve()
    token = json.loads((folder / 'SGD-admission-first.json').read_text())
    assert token['state'] == 'native0-qualified-five-study-inventory-frozen-before-SGD'
    assert token['recipe'] == pin(folder / 'recipe.json')
    assert token['declaration'] == pin(folder / 'queue-declaration-shared-v1-first.json')
    owned = json.loads(verified(token['actual_owned_process']).read_text())
    assert owned['pid'] == root_pid
    proc = Path('/proc') / str(root_pid)
    assert proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == owned['birth']
    assert [x.decode() for x in proc.joinpath('cmdline').read_bytes().split(b'\0') if x] == owned['argv']
    parent = json.loads(PARENT_REGISTRY.read_text())
    registry = json.loads(verified(token['required_registry']).read_text())
    assert registry['parent_registry'] == pin(PARENT_REGISTRY)
    assert registry['required_studies'][:4] == parent['required_studies']
    assert len(registry['required_studies']) == 5
    for key in ['retained_finished_PG_audits', 'mandatory_SFT_maximum_registry', 'conditional_TRAIN512_SFT_control']:
        assert registry[key] == parent[key], key
    new = registry['required_studies'][4]
    assert new['recipe'] == token['recipe']
    assert new['budget'] == 1024 and new['development_checkpoints'] == [256, 512, 768, 1024]
    actual = new['actual_identity_at_registry_freeze']
    assert actual['pid'] == owned['pid'] and actual['birth'] == owned['birth'] and actual['argv'] == owned['argv']
    assert actual['tool_handle'] == token['actual_root_tool_handle']
    assert new['native0_qualification'] == token['native0_qualification']
    native0 = json.loads(verified(token['native0_qualification']).read_text())
    assert native0['state'] == 'actual-native0-qualified-before-first-SGD'
    assert native0['recipe'] == token['recipe']
    for key in ['FULL0', 'native_proof', 'unchanged_formulation_proof', 'initialization']:
        verified(native0[key])
    proof = json.loads(verified(native0['native_proof']).read_text())
    assert proof['initial_ON_equals_OFF'] and proof['OFF_matches_new_cache']
    assert proof['connected_tensors'] == 2593 and proof['exact_FULL_restore']
    assert proof['complete_answer_prefix_ON_equals_OFF_cache_and_2593_connected_gradients']
    assert proof['ema'] == {'decay': None, 'values': None} and proof['new_training_updates'] == 0
    assert token['new_training_updates_before_gate'] == 0
    assert token['test_rows_parsed'] == 0
    return registry
