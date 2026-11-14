"""Transfer only the unchanged parent256 output check across the LR fork."""
import json
from pathlib import Path

from .benchmark import file_sha
from .qualified_text_guard_admission_v1 import passing_guard as ordinary_guard
from .stop256_refine_lr_state_v1 import branch_state
from .train import tensor_equal


def passing_guard(candidate, step, full):
    candidate, full = Path(candidate), Path(full)
    direct = ordinary_guard(candidate, step, full)
    if direct is not None:
        return direct
    if step != 256 or not (candidate / 'parent256-refinement-import-first.json').exists():
        return None
    import torch
    from particle_sliders.lifecycle import read_packet
    from particle_sliders.inference import read_inference
    receipt = json.loads((candidate / 'parent256-refinement-import-first.json').read_text())
    assert receipt['state'] == 'complete' and receipt['new_training_updates'] == receipt['new_model_calls'] == 0
    for item in receipt['immutable_pins']:
        assert file_sha(item['path']) == item['sha256']
    assert full.resolve() == Path(receipt['converted_FULL256']).resolve()
    parent = Path(receipt['parent_FULL256'])
    before, after = [read_packet(p)['payload'] for p in [parent, full]]
    expected = branch_state(before['game'], after['game']['config'])
    assert tensor_equal(expected, after['game']) and tensor_equal(before['last_metrics'], after['last_metrics'])
    assert before['ema'] == after['ema'] == {'decay': None, 'values': None}
    original, converted = [read_inference(p) for p in [receipt['parent_NATIVE256'], candidate / 'NATIVE-000256.pt']]
    assert original['specification'] == converted['specification']
    assert tensor_equal(original['network'], converted['network'])
    assert tensor_equal(converted['network'], after['game']['network'])
    guard = ordinary_guard(parent.parent, 256, parent)
    assert guard is not None and str(guard.resolve()) == receipt['parent_guard']['path']
    assert file_sha(guard) == receipt['parent_guard']['sha256']
    assert not torch.cuda.is_initialized()
    return guard
