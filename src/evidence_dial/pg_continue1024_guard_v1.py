"""Reuse a qualified1024 output check only for the exact imported raw state.

No verdict is relabeled or generated. Later checkpoints require their own
ordinary qualified check through the original admission function.
"""
import hashlib
import json
from pathlib import Path

from particle_sliders.lifecycle import read_packet
from particle_sliders.inference import read_inference
from .benchmark import file_sha
from .train import tensor_equal
from .pg_budget_1024to2048_v1 import rebind_budget_state
from .qualified_text_guard_admission_v1 import passing_guard as original_guard,QUALIFICATIONS


def passing_guard(out,step,checkpoint):
    result=original_guard(out,step,checkpoint)
    if result is not None or step!=1024:return result
    out=Path(out);receipt=out/'parent1024-continuation-import-first.json'
    if not receipt.exists():return None
    value=json.loads(receipt.read_text())
    assert value['state']=='complete' and value['old_budget']==1024 and value['new_budget']==2048
    assert value['no_native_forwards_or_updates_during_import']
    assert Path(checkpoint).resolve()==Path(value['converted_FULL1024']).resolve()
    for item in value['immutable_pins']:assert file_sha(item['path'])==item['sha256'],item['path']
    parent=read_packet(value['parent_FULL1024'])['payload']
    current=read_packet(checkpoint)['payload']
    assert parent['game']['completed_steps']==current['game']['completed_steps']==1024
    expected=rebind_budget_state(parent['game'],current['game']['config'])
    assert tensor_equal(current['game'],expected)
    assert current['ema']==parent['ema']=={'decay':None,'values':None}
    assert current['last_metrics']==parent['last_metrics']
    old_native=read_inference(value['parent_NATIVE1024'])
    new_native=read_inference(value['converted_NATIVE1024'])
    assert old_native['specification']==new_native['specification']
    assert tensor_equal(old_native['network'],new_native['network'])
    assert tensor_equal(new_native['network'],current['game']['network'])
    prefix=value['journal_prefix']
    with (out/'updates.jsonl').open('rb') as file:data=file.read(prefix['bytes'])
    assert hashlib.sha256(data).hexdigest()==prefix['sha256']
    guard=original_guard(Path(value['parent_FULL1024']).parent,1024,Path(value['parent_FULL1024']))
    assert guard is not None and str(guard.resolve())==str(Path(value['parent_guard']).resolve())
    return guard
