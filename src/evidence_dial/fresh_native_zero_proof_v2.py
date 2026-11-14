"""Durable native zero-output/gradient proof for the new TRAIN256 cohort.

Native firsts are saved stage by stage. An unresolved stage is never replayed
automatically. This does not run updates or alter the original PG formulation.
"""
import json
from pathlib import Path
import torch

from particle_sliders.lifecycle import atomic_save,read_packet
from .benchmark import file_sha
from .io import write_json
from .train import tensor_equal


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
    return value


def durable_first(folder,name,identity,call):
    """Commit one proof call; retained pending calls require explicit review."""
    folder=Path(folder)
    path=folder/(name+'.pt')
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['identity']==identity
        return saved['value']
    pending=folder/(name+'.pending.json')
    if pending.exists():raise RuntimeError('Unresolved native-zero first '+name+'; retain without automatic replay')
    write_json(pending,{'identity':identity,'stage':name,'native_first_pending':True})
    result=cpu(call())
    atomic_save(path,{'identity':identity,'value':result})
    return result


def run(session,native,out,original_full,cache):
    out,original_full,cache=Path(out),Path(original_full),Path(cache)
    assert session.game.completed_steps==0 and session.ema.state_dict()=={'decay':None,'values':None}
    initial=out/'FULL-000000.pt'
    if not initial.exists():session.checkpoint(initial)
    packet=read_packet(initial)
    assert tensor_equal(cpu(packet['payload']['game']),cpu(session.checkpoint()['payload']['game']))
    original=read_packet(original_full)['payload']['game']
    current=session.checkpoint()['payload']['game']
    before,after=original['config'],current['config']
    assert before.keys()==after.keys()
    special={'implementation_sources','backend','rows_sha256','conditions_sha256','row_count','recipe'}
    for key in before.keys()-special:
        assert before[key]==after[key],key
    a,b=dict(before['implementation_sources']),dict(after['implementation_sources'])
    old_module='evidence_dial.first_answer_binding_v2'
    new_module='evidence_dial.curated_first_answer_binding_v1'
    assert old_module in a and new_module not in a and new_module in b and old_module not in b
    root=Path(__file__).resolve().parents[2]
    assert a.pop(old_module)==file_sha(root/'src/evidence_dial/first_answer_binding_v2.py')
    assert b.pop(new_module)==file_sha(root/'src/evidence_dial/curated_first_answer_binding_v1.py')
    assert a==b,'Any other implementation source change is out of scope'
    a,b=before['backend'],after['backend']
    assert a.keys()==b.keys()
    for key in a.keys()-{'protocol','target_convention'}:assert a[key]==b[key],key
    assert b['protocol']=='cohere2-canonical-curated256-first-v1'
    assert before['row_count']==60 and after['row_count']==256
    assert before['recipe'].keys()==after['recipe'].keys()
    recipe_differences=[]
    for key in before['recipe']:
        if before['recipe'][key]!=after['recipe'][key]:recipe_differences.append(key)
    assert set(recipe_differences)<={'total_steps','batch_size','lr'}
    assert before['recipe']['total_steps']==128 and before['recipe']['batch_size']==2
    assert (after['recipe']['total_steps'],after['recipe']['batch_size']) in [(768,2),(128,8)]
    assert after['recipe']['lr'] in [.00025,.001]
    for component in ['network','critic','prior']:
        assert tensor_equal(cpu(original[component]),cpu(current[component])),component
    proof=out/'unchanged-formulation-proof.json'
    value={'original_FULL_sha256':file_sha(original_full),'new_FULL0_sha256':file_sha(initial),
        'all_initial_network_critic_prior_tensor_values_exact':True,
        'permitted_game_recipe_differences':recipe_differences,
        'permitted_data_changes':['rows_sha256','conditions_sha256','row_count60to256'],
        'permitted_metadata_changes':['protocol','target_convention','backend_class_source_module'],
        'all_other_config_fields_and_implementation_sources_exact':True,
        'no_weight_EMA':True,'updates':0}
    if proof.exists():assert json.loads(proof.read_text())==value
    else:write_json(proof,value)
    del original,current
    folder=out/'native-zero-firsts'
    identity={'FULL0_sha256':file_sha(initial),'cache_index_sha256':file_sha(cache/'index.json'),
        'proof_source_sha256':file_sha(__file__)}
    def once(name,call):
        return durable_first(folder,name,identity,call)
    row=session.game.rows[0]
    def response(scale):
        with torch.no_grad():return native._response(row.row.neutral_ids,scale=scale)
    off=once('ordinary-question-OFF',lambda:response(0.))
    on=once('ordinary-question-ON',lambda:response(1.))
    assert torch.equal(off,cpu(row.neutral)) and torch.equal(on,off)
    def gradient_proof():
        generated=native.generated(row)
        params=list(native.network.parameters())
        grads=torch.autograd.grad(generated.sum(),params,allow_unused=True)
        assert len(params)==2593 and all(g is not None and torch.isfinite(g).all() for g in grads)
        return {'connected_tensors':len(params),'all_finite':True,
            'gradient_elements':sum(g.numel() for g in grads),'new_updates':0}
    gradients=once('finite-connected-gradients',gradient_proof)
    assert gradients['connected_tensors']==2593 and gradients['all_finite']
    session.restore(packet)
    assert tensor_equal(cpu(packet['payload']['game']),cpu(session.checkpoint()['payload']['game']))
    assert session.ema.state_dict()=={'decay':None,'values':None}
    return {'initial_ON_equals_OFF':True,'OFF_matches_new_cache':True,
        'connected_tensors':2593,'exact_FULL_restore':True,'ema':session.ema.state_dict(),
        'new_training_updates':0,'durable_native_firsts':str(folder),'unresolved_firsts_replayed':False,
        'fresh_native_proof_for_new_TRAIN256_data':True}
