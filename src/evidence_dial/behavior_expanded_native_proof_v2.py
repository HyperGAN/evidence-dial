"""First-state behavior binding proof; reuse unchanged model/kernel evidence."""
from pathlib import Path
import json
import torch
from particle_sliders.lifecycle import read_packet
from .fresh_native_zero_proof_v2 import durable_first,cpu
from .train import tensor_equal
from .benchmark import file_sha
from .io import write_json


def compare_original_initial(original,current):
    before,after=original['config'],current['config'];assert before.keys()==after.keys()
    changes={'implementation_sources','backend','rows_sha256','conditions_sha256','row_count','recipe'}
    for key in before.keys()-changes:assert before[key]==after[key],key
    a,b=dict(before['implementation_sources']),dict(after['implementation_sources'])
    old='evidence_dial.first_answer_binding_v2';new='evidence_dial.behavior_expanded_binding_v2'
    assert old in a and new not in a and new in b and old not in b
    root=Path(__file__).resolve().parents[2]
    assert a.pop(old)==file_sha(root/'src/evidence_dial/first_answer_binding_v2.py')
    assert b.pop(new)==file_sha(root/'src/evidence_dial/behavior_expanded_binding_v2.py')
    assert a==b,'Other implementation changes are outside this behavior-data experiment'
    a,b=before['backend'],after['backend'];assert a.keys()==b.keys()
    for key in a.keys()-{'protocol','target_convention'}:assert a[key]==b[key],key
    from .behavior_expanded_binding_v2 import PROTOCOL
    assert b['protocol']==PROTOCOL and before['row_count']==60 and after['row_count']==138
    assert before['recipe'].keys()==after['recipe'].keys()
    differences={k for k in before['recipe'] if before['recipe'][k]!=after['recipe'][k]}
    assert differences<={'total_steps','batch_size','lr','d_lr_mult','prior_lr_mult'}
    assert after['recipe']['total_steps']==1024 and after['recipe']['batch_size']==8
    assert after['recipe']['lr'] in [.00025,.001] and after['recipe']['ema_decay']==0.
    for component in ['network','critic','prior']:
        assert tensor_equal(cpu(original[component]),cpu(current[component])),component
    return sorted(differences)


def run(session,native,out,original_full,cache,*,reuse=None):
    out,cache=Path(out),Path(cache)
    assert session.game.completed_steps==0 and session.ema.state_dict()=={'decay':None,'values':None}
    first=out/'FULL-000000.pt'
    if not first.exists():session.checkpoint(first)
    packet=read_packet(first);current=packet['payload']['game']
    differences=compare_original_initial(read_packet(original_full)['payload']['game'],current)
    identity={'FULL0_sha256':file_sha(first),'cache_index_sha256':file_sha(cache/'index.json'),'source_sha256':file_sha(__file__)}
    if reuse is not None:
        parent=Path(reuse);proof=json.loads((parent/'native-check.json').read_text())
        assert proof['initial_ON_equals_OFF'] and proof['positive_sample_hidden_matches_direct_frozen_hidden'] and proof['connected_tensors']==2593 and proof['exact_FULL_restore']
        source=read_packet(parent/'FULL-000000.pt')['payload']['game']
        for key in current['config'].keys()-{'recipe'}:assert current['config'][key]==source['config'][key],key
        for component in ['network','critic','prior']:assert tensor_equal(cpu(source[component]),cpu(current[component]))
        assert {k for k in current['config']['recipe'] if current['config']['recipe'][k]!=source['config']['recipe'][k]}<={'lr','d_lr_mult','prior_lr_mult'}
        session.restore(packet);assert tensor_equal(cpu(current),cpu(session.checkpoint()['payload']['game']))
        return {**proof,'reused_unchanged_native_firsts_from':str(parent),'reuse_source_FULL0_sha256':file_sha(parent/'FULL-000000.pt'),
            'current_FULL0_sha256':file_sha(first),'permitted_original_recipe_differences':differences,'new_native_forwards':0}
    folder=out/'native-zero-firsts'
    def once(name,call):return durable_first(folder,name,identity,call)
    row=session.game.rows[0]
    def response(scale,ids=None):
        with torch.no_grad():return native._response(ids or row.row.neutral_ids,scale=scale)
    off=once('ordinary-question-OFF',lambda:response(0.));on=once('ordinary-question-ON',lambda:response(1.))
    positive=once('positive-frozen-direct',lambda:response(0.,row.row.positive_ids))
    assert torch.equal(off,cpu(row.neutral)) and torch.equal(on,off)
    assert torch.equal(positive,cpu(row.target)),'Raw generation prefill and direct final-hidden targets differ'
    def gradients():
        value=native.generated(row);params=list(native.network.parameters())
        grad=torch.autograd.grad(value.sum(),params,allow_unused=True)
        assert len(params)==2593 and all(g is not None and torch.isfinite(g).all() for g in grad)
        return {'connected_tensors':len(params),'finite':True,'gradient_elements':sum(g.numel() for g in grad)}
    result=once('finite-connected-gradients',gradients);assert result['connected_tensors']==2593
    session.restore(packet);assert tensor_equal(cpu(current),cpu(session.checkpoint()['payload']['game']))
    assert session.ema.state_dict()=={'decay':None,'values':None}
    return {'initial_ON_equals_OFF':True,'OFF_matches_new_cache':True,'positive_sample_hidden_matches_direct_frozen_hidden':True,
        'connected_tensors':2593,'exact_FULL_restore':True,'no_weight_EMA':True,'new_training_updates':0,
        'all_initial_network_critic_prior_tensors_match_original':True,'permitted_original_recipe_differences':differences,
        'original_FULL0_sha256':file_sha(original_full),'current_FULL0_sha256':file_sha(first),'durable_firsts':str(folder)}
