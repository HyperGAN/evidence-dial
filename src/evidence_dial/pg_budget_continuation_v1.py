"""Explicit128-to512 budget continuation with no changed numerical state.

Parents remain immutable. The importer must separately verify complete parent
execution, qualified outputs and the exact frozen source/data. This helper
permits only total_steps metadata to differ, and refuses scheduled recipes for
which that change would alter the existing update law.
"""
from copy import deepcopy
from dataclasses import asdict

from particlegan import get_recipe
from particlegan.recipes import learning_rate_scales
from .train import tensor_equal


def require_same_schedule(old_config,new_config):
    old,new=deepcopy(old_config),deepcopy(new_config)
    a,b=old['recipe'],new['recipe']
    assert a['total_steps']==128 and b['total_steps']==512
    b['total_steps']=128
    assert tensor_equal(old,new),'Only declared128-to512 budget metadata may change'
    assert a['ema_decay']==0.
    assert a['lr_floor']==a['network_lr_floor']==1.
    assert a['beta2_end'] is None and a['reg_coeff_end'] is None
    assert a['input_noise_std']==a['output_noise_std']==0.
    assert a['continuous_policy'] is None
    x=dict(a);name=x.pop('name');before=get_recipe(name,**x)
    x['total_steps']=512;after=get_recipe(name,**x)
    assert asdict(before)==a
    expected=deepcopy(a);expected['total_steps']=512;assert asdict(after)==expected
    assert all(learning_rate_scales(step,before)==learning_rate_scales(step,after)==(1.,1.) for step in range(512))
    return {'all512_schedule_positions_identical':True,'beta2_and_penalty_schedules_disabled':True,
        'input_and_output_noise_schedules_disabled':True,'only_budget_metadata_differs':True}


def rebind_budget_state(parent_state,target_signature):
    """Copy metadata explicitly; never mutate a parent or reset optimizer/RNG."""
    require_same_schedule(parent_state['config'],target_signature)
    assert type(parent_state['completed_steps']) is int and 0<=parent_state['completed_steps']<=128
    converted=deepcopy(parent_state)
    converted['config']['recipe']['total_steps']=512
    assert converted['config']==target_signature
    check=deepcopy(converted);check['config']['recipe']['total_steps']=128
    assert tensor_equal(check,parent_state)
    return converted
