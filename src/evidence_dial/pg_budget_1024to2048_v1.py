"""Explicit1024→2048 continuation; only budget metadata may change."""
from copy import deepcopy
from dataclasses import asdict

from particlegan import get_recipe
from particlegan.recipes import learning_rate_scales
from .train import tensor_equal

OLD_BUDGET=1024
NEW_BUDGET=2048


def require_same_schedule(old_config,new_config):
    old,new=deepcopy(old_config),deepcopy(new_config)
    before,after=old['recipe'],new['recipe']
    assert before['total_steps']==OLD_BUDGET and after['total_steps']==NEW_BUDGET
    after['total_steps']=OLD_BUDGET
    assert tensor_equal(old,new),'Only declared1024-to2048 budget metadata may change'
    assert before['ema_decay']==0.
    assert before['lr_floor']==before['network_lr_floor']==1.
    assert before['beta2_end'] is None and before['reg_coeff_end'] is None
    assert before['input_noise_std']==before['output_noise_std']==0.
    assert before['continuous_policy'] is None
    config=dict(before);name=config.pop('name')
    old_recipe=get_recipe(name,**config)
    config['total_steps']=NEW_BUDGET
    new_recipe=get_recipe(name,**config)
    assert asdict(old_recipe)==before
    expected=deepcopy(before);expected['total_steps']=NEW_BUDGET
    assert asdict(new_recipe)==expected
    assert all(learning_rate_scales(step,old_recipe)==learning_rate_scales(step,new_recipe)==(1.,1.)
        for step in range(NEW_BUDGET))
    return {'all2048_schedule_positions_identical':True,
        'beta2_and_penalty_schedules_disabled':True,'input_and_output_noise_schedules_disabled':True,
        'only_budget_metadata_differs':True}


def rebind_budget_state(parent_state,target_signature):
    require_same_schedule(parent_state['config'],target_signature)
    assert type(parent_state['completed_steps']) is int and 0<=parent_state['completed_steps']<=OLD_BUDGET
    converted=deepcopy(parent_state)
    converted['config']['recipe']['total_steps']=NEW_BUDGET
    assert converted['config']==target_signature
    check=deepcopy(converted);check['config']['recipe']['total_steps']=OLD_BUDGET
    assert tensor_equal(check,parent_state)
    return converted
