"""Explicit hyperparameter fork at256; preserve all learned/numerical history.

This is a separately declared continuation, never an exact unchanged-recipe
resume. Only current base LR, its three optimizer-group LRs, and future budget
metadata change. The original cohort/checkpoint is never modified.
"""
from copy import deepcopy
from dataclasses import asdict

from particlegan import get_recipe
from particlegan.recipes import learning_rate_scales
from .train import tensor_equal

PARENT_STEP = 256
PARENT_BUDGET = 1024
BUDGET = 384
OLD_LR = .001
NEW_LR = .00025


def require_declared_branch(before, after):
    expected = deepcopy(before)
    assert before['recipe']['lr'] == OLD_LR and before['recipe']['total_steps'] == PARENT_BUDGET
    expected['recipe'].update(lr=NEW_LR, total_steps=BUDGET)
    assert tensor_equal(expected, after), 'Only declared LR/budget metadata may change'
    a, b = before['recipe'], after['recipe']
    assert a['ema_decay'] == b['ema_decay'] == 0.
    assert a['lr_floor'] == a['network_lr_floor'] == 1.
    assert a['beta2_end'] is a['reg_coeff_end'] is None
    assert a['input_noise_std'] == a['output_noise_std'] == 0.
    assert a['continuous_policy'] is None and a['d_lr_mult'] == 18. and a['prior_lr_mult'] == 30.
    recipes = []
    for config in [a, b]:
        options = dict(config)
        recipe = get_recipe(options.pop('name'), **options)
        assert asdict(recipe) == config
        recipes.append(recipe)
    assert all(learning_rate_scales(step, recipe) == (1., 1.)
        for recipe in recipes for step in range(BUDGET))
    return {'official_recipe_roundtrips_exact': True, 'all384_schedule_scales_constant': True,
        'only_base_LR_and_future_budget_metadata_changed': True}


def branch_state(parent, target_signature):
    assert parent['completed_steps'] == PARENT_STEP
    require_declared_branch(parent['config'], target_signature)
    converted = deepcopy(parent)
    converted['config'] = deepcopy(target_signature)
    old_rates, new_rates = {}, {}
    roles = {'generator': 1., 'critic': 18., 'prior': 30.}
    for opt in ['opt_g', 'opt_d']:
        for index, group in enumerate(converted[opt]['param_groups']):
            role = group['role']
            assert role in roles and role not in old_rates
            assert group['lr'] == OLD_LR * roles[role]
            old_rates[role] = group['lr']
            group['lr'] = NEW_LR * roles[role]
            new_rates[role] = group['lr']
    assert set(old_rates) == set(roles)
    inverse = deepcopy(converted)
    inverse['config'] = deepcopy(parent['config'])
    for opt in ['opt_g', 'opt_d']:
        for group in inverse[opt]['param_groups']:
            group['lr'] = old_rates[group['role']]
    assert tensor_equal(inverse, parent), 'No weight/history/RNG/stream/exposure leaf may change'
    assert converted['config'] == target_signature
    return converted
