"""Freeze all DEV maximum ties plus a fixed three-strength display grid."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--closure', type=Path, required=True)
    ap.add_argument('--guard-preparation', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from evidence_dial.behavior_fullanswer_confirmation_v2 import BASE, GPUS, pin, verified, read, declared_arms, exact_strength_health, validate_plan
    from evidence_dial.io import write_json
    assert not args.out.exists()
    closure = read(args.closure); candidates, maxima, display = declared_arms(closure)
    assert pin(closure['source']['path']) == closure['source']
    preparations = {}
    for path in args.guard_preparation:
        prep = read(path); assert prep['arm']['name'] not in preparations
        preparations[prep['arm']['name']] = (path, prep)
    assert set(preparations) == {a['name'] for a in candidates}
    arms = [{'name': 'off', 'kind': 'off', 'target': None, 'gpu': 0},
        {'name': 'prompt-only', 'kind': 'prompt-only', 'target_index': 0, 'gpu': 1}]
    inputs = [pin(args.closure), pin(BASE / 'fresh-confirmation-data/manifest-first.json')]
    sources = []
    for index, candidate in enumerate(candidates):
        path, prep = preparations[candidate['name']]
        health = exact_strength_health(candidate, prep); health['preparation'] = pin(path)
        arms.append({'name': candidate['name'], 'kind': 'particle', 'candidate': candidate,
            'gpu': index % 2, 'health': health})
        plan = read(verified(candidate['plan']))
        inputs += list(health.values()) + [candidate[k] for k in ['artifact', 'FULL', 'plan', 'quarter_admission', 'predictions', 'summary']]
        sources += plan['sources']
    manifest = read(BASE / 'fresh-confirmation-data/manifest-first.json'); holdout = manifest['file']
    assert manifest['cases'] == 32; verified(holdout); inputs.append(holdout)
    names = ['src/evidence_dial/behavior_fullanswer_confirmation_v2.py', 'tools/freeze_behavior_fullanswer_holdout_v3.py',
        'tools/measure_behavior_fullanswer_holdout_arm_v2.py', 'tools/run_behavior_fullanswer_holdout_gpu_v2.py',
        'tools/audit_behavior_fullanswer_holdout_v2.py', 'tools/close_behavior_expanded_training_v3.py',
        'tools/prepare_behavior_fullanswer_strength_guard_v2.py', 'src/evidence_dial/behavior_fullanswer_generate_v2.py',
        'src/evidence_dial/behavior_protocol_v1.py', 'src/evidence_dial/behavior_expanded_binding_v2.py',
        'src/evidence_dial/shared_gpu_resources_v1.py', 'src/evidence_dial/continuation_resources_v1.py',
        'src/evidence_dial/resources.py', 'src/evidence_dial/evaluate.py']
    sources += [pin(ROOT / n) for n in names]
    sources += [pin(importlib.import_module(n).__file__) for n in ['particle_sliders.inference', 'auditkit', 'auditkit.model']]
    def unique(items):
        result = {}
        for item in items:
            assert item is not None
            verified(item)
            if item['path'] in result: assert result[item['path']] == item
            result[item['path']] = item
        return list(result.values())
    plan = {'schema': 'behavior-fresh-holdout32-fullanswer-locked-v2', 'time': time.time(),
        'development_closure': pin(args.closure), 'data': holdout, 'cases': 32,
        'arms': arms, 'selected_development_maxima': [a['name'] for a in maxima],
        'display_checkpoint': display['artifact'], 'display_strengths': [.5, 1., 2.],
        'display_rule': closure['display_checkpoint_rule'], 'ema_decay': None, 'max_new_tokens': 4096,
        'physical_gpu_uuids': GPUS, 'holdout_based_reselection': False,
        'uncertainty': {'unit': 'matched_question', 'resamples': 10000, 'seed': 20261114},
        'claim_scope': 'All preselected ties reported; display arms descriptive, not used to select on holdout',
        'generation': 'Ordinary same user question OFF/ON; explicit fixed TARGETS[0] prompt-only control; native greedy with4096 safetycap; requireEOS before scoring',
        'inputs': unique(inputs), 'sources': unique(sources), 'new_training_updates': 0,
        'require_native_EOS':True, 'old_behavior_holdout32_spent':True, 'new_model_calls_at_freeze': 0, 'holdout_rows_read_at_freeze': 0}
    validate_plan(plan)
    args.out.mkdir(parents=True)
    write_json(args.out / 'plan-first.json', plan)
    write_json(args.out / 'selection-lock-first.json', {'time': time.time(), 'plan': pin(args.out / 'plan-first.json'),
        'arms': [a['name'] for a in arms], 'development_maxima': plan['selected_development_maxima'],
        'display_checkpoint': plan['display_checkpoint'], 'holdout_outputs_seen': 0,
        'no_test_reselection': True, 'holdout_question_rows_read': 0})
    print(__import__('json').dumps({'plan': pin(args.out / 'plan-first.json'), 'arms': len(arms),
        'cases_each': 32, 'GPUs': [0,1], 'new_model_calls': 0, 'holdout_rows_read': 0}))


if __name__ == '__main__': main()
