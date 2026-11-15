"""Lock exact trained arms, reuse saved DEV health probes, then measure fresh64."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from mine_behavior_promptfix_v4 import BASE, DIAG, GPUS, read, pin
OUT = BASE / 'locked-fresh64-v4'


def validate(plan):
    for item in plan['sources'] + plan['inputs']:
        assert pin(item['path']) == item, item['path']
    assert plan['ema_decay'] is None and plan['cases'] == 64 and plan['require_native_EOS']
    return plan


def freeze():
    from evidence_dial.io import read_jsonl, write_json
    from evidence_dial.behavior_fullanswer_health_v3 import check, passing_guard
    closure_path = BASE / 'development-search-closure-first.json'
    closure = read(closure_path)
    assert closure['state'] == 'promptfix-two-rate-24-cell-DEV-search-complete'
    assert closure['fresh64_still_sealed'] and closure['holdout_rows_read'] == 0
    manifest_path = DIAG / 'new-confirmation-data/manifest-first.json'
    manifest = read(manifest_path)
    assert pin(manifest['data']['path']) == manifest['data']
    assert manifest['cases'] == 64
    assert not (OUT / 'plan-first.json').exists()
    assert not (OUT / 'arms').exists(), 'No test generations may precede the exact arm lock'
    assert not any((OUT / f'GPU{g}/process-first.json').exists() for g in [0, 1])
    selection = closure['all_maximum_ties']
    representative = closure['display_checkpoint_representative']
    candidates = [(a, True) for a in selection]
    for strength in [.25, .5, 1.]:
        candidate = next(a for a in closure['all24_arms'] if a['artifact'] == representative['artifact'] and a['strength'] == strength)
        if not any(a['name'] == candidate['name'] for a, _ in candidates):
            candidates.append((candidate, False))
    previous_closure_path = ROOT / 'artifacts/behavior-expanded-h1024-20261114-v2/development-search-closure-first.json'
    previous = read(previous_closure_path)['display_checkpoint_representative']
    previous_reference = {**previous, 'name': 'previous-expanded138-' + previous['name'],
        'source_candidate_name': previous['name'], 'reference': True, 'inference_binding': 'expanded138-v2'}
    candidates.append((previous_reference, False))
    target = read(BASE / 'declaration-first.json')['selected_prompt_only_control']['target']
    arms = [{'name': 'off', 'kind': 'off', 'target': None, 'selected': False},
            {'name': 'prompt-only', 'kind': 'prompt-only', 'target': target, 'selected': False}]
    arms += [{'name': candidate['name'], 'kind': 'particle', 'candidate': candidate,
              'target': None, 'selected': selected, 'reference': candidate.get('reference', False)}
             for candidate, selected in candidates]
    # Balance declared work using already-read DEV token counts, not test answers.
    diagnostic = read(DIAG / 'independent-audit-first.json')
    estimates = {'off': next(a['mean_tokens'] for a in diagnostic['arms'] if a['name'] == 'off'),
                 'prompt-only': next(a['mean_tokens'] for a in diagnostic['arms'] if a['name'] == 'target0-unrestricted')}
    loads = [0., 0.]
    for arm in arms:
        if arm['kind'] == 'particle':
            records = read_jsonl(arm['candidate']['predictions']['path'])
            estimates[arm['name']] = sum(len(r['native']['output_ids']) for r in records) / len(records)
        gpu = 0 if loads[0] <= loads[1] else 1
        arm['gpu'] = gpu
        arm['DEV_estimated_tokens_per_case'] = estimates[arm['name']]
        loads[gpu] += estimates[arm['name']]
    names = ['tools/confirm_behavior_promptfix_v4.py', 'tools/audit_behavior_promptfix_confirmation_v4.py',
        'src/evidence_dial/behavior_promptfix_confirm_v4.py', 'src/evidence_dial/behavior_fullanswer_generate_v2.py',
        'src/evidence_dial/behavior_protocol_v1.py', 'src/evidence_dial/behavior_promptfix_binding_v4.py',
        'src/evidence_dial/behavior_expanded_binding_v2.py',
        'src/evidence_dial/behavior_fullanswer_health_v3.py', 'src/evidence_dial/evaluate.py',
        'src/evidence_dial/shared_gpu_resources_v1.py', 'src/evidence_dial/continuation_resources_v1.py']
    sources = [pin(ROOT / n) for n in names]
    inputs = [pin(closure_path), pin(manifest_path), manifest['data'],
        pin(DIAG / 'independent-audit-first.json'), pin(BASE / 'training-declaration-first.json'), pin(previous_closure_path)]
    for candidate, _ in candidates:
        inputs += [candidate['artifact'], candidate['FULL'], candidate['predictions'], candidate['summary'], candidate['quarter_admission']]
    lock = {'schema': 'promptfix-fresh64-exact-arms-v4', 'arms': arms, 'sources': sources,
        'inputs': inputs, 'data': manifest['data'], 'cases': 64, 'ema_decay': None,
        'require_native_EOS': True, 'max_new_tokens': 4096, 'physical_gpu_uuids': GPUS,
        'selected_maximum_ties': [a['name'] for a in selection],
        'display_only': [a['name'] for a in arms if a['kind'] == 'particle' and not a['selected'] and not a['reference']],
        'reference_arms': [previous_reference['name']],
        'display_checkpoint': representative['artifact'], 'display_strength': representative['strength'],
        'test_reselection': False, 'same_user_question_all_arms': True,
        'generation': 'Unchanged pinned native greedy; natural EOS; cutoffs unfinished and unscored',
        'shared_Jev_first_cache': True,
        'uncertainty': {'unit': 'matched_question', 'resamples': 10000, 'seed': 20261114},
        'claim_scope': 'Fresh synthetic questions in overlapping broad families; individual intervals; all locked arms reported'}
    lock_path = OUT / 'arms-lock-before-health-first.json'
    if lock_path.exists():
        assert read(lock_path) == lock, 'Exact frozen comparison cannot change'
    else:
        write_json(lock_path, lock)
    guards = []
    for candidate, _ in candidates:
        health = OUT / 'health' / candidate['name']
        step = candidate['step']
        probes_path = health / f'probes-{step:06d}.json'
        records = read_jsonl(candidate['predictions']['path'])
        probes = [next(r for r in records if r['case_type'] == kind)
            for kind in ['discouragement', 'unsupported_conclusion', 'reasonable_boundary', 'uncertainty']]
        value = {'checkpoint': step, 'strength': candidate['strength'],
            'probes': [{k: r[k] for k in ['id', 'question', 'output', 'native']} for r in probes],
            'saved_DEV_predictions': candidate['predictions'], 'new_generations': 0}
        if probes_path.exists():
            assert read(probes_path) == value
        else:
            write_json(probes_path, value)
        # Reuse byte-identical previous health firsts when questions/answers match.
        previous = Path(candidate['FULL']['path']).parent / 'health-Jev-firsts'
        destination = health / 'health-Jev-firsts'
        destination.mkdir(parents=True, exist_ok=True)
        if previous.exists():
            for path in previous.glob('*.json'):
                if '-call-commit' in path.name:
                    continue
                saved = read(path)
                if saved.get('http_status') != 200:
                    continue
                dest = destination / path.name
                if not dest.exists():
                    dest.write_bytes(path.read_bytes())
        guard = health / f'guard-{step:06d}-fullanswer-Jev-v2.json'
        if not guard.exists():
            check(health, step, Path(candidate['FULL']['path']))
        assert passing_guard(health, step, Path(candidate['FULL']['path'])) == guard
        guards.append({'name': candidate['name'], 'guard': pin(guard)})
    plan = {**lock, 'time': time.time(), 'arms_lock': pin(lock_path), 'exact_strength_health': guards,
        'new_test_model_calls_at_freeze': 0, 'new_test_Jev_calls_at_freeze': 0}
    plan['inputs'] += [g['guard'] for g in guards] + [pin(lock_path)]
    validate(plan)
    write_json(OUT / 'plan-first.json', plan)
    print(json.dumps({'plan': pin(OUT / 'plan-first.json'), 'arms': [a['name'] for a in arms],
        'GPU_estimated_loads': loads, 'new_test_calls': 0}), flush=True)


def run(gpu):
    plan = validate(read(OUT / 'plan-first.json'))
    os.environ['CUDA_VISIBLE_DEVICES'] = GPUS[gpu]
    import torch
    from evidence_dial.io import read_jsonl, write_json
    from evidence_dial.evaluate import tokenizer, load_base
    from evidence_dial.behavior_promptfix_confirm_v4 import evaluate
    from evidence_dial.behavior_promptfix_binding_v4 import inference_backend
    from evidence_dial.behavior_expanded_binding_v2 import inference_backend as previous_inference_backend
    from evidence_dial.shared_gpu_resources_v1 import SharedGPU0Resources, SharedGPU1Resources
    from particle_sliders.inference import open_inference
    folder = OUT / f'GPU{gpu}'
    assert not (folder / 'process-first.json').exists()
    write_json(folder / 'process-first.json', {'time': time.time(), 'pid': os.getpid(),
        'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'argv': [x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        'plan': pin(OUT / 'plan-first.json')})
    def stop(sig, frame):
        raise KeyboardInterrupt('Owned fresh64 inference stopped; preserve every first')
    signal.signal(signal.SIGTERM, stop)
    write_json(folder / 'resource-plan-first.json', {'physical_gpu_uuid': GPUS[gpu], 'elapsed_work_deadline': None})
    rows = read_jsonl(plan['data']['path'])
    assert len(rows) == len({r['id'] for r in rows}) == len({r['question'] for r in rows}) == 64
    assert all(r['split'] == 'holdout' for r in rows)
    Resources = SharedGPU0Resources if gpu == 0 else SharedGPU1Resources
    def measure(arm, tok, resource):
        out = OUT / 'arms' / arm['name']
        if arm['kind'] == 'particle':
            c = arm['candidate']
            backend = previous_inference_backend() if c.get('inference_binding') == 'expanded138-v2' else inference_backend()
            with open_inference(c['artifact']['path'], backend=backend, operation='response') as inference:
                native = inference.binding.response.__self__
                with native.slider.scaled(c['strength']):
                    evaluate(rows, native.model, tok, out, resource, arm=arm['name'], first_cache=OUT / 'jev-firsts')
        else:
            model = load_base()
            assert not any('lora_' in n for n, _ in model.named_parameters())
            evaluate(rows, model, tok, out, resource, arm=arm['name'], target=arm['target'], first_cache=OUT / 'jev-firsts')
    try:
        with Resources(gpu, folder / 'resources-first.json', plan=folder / 'resource-plan-first.json') as resource:
            tok = tokenizer()
            for arm in [a for a in plan['arms'] if a['gpu'] == gpu]:
                measure(arm, tok, resource)
                gc.collect()
                torch.cuda.empty_cache()
                resource.check()
        validate(read(OUT / 'plan-first.json'))
        write_json(folder / 'completion-first.json', {'time': time.time(), 'complete': True, 'cases_per_arm': 64,
            'arms': [a['name'] for a in plan['arms'] if a['gpu'] == gpu], 'test_reselection': False})
    except BaseException as error:
        write_json(folder / 'failed-first.json', {'time': time.time(), 'error': repr(error), 'firsts_preserved': True})
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['freeze', 'run'])
    ap.add_argument('--gpu', type=int, choices=[0, 1])
    args = ap.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        assert args.gpu is not None
        run(args.gpu)
