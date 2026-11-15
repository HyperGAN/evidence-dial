"""Fresh complete-answer TRAIN curation after the prospective prompt ablation.

Same-question neutral/user-only versus positive/system-target, empty history.
Original caches, sources, results and trained weights are never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BASE = ROOT / 'artifacts/behavior-promptfix-20261114-v4'
DIAG = ROOT / 'artifacts/behavior-prompt-diagnosis-20261114-v3'
GPUS = ['GPU-72c1b506-891d-b8bc-b353-e020585e1c47', 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69']


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def validate(plan):
    for p in plan['sources'] + plan['inputs']:
        assert pin(p['path']) == p, p['path']
    assert plan['ema_decay'] is None and plan['history'] == []
    return plan


def prepare():
    from evidence_dial.io import read_jsonl, write_json
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.behavior_protocol_v1 import messages
    audit = read(DIAG / 'independent-audit-first.json')
    assert audit['state'] == 'all64-new-DEV-native-EOS-and-first-judgments-audited'
    diagnostic = read(DIAG / 'plan-first.json')
    targets = [v for v in diagnostic['variants'] if v['name'].endswith('-unrestricted')]
    assert len(targets) == 3
    assert audit['DEV_maximum_prompt_ties'] == ['target0-unrestricted']
    data = ROOT / 'artifacts/behavior-dial-20261114-v1/data/train.jsonl'
    rows = read_jsonl(data)
    assert len(rows) == 64 and all(r['split'] == 'train' for r in rows)
    tok = tokenizer()
    assert max(len(tok.apply_chat_template(messages(r['question'], t['target']), add_generation_prompt=True)) for r in rows for t in targets) <= 512
    assert not BASE.exists()
    names = ['tools/mine_behavior_promptfix_v4.py', 'src/evidence_dial/behavior_protocol_v1.py',
        'src/evidence_dial/behavior_fullanswer_generate_v2.py', 'src/evidence_dial/evaluate.py',
        'src/evidence_dial/shared_gpu_resources_v1.py', 'src/evidence_dial/continuation_resources_v1.py',
        'src/evidence_dial/resources.py', 'src/evidence_dial/followup_resources.py',
        'src/evidence_dial/gpu0_resources.py', 'src/evidence_dial/io.py']
    sources = [pin(ROOT / n) for n in names]
    inputs = [pin(data), pin(DIAG / 'plan-first.json'), pin(DIAG / 'independent-audit-first.json'),
        pin(DIAG / 'new-confirmation-data/manifest-first.json'),
        pin(ROOT / 'artifacts/behavior-dial-20261114-v1/Jev-fixture-qualification-first.json')]
    for gpu in [0, 1]:
        plan = {'schema': 'behavior-promptfix-full-EOS-TRAIN-mining-v4', 'time': time.time(),
            'gpu': gpu, 'physical_gpu_uuid': GPUS[gpu], 'sources': sources, 'inputs': inputs,
            'data': pin(data), 'assigned_ids': [r['id'] for i, r in enumerate(rows) if i % 2 == gpu],
            'targets': targets, 'neutral_system_message': False, 'neutral_instruction': '', 'history': [],
            'ema_decay': None, 'model': 'raw pinned pretrained TinyAya; no adapter',
            'generation': 'unchanged native greedy; 4096 safety cap; require EOS for all judgments',
            'maximum_generations': 128, 'positive_acceptance_effect_min': 3.0,
            'positive_acceptance_gain': 'strictly greater than matched OFF on the same TRAIN question',
            'cache_selection': 'One highest-effect accepted target per question; earliest declared target breaks curation-only ties',
            'all_candidates_and_firsts_retained': True, 'question_weighting': 'equal over admitted unique TRAIN questions',
            'DEV_or_test_generation': False, 'training_updates': 0, 'elapsed_work_deadline': None}
        write_json(BASE / f'GPU{gpu}/plan-first.json', plan)
    write_json(BASE / 'declaration-first.json', {'time': time.time(), 'state': 'declared-before-new-TRAIN-calls',
        'shards': [pin(BASE / f'GPU{g}/plan-first.json') for g in [0, 1]],
        'selected_prompt_only_control': next(t for t in targets if t['name'] == 'target0-unrestricted'),
        'total_generations': 256, 'training_questions': 64, 'positive_variants_each': 3,
        'full_answers_required': True, 'new_training_updates': 0,
        'new_test_set_sealed': pin(DIAG / 'new-confirmation-data/manifest-first.json')})
    print(json.dumps({'declaration': pin(BASE / 'declaration-first.json'), 'new_calls': 0}), flush=True)


def run(gpu):
    plan_path = BASE / f'GPU{gpu}/plan-first.json'
    plan = validate(read(plan_path))
    folder = plan_path.parent
    os.environ['CUDA_VISIBLE_DEVICES'] = plan['physical_gpu_uuid']
    import torch
    from particle_sliders.lifecycle import atomic_save
    from evidence_dial.io import read_jsonl, write_json
    from evidence_dial.evaluate import load_base, tokenizer
    from evidence_dial.behavior_fullanswer_generate_v2 import generate
    from evidence_dial.behavior_protocol_v1 import ask
    from evidence_dial.shared_gpu_resources_v1 import SharedGPU0Resources, SharedGPU1Resources
    assert not (folder / 'process-first.json').exists()
    write_json(folder / 'process-first.json', {'time': time.time(), 'pid': os.getpid(),
        'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19], 'plan': pin(plan_path)})
    def stop(sig, frame):
        raise KeyboardInterrupt('Owned TRAIN miner stopped; preserve all firsts')
    signal.signal(signal.SIGTERM, stop)
    rows = [r for r in read_jsonl(plan['data']['path']) if r['id'] in plan['assigned_ids']]
    assert len(rows) == 32 and all(r['split'] == 'train' for r in rows)
    Resources = SharedGPU0Resources if gpu == 0 else SharedGPU1Resources
    selected = []
    try:
        with Resources(gpu, folder / 'resources-first.json', plan=plan_path) as resource:
            model, tok = load_base(), tokenizer()
            assert not any('lora_' in n for n, _ in model.named_parameters())
            for number, row in enumerate(rows, 1):
                decisions = []
                for target in [{'name': 'off', 'target': None}] + plan['targets']:
                    out = folder / 'cases' / row['id'] / target['name']
                    first = []
                    def capture(module, inputs, output):
                        if not first:
                            first.append(output.last_hidden_state[:, -1, :].detach().float().cpu().clone())
                    hook = model.model.register_forward_hook(capture)
                    try:
                        value = generate(model, tok, row['question'], out / 'native-first.json', resource,
                            arm=target['name'], target=target['target'])
                    finally:
                        hook.remove()
                    assert len(first) == 1 and first[0].shape == (1, 2048) and torch.isfinite(first[0]).all()
                    assert value['native']['output_ids'][-1] == 3
                    state_path = out / 'prefill-first.pt'
                    assert not state_path.exists()
                    atomic_save(state_path, {'hidden': first[0], 'input_ids': value['native']['input_ids'],
                        'native_first': pin(out / 'native-first.json'), 'target': target['target']})
                    scores = ask(row['question'], value['output'], folder / 'jev-firsts')
                    assert scores['measured']
                    record = {'id': row['id'], 'question': row['question'], 'case_type': row['case_type'],
                        'name': target['name'], 'target': target['target'], 'native': pin(out / 'native-first.json'),
                        'state': pin(state_path), 'scores': scores, 'effect': scores['effect'] / 4}
                    write_json(out / 'decision-first.json', record)
                    decisions.append(record)
                    print(json.dumps({'gpu': gpu, 'case': number, 'total_cases': 32, 'id': row['id'],
                        'target': target['name'], 'effect': record['effect'], 'tokens': len(value['native']['output_ids'])}), flush=True)
                off = decisions[0]
                accepted = [d for d in decisions[1:] if d['scores']['effect'] >= plan['positive_acceptance_effect_min'] and d['effect'] > off['effect']]
                best = max(accepted, key=lambda d: d['effect']) if accepted else None
                choice = {'id': row['id'], 'case_type': row['case_type'], 'off': off,
                    'candidates': decisions[1:], 'accepted': accepted, 'selected': best,
                    'reason': 'highest accepted finished target; balanced unique questions' if best else 'no target meets frozen absolute and matched-uplift rules'}
                write_json(folder / 'choices' / (row['id'] + '.json'), choice)
                selected.append(choice)
                print(json.dumps({'gpu': gpu, 'finished_questions': number, 'admitted_so_far': sum(c['selected'] is not None for c in selected)}), flush=True)
        validate(read(plan_path))
        write_json(folder / 'completion-first.json', {'time': time.time(), 'complete': True, 'cases': 32,
            'native_generations': 128, 'selected_questions': sum(c['selected'] is not None for c in selected),
            'new_training_updates': 0, 'DEV_or_test_calls': 0, 'plan': pin(plan_path)})
    except BaseException as error:
        write_json(folder / 'failed-first.json', {'time': time.time(), 'error': repr(error), 'firsts_preserved': True})
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'run'])
    ap.add_argument('--gpu', type=int, choices=[0, 1])
    args = ap.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        assert args.gpu is not None
        run(args.gpu)
