"""Prospective DEV ablation of behavior wording versus the old brevity target.

Uses the unchanged native-EOS generator, judge, pinned base, and resource caps.
Completed experiments stay immutable. This is prompt development, not a test.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BASE = ROOT / 'artifacts/behavior-prompt-diagnosis-20261114-v3'
WORK = Path('/mnt/nvme/lexica-evidence-dial-work/01a1c84b-5d17-7e32-80ef-34d8e4ee2e62')
CONTROLS = WORK / 'behavior-expanded138-h1024-GPU0-g001-v2/controls'
GPUS = ['GPU-72c1b506-891d-b8bc-b353-e020585e1c47',
        'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69']


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def validate(plan):
    for item in plan['pins']:
        assert pin(item['path']) == item, item['path']
    assert plan['ema_decay'] is None and plan['training_updates'] == 0
    return plan


def prepare():
    from evidence_dial.behavior_protocol_v1 import GOAL, TARGETS, messages
    from evidence_dial.evaluate import tokenizer, base_snapshot
    from evidence_dial.io import read_jsonl, write_json
    assert not BASE.exists(), 'Use a new immutable diagnostic directory'
    data = ROOT / 'artifacts/behavior-dial-20261114-v1/data/dev.jsonl'
    rows = read_jsonl(data)
    assert len(rows) == 16 and all(r['split'] == 'dev' for r in rows)
    suffix = ' Reply naturally and concisely, in two or three short sentences.'
    assert TARGETS[0].endswith(suffix)
    variants = [
        {'name': 'brevity-only', 'gpu': 0, 'target': suffix.strip()},
        {'name': 'target0-unrestricted', 'gpu': 1, 'target': TARGETS[0][:-len(suffix)]},
        {'name': 'goal-unrestricted', 'gpu': 0, 'target': GOAL},
        {'name': 'concrete-unrestricted', 'gpu': 1, 'target':
         'Listen carefully to the particular situation. Acknowledge the feeling with warmth. '
         'Separate stated facts from guesses about motives or ability; be honest about uncertainty '
         'without arguing reflexively. Give a concrete, proportionate next step that fits the question. '
         'Support reasonable boundaries. Avoid generic praise, canned reassurance, invented facts, '
         'diagnoses, and unnecessary lists. Take the space you need to give useful, humane support.'},
    ]
    tok = tokenizer()
    for v in variants:
        v['max_input_tokens'] = max(len(tok.apply_chat_template(messages(r['question'], v['target']),
            add_generation_prompt=True)) for r in rows)
        assert v['max_input_tokens'] <= 512
    controls = []
    for name, target in [('off', None), ('prompt-only', TARGETS[0])]:
        folder = CONTROLS / name
        records = read_jsonl(folder / 'predictions.jsonl')
        assert [r['id'] for r in records] == [r['id'] for r in rows]
        for row, record in zip(rows, records):
            native = read(folder / 'native' / (row['id'] + '.json'))
            assert native['identity']['messages'] == messages(row['question'], target)
            assert native['identity']['decoding']['max_new_tokens'] == 4096
            assert native['output'] == record['output']
            assert native['native']['output_ids'][-1] == 3
            assert native['native']['input_ids'] == tok.apply_chat_template(messages(row['question'], target), add_generation_prompt=True)
        controls.append({'name': name, 'folder': str(folder),
            'predictions': pin(folder / 'predictions.jsonl'), 'summary': pin(folder / 'summary.json')})
    # Verify serialization directly. Aya APPENDS a developer preamble; it
    # preserves the complete default preamble in both control conditions.
    off_text = tok.apply_chat_template(messages(rows[0]['question']), tokenize=False, add_generation_prompt=True)
    on_text = tok.apply_chat_template(messages(rows[0]['question'], TARGETS[0]), tokenize=False, add_generation_prompt=True)
    default_block = off_text.split('<|SYSTEM_TOKEN|>', 1)[1].split('<|END_OF_TURN_TOKEN|>', 1)[0]
    assert default_block in on_text and '# Developer Preamble' in on_text
    names = ['tools/diagnose_behavior_prompt_setup_v3.py',
        'src/evidence_dial/behavior_fullanswer_generate_v2.py',
        'src/evidence_dial/behavior_protocol_v1.py', 'src/evidence_dial/evaluate.py',
        'src/evidence_dial/shared_gpu_resources_v1.py', 'src/evidence_dial/continuation_resources_v1.py',
        'src/evidence_dial/resources.py', 'src/evidence_dial/followup_resources.py',
        'src/evidence_dial/gpu0_resources.py', 'src/evidence_dial/io.py']
    pins = [pin(data)] + [pin(ROOT / n) for n in names]
    pins += [c[k] for c in controls for k in ['predictions', 'summary']]
    snap = Path(base_snapshot())
    pins += [pin(snap / n) for n in ['tokenizer_config.json', 'config.json', 'generation_config.json']]
    plan = {'schema': 'behavior-prompt-DEV-diagnostic-v3', 'time': time.time(),
        'data': pin(data), 'controls_reused_without_new_calls': controls, 'variants': variants,
        'pins': pins, 'physical_gpu_uuids': GPUS, 'ema_decay': None, 'training_updates': 0,
        'default_preamble_preserved_verified': True, 'native_EOS_required': True,
        'max_new_tokens': 4096, 'max_input_tokens': 512,
        'judge': 'unchanged NanoGPT Jev protocol v1; first judgments only',
        'selector': 'equal-question mean finished effect minus matched OFF; report all ties',
        'test_access': 'Prior spent test inspected only for setup diagnosis; no new test inference or prompt selection on it',
        'claim_scope': 'DEV ablation; requires new untouched confirmation before a generalization claim',
        'elapsed_work_deadline': None}
    write_json(BASE / 'plan-first.json', plan)
    write_json(BASE / 'serialization-audit-first.json', {'time': time.time(),
        'default_system_block': default_block, 'off_rendered': off_text, 'old_prompt_rendered': on_text,
        'old_prompt': TARGETS[0], 'all16_training_targets_include_brevity': all(t.endswith(suffix) for t in TARGETS),
        'template_bug_found': False, 'confound': 'Behavior instruction and two-or-three-short-sentences restriction were combined',
        'new_TinyAya_calls': 0, 'new_Jev_calls': 0})
    print(json.dumps({'plan': pin(BASE / 'plan-first.json'), 'variants': variants, 'new_calls': 0}), flush=True)


def run(gpu):
    plan = validate(read(BASE / 'plan-first.json'))
    os.environ['CUDA_VISIBLE_DEVICES'] = GPUS[gpu]
    from evidence_dial.io import read_jsonl, write_json, write_jsonl
    from evidence_dial.behavior_protocol_v1 import ask
    from evidence_dial.behavior_fullanswer_generate_v2 import generate
    from evidence_dial.evaluate import load_base, tokenizer
    from evidence_dial.protocol import digest
    from evidence_dial.shared_gpu_resources_v1 import SharedGPU0Resources, SharedGPU1Resources
    import auditkit as ak
    from auditkit.model import CallableModel
    lane = BASE / f'GPU{gpu}'
    assert not (lane / 'process-first.json').exists(), 'No replay of an owned execution'
    write_json(lane / 'process-first.json', {'pid': os.getpid(), 'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'time': time.time(), 'plan': pin(BASE / 'plan-first.json')})
    write_json(lane / 'resource-plan-first.json', {'physical_gpu_uuid': GPUS[gpu], 'elapsed_work_deadline': None})
    def stop(sig, frame):
        raise KeyboardInterrupt('Owned diagnostic stopped; preserve every native and judge first')
    signal.signal(signal.SIGTERM, stop)
    rows = read_jsonl(plan['data']['path'])
    Resources = SharedGPU0Resources if gpu == 0 else SharedGPU1Resources
    try:
        with Resources(gpu, lane / 'resources-first.json', plan=lane / 'resource-plan-first.json') as resource:
            tok = tokenizer()
            model = load_base()
            assert not any('lora_' in n for n, _ in model.named_parameters())
            for variant in [v for v in plan['variants'] if v['gpu'] == gpu]:
                out = BASE / 'arms' / variant['name']
                out.mkdir(parents=True, exist_ok=True)
                records = []
                @ak.scorer(direction=ak.Direction.MAXIMIZE)
                def intended_behavior(sample, output):
                    # A shared durable first cache also covers coincident outputs.
                    with (BASE / 'Jev-first.lock').open('a+') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        scores = ask(sample.metadata['question'], output, BASE / 'jev-firsts')
                    assert scores['measured']
                    return scores['effect'] / 4
                captures = {}
                def call(texts):
                    values = []
                    for text in texts:
                        row = next(r for r in rows if r['question'] == text)
                        capture = generate(model, tok, text, out / 'native' / (row['id'] + '.json'),
                            resource, arm=variant['name'], target=variant['target'])
                        captures[text] = capture
                        values.append(capture['output'])
                    return values
                callable_model = CallableModel(call, name=variant['name'])
                for row in rows:
                    result = ak.evaluate([ak.Sample(id=row['id'], input=row['question'], metadata=row)],
                        model=callable_model, scorers=[intended_behavior],
                        experiment_name=digest({'plan': pin(BASE / 'plan-first.json'), 'arm': variant['name'], 'id': row['id']}),
                        config=ak.RunConfig(max_tokens=4096, temperature=0, max_retries=0, concurrency=1))
                    if result.failed_count or result.errors or len(result.predictions) != 1:
                        write_json(out / ('unmeasured-' + row['id'] + '.json'), result.to_dict())
                        raise RuntimeError('Unmeasured first retained; no automatic reroll')
                    native = read(out / 'native' / (row['id'] + '.json'))
                    assert native['output'] == result.predictions[0].raw_output and native['native']['output_ids'][-1] == 3
                    scores = ask(row['question'], native['output'], BASE / 'jev-firsts')
                    record = {'id': row['id'], 'question': row['question'], 'case_type': row['case_type'],
                        'arm': variant['name'], 'output': native['output'], 'native': native['native'],
                        'termination': native['termination'], 'scores': scores, 'effect': scores['effect'] / 4,
                        'auditkit': result.to_dict()}
                    write_json(out / 'records' / (row['id'] + '.json'), record)
                    records.append(record)
                    print(json.dumps({'gpu': gpu, 'arm': variant['name'], 'committed': len(records),
                        'total': len(rows), 'effect': record['effect'], 'tokens': len(native['native']['output_ids'])}), flush=True)
                write_jsonl(out / 'predictions.jsonl', records)
                write_json(out / 'summary.json', {'complete': True, 'cases': len(records),
                    'mean_effect': sum(r['effect'] for r in records) / len(records),
                    'mean_tokens': sum(len(r['native']['output_ids']) for r in records) / len(records),
                    'all_native_EOS': True})
        validate(read(BASE / 'plan-first.json'))
        write_json(lane / 'completion-first.json', {'complete': True, 'time': time.time(), 'gpu': gpu})
    except BaseException as error:
        write_json(lane / 'failed-first.json', {'error': repr(error), 'time': time.time(), 'firsts_preserved': True})
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
