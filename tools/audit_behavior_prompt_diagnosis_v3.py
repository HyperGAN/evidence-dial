"""Read-only joins of the six DEV arms, native answers, and Jev firsts."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_behavior_prompt_setup_v3 import BASE, read, pin, validate
from evidence_dial.behavior_protocol_v1 import messages, TARGETS, QUESTIONS, ENDPOINT, JEV
from evidence_dial.io import read_jsonl, write_json
from evidence_dial.protocol import digest
from evidence_dial.evaluate import tokenizer


def main():
    import numpy as np
    plan = validate(read(BASE / 'plan-first.json'))
    assert not (BASE / 'independent-audit-first.json').exists()
    for gpu in [0, 1]:
        assert read(BASE / f'GPU{gpu}/completion-first.json')['complete']
        assert read(BASE / f'GPU{gpu}/resources-first.completed.json')['state'] == 'complete'
    data = read_jsonl(plan['data']['path'])
    tok = tokenizer()
    declared = [{'name': c['name'], 'folder': c['folder'],
        'target': TARGETS[0] if c['name'] == 'prompt-only' else None} for c in plan['controls_reused_without_new_calls']]
    declared += [dict(v, folder=str(BASE / 'arms' / v['name'])) for v in plan['variants']]
    arms = []
    vectors = {}
    for arm in declared:
        folder = Path(arm['folder'])
        records = read_jsonl(folder / 'predictions.jsonl')
        assert [r['id'] for r in records] == [r['id'] for r in data]
        for row, r in zip(data, records):
            native = read(folder / 'native' / (row['id'] + '.json'))
            convo = messages(row['question'], arm['target'])
            ids = tok.apply_chat_template(convo, add_generation_prompt=True)
            assert native['identity']['messages'] == convo
            assert native['native']['input_ids'] == native['identity']['input_ids'] == ids
            assert native['native']['output_ids'][-1] == 3
            assert native['identity']['decoding']['max_new_tokens'] == 4096
            assert native['output'] == r['output'] == tok.decode(native['native']['output_ids'], skip_special_tokens=True)
            assert r['native'] == native['native']
            first = read(r['scores']['source'])
            request = {'model': JEV, 'state': {'user_message': row['question'], 'response': r['output']}, 'questions': QUESTIONS}
            assert first['request'] == request and first['http_status'] == 200
            assert first['response_sha256'] == digest(first['response'])
            assert Path(r['scores']['source']).stem == digest({'endpoint': ENDPOINT, 'request': request})
            assert first['response']['model'] == JEV
            raw = first['response']['answers']
            assert raw['effect']['score'] == r['scores']['effect'] == r['effect'] * 4
            for key in ['sympathy', 'sycophancy']:
                assert raw[key]['noul'] == r['scores'][key]
        vectors[arm['name']] = [Decimal(str(r['effect'])) for r in records]
        groups = Counter(r['case_type'] for r in records)
        mean = sum(vectors[arm['name']]) / len(records)
        arms.append({'name': arm['name'], 'n': len(records), 'mean_effect': str(mean),
            'mean_tokens': sum(len(r['native']['output_ids']) for r in records) / len(records),
            'mean_sympathy': sum(r['scores']['sympathy'] for r in records) / len(records),
            'mean_sycophancy': sum(r['scores']['sycophancy'] for r in records) / len(records),
            'by_type': {k: {'n': n, 'mean_effect': str(sum(Decimal(str(r['effect'])) for r in records if r['case_type'] == k) / n)} for k, n in groups.items()},
            'predictions': pin(folder / 'predictions.jsonl')})
    rng = np.random.default_rng(20261114)
    indices = rng.integers(0, len(data), size=(10000, len(data)))
    comparisons = []
    for left, right in [(a['name'], 'off') for a in arms if a['name'] != 'off'] + [
        ('target0-unrestricted', 'prompt-only'), ('brevity-only', 'off'),
        ('concrete-unrestricted', 'target0-unrestricted')]:
        diffs = [a - b for a, b in zip(vectors[left], vectors[right])]
        values = np.array([float(x) for x in diffs])
        comparisons.append({'left': left, 'right': right,
            'paired_mean_difference': str(sum(diffs) / len(diffs)),
            'paired_question_bootstrap95_individual': np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
            'wins_ties_losses': [sum(x > 0 for x in diffs), sum(x == 0 for x in diffs), sum(x < 0 for x in diffs)]})
    variant_names = {v['name'] for v in plan['variants']}
    maximum = max(sum(vectors[n]) for n in variant_names)
    ties = [v['name'] for v in plan['variants'] if sum(vectors[v['name']]) == maximum]
    report = {'time': time.time(), 'plan': pin(BASE / 'plan-first.json'), 'audit_source': pin(__file__),
        'state': 'all64-new-DEV-native-EOS-and-first-judgments-audited',
        'arms': arms, 'comparisons': comparisons, 'DEV_maximum_prompt_ties': ties,
        'selector': plan['selector'], 'new_test_calls': 0, 'new_training_updates': 0,
        'new_answers': 64, 'reused_control_answers': 32, 'new_Jev_ratings_in_audit': 0,
        'limits': ['16 synthetic DEV cases', 'Four prompts compared; individual intervals are descriptive, not simultaneous',
                   'No held-out generalization claim', 'Original completed results and scores unchanged'],
        'actual_outer_process_termination': 'Verify unified execution handles separately'}
    write_json(BASE / 'independent-audit-first.json', report)
    print(json.dumps({'audit': pin(BASE / 'independent-audit-first.json'), 'arms': arms,
        'comparisons': comparisons, 'DEV_maximum_prompt_ties': ties}), flush=True)


if __name__ == '__main__':
    main()
