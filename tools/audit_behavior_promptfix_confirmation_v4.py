"""Verify every fresh64 answer and judge first, then report all locked arms."""
from collections import Counter
from decimal import Decimal
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from confirm_behavior_promptfix_v4 import OUT, read, pin, validate


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    from evidence_dial.io import read_jsonl, write_json
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.behavior_protocol_v1 import messages, QUESTIONS, JEV, ENDPOINT
    from evidence_dial.protocol import digest
    from particle_sliders.inference import read_inference
    plan = validate(read(OUT / 'plan-first.json'))
    assert not (OUT / 'independent-completion-audit-first.json').exists()
    for gpu in [0, 1]:
        folder = OUT / f'GPU{gpu}'
        identity = read(folder / 'process-first.json')
        terminal = read(folder / 'actual-tool-terminal-receipt-first.json')
        assert terminal['actual_tool_terminal_collected'] and terminal['exit_code'] == 0
        assert terminal['actual_tool_terminal_chunk'] and terminal['reaped']
        assert terminal['pid'] == identity['pid'] and terminal['birth'] == identity['birth']
        assert terminal['command'] == identity['argv']
        proc = Path(f'/proc/{identity["pid"]}/stat')
        assert not proc.exists() or proc.read_text().rsplit(')', 1)[1].split()[19] != identity['birth']
        assert read(folder / 'completion-first.json')['complete']
        resource = read(folder / 'resources-first.completed.json')
        assert resource['state'] == 'complete' and resource['failure'] is None
        with (folder / '.shared-gpu-job.lock').open('r+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lease, fcntl.LOCK_UN)
    data = read_jsonl(plan['data']['path'])
    assert len(data) == 64 and all(r['split'] == 'holdout' for r in data)
    tok = tokenizer()
    reports, vectors = [], {}
    artifact_checks = {}
    for arm in plan['arms']:
        folder = OUT / 'arms' / arm['name']
        records = read_jsonl(folder / 'predictions.jsonl')
        assert [r['id'] for r in records] == [r['id'] for r in data]
        if arm['kind'] == 'particle':
            candidate = arm['candidate']
            assert pin(candidate['artifact']['path']) == candidate['artifact']
            sha = candidate['artifact']['sha256']
            if sha not in artifact_checks:
                packet = read_inference(candidate['artifact']['path'])
                assert packet['selection'] == 'raw' and packet['origin']['completed_steps'] == candidate['step']
                assert packet['specification']['configuration']['ema_decay'] is None
                admitted = read(candidate['quarter_admission']['path'])
                assert admitted['NATIVE'] == candidate['artifact'] and admitted['no_weight_EMA']
                artifact_checks[sha] = candidate['artifact']
        for row, r in zip(data, records):
            native = read(folder / 'native' / (row['id'] + '.json'))
            conversation = messages(row['question'], arm['target'])
            ids = tok.apply_chat_template(conversation, add_generation_prompt=True)
            assert r['question'] == row['question'] and r['case_type'] == row['case_type']
            assert r['arm'] == native['identity']['arm'] == arm['name']
            assert native['identity']['messages'] == conversation
            assert native['identity']['input_ids'] == native['native']['input_ids'] == r['native']['input_ids'] == ids
            assert native['identity']['decoding'] == {'do_sample': False, 'max_new_tokens': 4096, 'eos_token_id': 3, 'use_cache': True}
            output_ids = r['native']['output_ids']
            assert output_ids == native['native']['output_ids'] and output_ids[-1] == 3 and 1 <= len(output_ids) <= 4096
            assert r['output'] == native['output'] == tok.decode(output_ids, skip_special_tokens=True)
            assert native['termination'] == r['termination'] == 'native_eos'
            request = {'model': JEV, 'state': {'user_message': row['question'], 'response': r['output']}, 'questions': QUESTIONS}
            first = read(r['scores']['source'])
            assert first['request'] == request and first['http_status'] == 200 and first['response']['model'] == JEV
            assert first['response_sha256'] == digest(first['response'])
            assert Path(r['scores']['source']).stem == digest({'endpoint': ENDPOINT, 'request': request})
            answers = first['response']['answers']
            assert answers['effect']['type'] == 'score' and answers['effect']['score'] == r['scores']['effect'] == r['effect'] * 4
            assert math.isfinite(r['effect']) and 0 <= r['effect'] <= 1
            for k in ['sympathy', 'sycophancy']:
                assert answers[k]['type'] == 'noul' and answers[k]['noul'] == r['scores'][k]
                assert math.isfinite(r['scores'][k]) and 0 <= r['scores'][k] <= 1
            auditkit = r['auditkit']
            assert not auditkit['errors'] and auditkit['failed_count'] == 0
            assert len(auditkit['predictions']) == 1
            assert auditkit['predictions'][0]['raw_output'] == r['output']
            assert auditkit['predictions'][0]['sample_id'] == row['id']
        vectors[arm['name']] = [Decimal(str(r['scores']['effect'])) / 4 for r in records]
        n = len(records)
        mean = sum(vectors[arm['name']]) / n
        summary = read(folder / 'summary.json')
        assert summary['complete'] and summary['cases'] == 64
        assert summary['mean_effect'] == sum(r['effect'] for r in records) / n
        groups = Counter(r['case_type'] for r in records)
        reports.append({'name': arm['name'], 'kind': arm['kind'], 'primary_selected': arm['selected'],
            'previous_setup_reference': arm.get('reference', False),
            'cases': n, 'mean_effect': float(mean), 'mean_effect_exact_decimal': str(mean),
            'mean_sympathy_probability': sum(r['scores']['sympathy'] for r in records) / n,
            'mean_sycophancy_probability': sum(r['scores']['sycophancy'] for r in records) / n,
            'mean_tokens': sum(len(r['native']['output_ids']) for r in records) / n,
            'by_type': {k: {'cases': count, 'mean_effect': float(sum(Decimal(str(r['scores']['effect'])) / 4
                for r in records if r['case_type'] == k) / count)} for k, count in groups.items()},
            'predictions': pin(folder / 'predictions.jsonl'), 'summary': pin(folder / 'summary.json')})
    indices = np.random.default_rng(plan['uncertainty']['seed']).integers(0, 64, size=(10000, 64))
    comparisons = []
    for arm in plan['arms']:
        baselines = ['off', 'prompt-only']
        if arm['kind'] == 'particle' and not arm.get('reference', False):
            baselines += plan['reference_arms']
        for baseline in baselines:
            if arm['name'] == baseline or (arm['name'] == 'off'):
                continue
            values = [a - b for a, b in zip(vectors[arm['name']], vectors[baseline])]
            sampled = np.array([float(x) for x in values])[indices].mean(axis=1)
            comparisons.append({'arm': arm['name'], 'baseline': baseline, 'primary_selected': arm['selected'],
                'mean_difference': float(sum(values) / 64), 'mean_difference_exact_decimal': str(sum(values) / 64),
                'individual_paired_bootstrap95': np.quantile(sampled, [.025, .975]).tolist(),
                'wins_ties_losses': [sum(x > 0 for x in values), sum(x == 0 for x in values), sum(x < 0 for x in values)]})
    report = {'time': time.time(), 'source': pin(__file__), 'plan': pin(OUT / 'plan-first.json'),
        'state': 'all-locked-fresh64-full-EOS-answers-and-first-judgments-independently-audited',
        'arms': reports, 'comparisons': comparisons, 'selected_on_DEV': plan['selected_maximum_ties'],
        'new_test_reselection': False, 'all_answers_native_EOS': True, 'no_weight_EMA': True,
        'raw_artifacts': list(artifact_checks.values()), 'new_calls_in_audit': 0,
        'total_test_answers': len(plan['arms']) * 64,
        'limits': ['Synthetic questions; broad families overlap', 'Same curation and evaluation judge',
                   'One training seed', 'Individual intervals, no simultaneous coverage claim',
                   'Display strengths are descriptive; no test-based winner selection']}
    write_json(OUT / 'independent-completion-audit-first.json', report)
    print(json.dumps({'audit': pin(OUT / 'independent-completion-audit-first.json'),
        'arms': reports, 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
