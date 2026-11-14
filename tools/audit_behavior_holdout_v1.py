"""Independently verify all locked native outputs, Jev firsts and paired results."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    assert not args.output.exists(); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from evidence_dial.behavior_confirmation_v1 import pin, verified, read, validate_plan
    from evidence_dial.behavior_protocol_v1 import messages, TARGETS, QUESTIONS, JEV, ENDPOINT
    from evidence_dial.protocol import digest
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.io import read_jsonl, write_json
    import numpy as np
    plan = validate_plan(read(args.plan)); folder = args.plan.resolve().parent
    selection = read(folder / 'selection-lock-first.json')
    assert selection['plan'] == pin(args.plan) and selection['holdout_outputs_seen'] == 0
    assert selection['arms'] == [a['name'] for a in plan['arms']] and selection['no_test_reselection']
    rows = read_jsonl(verified(plan['data'])); assert len(rows) == 32 and all(r['split'] == 'holdout' for r in rows)
    assert len({r['question'] for r in rows}) == len({r['id'] for r in rows}) == 32
    terminals = []
    for gpu in [0,1]:
        shard = folder / f'GPU{gpu}'; identity = read(shard / 'controller-first.json')
        terminal_path = shard / 'actual-tool-terminal-receipt-first.json'; terminal = read(terminal_path)
        assert terminal['actual_tool_terminal_collected'] and terminal['actual_tool_terminal_chunk']
        assert terminal['exit_code'] == 0 and terminal['reaped']
        assert terminal['pid'] == identity['pid'] and terminal['birth'] == identity['birth'] and terminal['command'] == identity['argv']
        assert not Path(f'/proc/{identity["pid"]}').exists()
        complete = read(shard / 'completion-first.json'); expected = [a['name'] for a in plan['arms'] if a['gpu'] == gpu]
        assert complete['complete'] and complete['arms'] == expected and not complete['selection_performed']
        assert not (shard / 'controller-failed-first.json').exists()
        for index, name in enumerate(expected,1):
            execution_path = shard / 'executions' / f'arm-{index:03d}.json'
            execution = read(execution_path); end = read(execution_path.with_suffix('.terminal.json'))
            assert execution['name'] == name and end['reaped'] and end['returncode'] == 0
            assert not Path(f'/proc/{execution["pid"]}').exists()
        terminals.append(pin(terminal_path))
    tok = tokenizer(); all_records = {}; summaries = []; native_count = 0; judgment_firsts = set()
    for arm in plan['arms']:
        out = folder / 'arms' / arm['name']; identity = read(out / 'actual-owned-process-first.json')
        assert identity['plan'] == pin(args.plan) and identity['name'] == arm['name']
        assert not Path(f'/proc/{identity["pid"]}').exists()
        complete = read(out / 'completion-first.json'); assert complete['complete'] and complete['cases'] == 32
        assert complete['training_updates'] == 0 and complete['weight_EMA'] is False and not complete['holdout_selection']
        assert not (out / 'arm-failed-first.json').exists()
        resource = read(out / 'resources-first.completed.json')
        assert resource['state'] == 'complete' and resource['failure'] is None
        with (out / '.shared-gpu-job.lock').open('r+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(lease, fcntl.LOCK_UN)
        if arm['kind'] == 'particle':
            c = arm['candidate']; label = f'particle-raw-{c["artifact"]["sha256"]}-strength-{c["strength"]:g}'; target = None
        else:
            label = arm['name']; target = TARGETS[0] if arm['kind'] == 'prompt-only' else None
        records = read_jsonl(out / 'predictions.jsonl'); assert [r['id'] for r in records] == [r['id'] for r in rows]
        for row, record in zip(rows, records):
            saved = read(out / 'native' / f'{row["id"]}.json')
            expected_ids = tok.apply_chat_template(messages(row['question'], target), add_generation_prompt=True)
            assert record['arm'] == saved['identity']['arm'] == label
            assert saved['identity']['messages'] == messages(row['question'], target)
            assert record['question'] == row['question'] and record['case_type'] == row['case_type']
            assert record['native']['input_ids'] == saved['native']['input_ids'] == saved['identity']['input_ids'] == expected_ids
            decoding = saved['identity']['decoding']
            assert decoding == {'do_sample':False, 'max_new_tokens':64, 'eos_token_id':3, 'use_cache':True}
            ids = record['native']['output_ids']
            assert ids == saved['native']['output_ids'] and 1 <= len(ids) <= 64 and (ids[-1] == 3 or len(ids) == 64)
            assert record['output'] == saved['output'] == tok.decode(ids, skip_special_tokens=True)
            assert saved['termination'] == ('native_eos' if ids[-1] == 3 else 'max_new_tokens')
            request = {'model':JEV, 'state':{'user_message':row['question'], 'response':record['output']}, 'questions':QUESTIONS}
            key = digest({'endpoint':ENDPOINT, 'request':request}); first_path = out / 'jev-firsts' / (key + '.json'); first = read(first_path)
            assert first['http_status'] == 200 and first['request'] == request and first['response_sha256'] == digest(first['response'])
            assert first['response']['model'] == JEV
            answers = first['response']['answers']
            for key, kind, field in [('effect','score','score'), ('sympathy','noul','noul'), ('sycophancy','noul','noul')]:
                assert answers[key]['type'] == kind and record['scores'][key] == answers[key][field]
                assert math.isfinite(record['scores'][key])
            assert record['scores']['measured'] and record['effect'] == answers['effect']['score'] / 4.
            assert 0 <= record['effect'] <= 1
            judgment_firsts.add(str(first_path)); native_count += 1
        summary = read(out / 'summary.json')
        assert summary['complete'] and summary['cases'] == 32 and summary['arm'] == label
        for source, field in [('effect','mean_effect'), ('sympathy','mean_sympathy_probability'), ('sycophancy','mean_sycophancy_probability')]:
            values = [r['effect'] if source == 'effect' else r['scores'][source] for r in records]
            assert summary[field] == sum(values) / 32
        all_records[arm['name']] = records
        summaries.append({'name':arm['name'], 'kind':arm['kind'], 'selected_on_DEV':arm['name'] in plan['selected_development_maxima'],
            'display_grid':arm['kind'] == 'particle' and arm['candidate']['artifact'] == plan['display_checkpoint'],
            'mean_effect':summary['mean_effect'], 'mean_sympathy_probability':summary['mean_sympathy_probability'],
            'mean_sycophancy_probability':summary['mean_sycophancy_probability'],
            'predictions':pin(out / 'predictions.jsonl'), 'summary':pin(out / 'summary.json')})
    off_ids = [r['native']['input_ids'] for r in all_records['off']]
    for arm in plan['arms']:
        if arm['kind'] == 'particle': assert [r['native']['input_ids'] for r in all_records[arm['name']]] == off_ids
    rng = np.random.default_rng(plan['uncertainty']['seed'])
    indices = rng.integers(0,32,size=(plan['uncertainty']['resamples'],32)); comparisons = []
    for arm in plan['arms']:
        if arm['name'] == 'off': continue
        for baseline in ['off','prompt-only']:
            if arm['name'] == baseline: continue
            delta = np.asarray([a['effect']-b['effect'] for a,b in zip(all_records[arm['name']],all_records[baseline])],dtype=np.float64)
            interval = np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()
            comparisons.append({'arm':arm['name'], 'baseline':baseline, 'cases':32,
                'mean_effect_difference':float(delta.mean()), 'paired_bootstrap95':interval,
                'unit':'matched_question', 'resamples':10000, 'seed':20261114,
                'scope':'Individual descriptive intervals; no test-based selection or simultaneous-coverage claim'})
    result = {'time':time.time(), 'state':'all-locked-behavior-holdout-arms-independently-admitted',
        'source':pin(__file__), 'plan':pin(args.plan), 'selection_lock':pin(folder / 'selection-lock-first.json'),
        'actual_root_terminals':terminals, 'all_own_processes_absent_and_private_leases_released':True,
        'cases_each':32, 'native_generations_verified':native_count, 'Jev_judgment_records_verified':native_count,
        'unique_per_arm_Jev_firsts_verified':len(judgment_firsts), 'arms':summaries, 'comparisons':comparisons,
        'all_preselected_ties_reported':True, 'holdout_based_reselection':False,
        'new_model_calls':0, 'new_Jev_calls':0, 'new_updates':0,
        'limitations':['Single synthetic question set and training seed', 'Same Jev rubric/judge in curation and evaluation',
            'Native evaluation capped at64 tokens; truncation retained', 'Question bootstrap does not establish semantic-family independence']}
    write_json(args.output,result)
    print(json.dumps({'audit':pin(args.output),'arms':len(summaries),'cases_each':32,'comparisons':comparisons}))


if __name__ == '__main__': main()
