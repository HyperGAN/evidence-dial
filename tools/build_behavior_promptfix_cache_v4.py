"""Independently audit all new TRAIN firsts before admitting a PG cache."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from mine_behavior_promptfix_v4 import BASE, read, pin, validate


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from particle_sliders.audio import AudioRow, PreparedAudioResponse
    from particle_sliders.lifecycle import atomic_save
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.behavior_protocol_v1 import messages, QUESTIONS, ENDPOINT, JEV
    from evidence_dial.protocol import digest
    from evidence_dial.io import read_jsonl, write_json
    assert not (BASE / 'cache').exists()
    plans = [validate(read(BASE / f'GPU{g}/plan-first.json')) for g in [0, 1]]
    for gpu in [0, 1]:
        completion = read(BASE / f'GPU{gpu}/completion-first.json')
        assert completion['complete'] and completion['native_generations'] == 128
        assert read(BASE / f'GPU{gpu}/resources-first.completed.json')['state'] == 'complete'
        identity = read(BASE / f'GPU{gpu}/process-first.json')
        terminal = read(BASE / f'GPU{gpu}/actual-tool-terminal-receipt-first.json')
        assert terminal['actual_tool_terminal_collected'] and terminal['actual_tool_terminal_chunk']
        assert terminal['exit_code'] == 0 and terminal['reaped']
        assert terminal['pid'] == identity['pid'] and terminal['birth'] == identity['birth']
        proc = Path(f'/proc/{identity["pid"]}/stat')
        assert not proc.exists() or proc.read_text().rsplit(')', 1)[1].split()[19] != identity['birth']
    rows = read_jsonl(plans[0]['data']['path'])
    tok = tokenizer()
    prepared, index, audited = [], [], []
    coverage = Counter()
    for row in rows:
        gpu = next(g for g, p in enumerate(plans) if row['id'] in p['assigned_ids'])
        plan = plans[gpu]
        choice_path = BASE / f'GPU{gpu}/choices' / (row['id'] + '.json')
        choice = read(choice_path)
        assert choice['id'] == row['id'] and choice['case_type'] == row['case_type']
        records = [choice['off']] + choice['candidates']
        assert len(records) == 4
        values = []
        for target, record in zip([{'name': 'off', 'target': None}] + plan['targets'], records):
            assert record['name'] == target['name'] and record['target'] == target['target']
            assert record['id'] == row['id'] and record['question'] == row['question']
            assert pin(record['native']['path']) == record['native']
            assert pin(record['state']['path']) == record['state']
            native = read(record['native']['path'])
            convo = messages(row['question'], target['target'])
            ids = tok.apply_chat_template(convo, add_generation_prompt=True)
            assert native['identity']['messages'] == convo
            assert native['native']['input_ids'] == native['identity']['input_ids'] == ids
            assert native['identity']['decoding']['max_new_tokens'] == 4096
            assert native['native']['output_ids'][-1] == 3
            assert native['output'] == tok.decode(native['native']['output_ids'], skip_special_tokens=True)
            state = torch.load(record['state']['path'], map_location='cpu', weights_only=False)
            assert state['input_ids'] == ids and state['native_first'] == record['native'] and state['target'] == target['target']
            assert state['hidden'].dtype == torch.float32 and state['hidden'].shape == (1, 2048)
            assert torch.isfinite(state['hidden']).all()
            first = read(record['scores']['source'])
            request = {'model': JEV, 'state': {'user_message': row['question'], 'response': native['output']}, 'questions': QUESTIONS}
            assert first['request'] == request and first['http_status'] == 200
            assert first['response']['model'] == JEV and first['response_sha256'] == digest(first['response'])
            assert Path(record['scores']['source']).stem == digest({'endpoint': ENDPOINT, 'request': request})
            raw = first['response']['answers']
            assert raw['effect']['score'] == record['scores']['effect'] == record['effect'] * 4
            for k in ['sympathy', 'sycophancy']:
                assert raw[k]['noul'] == record['scores'][k]
            values.append(state)
        eligible = [r for r in records[1:] if r['scores']['effect'] >= 3 and r['effect'] > records[0]['effect']]
        assert choice['accepted'] == eligible
        selected = max(eligible, key=lambda r: r['effect']) if eligible else None
        assert choice['selected'] == selected
        if selected is not None:
            state = values[records.index(selected)]
            neutral = values[0]
            prepared.append(PreparedAudioResponse(AudioRow(tuple(neutral['input_ids']), tuple(state['input_ids']), 0, 1.),
                state['hidden'].clone(), neutral['hidden'].clone()))
            index.append({'id': row['id'], 'case_type': row['case_type'], 'history_ids': [],
                'neutral_ids': neutral['input_ids'], 'positive_ids': state['input_ids'],
                'target_instruction': selected['target'], 'selected_target': selected['name'],
                'effect': selected['effect'], 'OFF_effect': records[0]['effect'],
                'choice': pin(choice_path), 'positive_state': selected['state'], 'neutral_state': records[0]['state'],
                'target_from_raw_positive_first_hidden': True})
            coverage[row['case_type']] += 1
        audited.append(pin(choice_path))
    # Sparse coverage is reported honestly; do not lower frozen thresholds.
    assert len(index) == len({r['id'] for r in index}) and index
    cache = BASE / 'cache'
    cache.mkdir()
    atomic_save(cache / 'responses.pt', prepared)
    write_json(cache / 'index.json', {'mode': 'behavior-promptfix-fullanswer-MetaGlimmer-v4',
        'cases': len(index), 'unique_training_questions': len(index), 'rows': index,
        'responses_sha256': pin(cache / 'responses.pt')['sha256'], 'training_data': plans[0]['data'],
        'neutral_behavior_instruction': '', 'neutral_system_message_present': False, 'shared_history': [],
        'training_weighting': 'one best qualifying context per unique TRAIN question',
        'pair_coverage': dict(coverage), 'all_selected_positive_answers_native_EOS': True,
        'acceptance': 'effect>=3/4 and strictly better than matched complete OFF; curation-only target-order ties'})
    report = {'time': time.time(), 'source': pin(__file__), 'state': 'all256-full-EOS-TRAIN-firsts-audited',
        'cache_index': pin(cache / 'index.json'), 'responses': pin(cache / 'responses.pt'),
        'training_pairs': len(index), 'unique_training_questions': len(index), 'pair_coverage': dict(coverage),
        'all_candidate_choices': audited, 'source_train': plans[0]['data'],
        'shards': [pin(BASE / f'GPU{g}/plan-first.json') for g in [0, 1]],
        'actual_outer_terminals': [pin(BASE / f'GPU{g}/actual-tool-terminal-receipt-first.json') for g in [0, 1]],
        'new_TinyAya_calls_in_audit': 0, 'new_Jev_calls_in_audit': 0, 'DEV_or_test_rows_in_training': 0,
        'new_training_updates': 0, 'thresholds_changed': False,
        'outer_process_termination': 'Both actual unified terminal outcomes collected0 and original PIDs absent'}
    write_json(BASE / 'independent-cache-audit-first.json', report)
    print(json.dumps({'audit': pin(BASE / 'independent-cache-audit-first.json'),
        'training_pairs': len(index), 'coverage': dict(coverage)}), flush=True)


if __name__ == '__main__':
    main()
