"""Exercise the confirmation scorer offline using already saved DEV captures.

This is a CPU fixture, never confirmation data or new model output. Complete
captures exercise the actual cached-generation path, AuditKIT integration and
shared first-judgment cache. Failure fixtures must remain unmeasured.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from mine_behavior_promptfix_v4 import BASE, read, pin
OUT = BASE / 'confirmation-offline-qualification-v6-first'


def main():
    import torch
    from auditkit.errors import ModelError
    from evidence_dial import behavior_promptfix_confirm_v5 as evaluator
    from evidence_dial import behavior_protocol_v1 as protocol
    from evidence_dial.continuation_resources_v1 import CPUObserverResources
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.io import read_jsonl, write_json
    from evidence_dial.protocol import digest

    assert not OUT.exists(), 'Keep the original fixture; do not replay this qualification'
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    admission_path = BASE / 'GPU0-quarter256-independent-admission-first.json'
    admission = read(admission_path)
    assert admission['all48_answers_native_EOS'] and admission['no_weight_EMA']
    arm = next(a for a in admission['arms'] if a['strength'] == 1.)
    assert pin(arm['predictions']['path']) == arm['predictions']
    # Fixture selection is by declared order and length, never by effect score.
    records = [r for r in read_jsonl(arm['predictions']['path'])
               if len(r['native']['output_ids']) > 64][:2]
    assert len(records) == 2
    rows = [{'id': r['id'], 'question': r['question'], 'case_type': r['case_type'], 'split': 'dev'}
            for r in records]
    source_native = Path(arm['predictions']['path']).parent / 'native'
    off_path = Path(admission['control_admission']['path']).parent / 'off/predictions.jsonl'
    off_map = {r['id']: r for r in read_jsonl(off_path)}
    off_records = [off_map[r['id']] for r in records]
    assert all(a['question'] == b['question'] and a['output'] != b['output'] for a, b in zip(records, off_records))
    native_files = {(r['id'], digest(r['output'])): folder / (r['id'] + '.json')
                    for group, folder in [(records, source_native), (off_records, off_path.parent / 'native')]
                    for r in group}
    def native_source(record):
        return native_files[(record['id'], digest(record['output']))]
    write_json(OUT / 'process-first.json', {'time': time.time(), 'pid': os.getpid(),
        'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'argv': [x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]})
    sources = [pin(__file__), pin(admission_path), arm['predictions'],
        pin(Path(evaluator.__file__)), pin(Path(protocol.__file__)), pin(off_path)]
    cache = OUT / 'shared-firsts'
    cache.mkdir()
    for r in records + off_records:
        source = Path(r['scores']['source'])
        assert read(source)['http_status'] == 200
        destination = cache / source.name
        assert not destination.exists()
        destination.write_bytes(source.read_bytes())
        sources += [pin(source), pin(native_source(r))]
    write_json(OUT / 'declaration-first.json', {'time': time.time(), 'schema': 'offline-CPU-DEV-replay-only',
        'sources': sources, 'fixture_ids': [r['id'] for r in rows],
        'fresh64_rows_read': 0, 'new_model_calls': 0, 'new_Jev_calls': 0,
        'network_policy': 'All requests sessions and socket connections raise if attempted',
        'production_records_modified': False})

    def captures(name, items, *, arm_name=None, unfinished=False):
        folder = OUT / name
        for r in items:
            value = copy.deepcopy(read(native_source(r)))
            assert value['native'] == r['native'] and value['output'] == r['output']
            value['identity']['arm'] = arm_name or name
            if unfinished:
                assert value['native']['output_ids'][-1] == 3
                value['native']['output_ids'] = value['native']['output_ids'][:-1]
                value['termination'] = 'max_new_tokens'
            write_json(folder / 'native' / (r['id'] + '.json'), value)
        return folder

    def reject_network(*args, **kwargs):
        raise AssertionError('Network forbidden in offline DEV qualification')

    class NoForwardResource:
        def check(self):
            raise AssertionError('An uncached generation attempted resource allocation')

    with CPUObserverResources(OUT / 'resources-first.json') as resource:
        resource.check()
        tok = tokenizer()
        # Keep this fixture out of other jobs' AuditKIT caches.
        os.environ['XDG_CACHE_HOME'] = str(OUT / 'isolated-cache')
        with patch('requests.sessions.Session.request', side_effect=reject_network) as http, \
             patch('socket.socket.connect', side_effect=reject_network) as socket_connect, \
             patch.object(evaluator, 'ask', wraps=protocol.ask) as ask_counter, \
             patch.object(evaluator, 'generate', wraps=evaluator.generate) as native_counter:
            observed = []
            for name, expected_records in [('complete-run-A', records), ('complete-run-B', off_records), ('complete-run-C', records)]:
                arm_name = 'same-arm-label-across-distinct-runs'
                folder = captures(name, expected_records, arm_name=arm_name)
                result = evaluator.evaluate(rows, None, tok, folder, NoForwardResource(),
                    arm=arm_name, first_cache=cache)
                assert len(result) == 2
                for actual, expected in zip(result, expected_records):
                    assert actual['output'] == expected['output'] and actual['native'] == expected['native']
                    assert actual['termination'] == 'native_eos' and actual['effect'] == expected['effect']
                    assert len(actual['native']['output_ids']) > 64
                    assert actual['scores']['source'] == str(cache / Path(expected['scores']['source']).name)
                    assert actual['auditkit']['failed_count'] == 0 and not actual['auditkit']['errors']
                    prediction = actual['auditkit']['predictions'][0]
                    assert prediction['sample_id'] == expected['id'] and prediction['score'] == expected['effect']
                    assert actual['auditkit']['headline']['intended_behavior'] == expected['effect']
                observed.append(result)
            assert [r['scores']['source'] for r in observed[0]] == [r['scores']['source'] for r in observed[2]]
            assert [r['scores']['source'] for r in observed[0]] != [r['scores']['source'] for r in observed[1]]
            assert native_counter.call_count == 6, 'AuditKIT incorrectly skipped a run via a stale cache result'
            for row_number in [0, 1]:
                assert len({group[row_number]['auditkit']['fingerprint'] for group in observed}) == 3
            assert len(list(cache.glob('*.json'))) == 4 and not list(cache.glob('*call-commit*'))
            successful_cache_lookups = ask_counter.call_count

            name = 'unfinished-must-remain-unmeasured'
            folder = captures(name, records[:1], unfinished=True)
            before = ask_counter.call_count
            try:
                evaluator.evaluate(rows[:1], None, tok, folder, NoForwardResource(), arm=name, first_cache=cache)
            except (RuntimeError, ModelError) as error:
                assert 'unmeasured' in str(error) or 'Saved answer unfinished' in str(error)
            else:
                raise AssertionError('Unfinished answer was accepted')
            assert ask_counter.call_count == before, 'Unfinished prose reached the effect judge'
            saved = read(folder / 'native' / (rows[0]['id'] + '.json'))
            assert saved['native']['output_ids'][-1] != 3 and saved['termination'] == 'max_new_tokens'
            assert not (folder / 'summary.json').exists() and not (folder / 'predictions.jsonl').exists()
            assert not list((folder / 'records').glob('*.json'))

            name = 'unknown-committed-judge-call'
            folder = captures(name, records[:1])
            pending_cache = OUT / 'unknown-call-cache'
            request = {'model': protocol.JEV,
                'state': {'user_message': rows[0]['question'], 'response': records[0]['output']},
                'questions': protocol.QUESTIONS}
            key = digest({'endpoint': protocol.ENDPOINT, 'request': request})
            write_json(pending_cache / (key + '-call-commit.json'),
                {'fixture_only': True, 'request_sha256': digest(request)})
            try:
                evaluator.evaluate(rows[:1], None, tok, folder, NoForwardResource(), arm=name, first_cache=pending_cache)
            except (RuntimeError, ModelError) as error:
                assert 'unmeasured' in str(error) or 'Unresolved committed Jev call' in str(error)
            else:
                raise AssertionError('Unknown committed judgment was rerated')
            assert not (pending_cache / (key + '.json')).exists()
            assert not (folder / 'summary.json').exists() and not list((folder / 'records').glob('*.json'))
            assert http.call_count == socket_connect.call_count == 0
            assert not torch.cuda.is_initialized()
        resource.check()
    report = {'time': time.time(), 'state': 'offline-confirmation-DEV-replay-checks-passed',
        'source': pin(__file__), 'declaration': pin(OUT / 'declaration-first.json'),
        'complete_fixture_answers': 6, 'unique_saved_judgments_reused': 4,
        'same_arm_label_different_run_answers_and_fingerprints_preserved': True,
        'successful_cache_lookups': successful_cache_lookups,
        'actual_AuditKIT_scoring_path_executed': True, 'actual_cached_native_generation_path_executed': True,
        'long_answers_exceed64_tokens_preserved': True,
        'unfinished_answer_not_judged_or_completed': True,
        'unknown_committed_call_not_rerated_or_completed': True,
        'new_model_calls': 0, 'new_Jev_calls': 0, 'network_calls': 0, 'CUDA_initialized': False,
        'fresh64_rows_read': 0, 'production_sources_or_records_modified': False,
        'scope': 'Offline CPU plumbing qualification; not native GPU generation, new model performance or confirmation evidence'}
    write_json(OUT / 'checks-passed-first.json', report)
    print(json.dumps({'report': pin(OUT / 'checks-passed-first.json'), 'checks_passed': True,
        'new_model_or_Jev_calls': 0, 'fresh64_rows_read': 0}))


if __name__ == '__main__':
    main()
