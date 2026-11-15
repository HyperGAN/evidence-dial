"""Qualify the public full-answer owner using a real raw packet and saved DEV.

Only the model-loading boundary is replaced with a CPU owner. The actual
packet reader, capabilities, tokenizer and cached full-answer generator run.
No model forward, CUDA allocation, judge request or fresh test row is used.
"""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from mine_behavior_promptfix_v4 import BASE, read, pin
OUT = BASE / 'public-fullanswer-owner-qualification-first'


def main():
    import torch
    from evidence_dial import behavior_inference_v5 as owner_api
    from evidence_dial.binding import CAPS
    from evidence_dial.continuation_resources_v1 import CPUObserverResources
    from evidence_dial.io import read_jsonl, write_json

    assert not OUT.exists()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    admission_path = BASE / 'GPU0-quarter256-independent-admission-first.json'
    admission = read(admission_path)
    candidate = next(a for a in admission['arms'] if a['strength'] == 1.)
    assert pin(candidate['artifact']['path']) == candidate['artifact']
    records = read_jsonl(candidate['predictions']['path'])
    record = next(r for r in records if len(r['native']['output_ids']) > 64)
    native_path = Path(candidate['predictions']['path']).parent / 'native' / (record['id'] + '.json')
    original = read(native_path)
    assert original['termination'] == 'native_eos' and original['native']['output_ids'][-1] == 3
    write_json(OUT / 'process-first.json', {'time': time.time(), 'pid': os.getpid(),
        'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'argv': [x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]})
    write_json(OUT / 'declaration-first.json', {'time': time.time(), 'scope': 'Offline CPU owner contract only',
        'sources': [pin(__file__), pin(Path(owner_api.__file__)), pin(admission_path),
            candidate['artifact'], candidate['predictions'], pin(native_path)],
        'new_model_calls': 0, 'new_Jev_calls': 0, 'fresh64_rows_read': 0})

    class Slider:
        def __init__(self):
            self.strength = 1.
        @contextmanager
        def scaled(self, strength):
            previous = self.strength
            self.strength = strength
            try:
                yield
            finally:
                self.strength = previous

    class Native:
        def __init__(self):
            self.model, self.slider = None, Slider()
            self.legacy_calls = 0
        def response(self, **kwargs):
            self.legacy_calls += 1
            raise AssertionError('The legacy64-token callback must not be called')

    native = Native()
    lifecycle = {'opened': 0, 'closed': 0}
    @contextmanager
    def cpu_open(packet, *, backend, operation):
        assert isinstance(packet, dict) and packet['selection'] == 'raw'
        assert packet['specification']['configuration']['ema_decay'] is None
        assert backend.name == packet['specification']['backend'] and operation == 'response'
        lifecycle['opened'] += 1
        try:
            yield SimpleNamespace(binding=SimpleNamespace(response=native.response), capabilities=CAPS)
        finally:
            lifecycle['closed'] += 1

    def reject_network(*args, **kwargs):
        raise AssertionError('Network is forbidden in this offline fixture')

    with CPUObserverResources(OUT / 'resources-first.json') as resource:
        with patch.object(owner_api, 'open_inference', side_effect=cpu_open), \
             patch('requests.sessions.Session.request', side_effect=reject_network) as http, \
             patch('socket.socket.connect', side_effect=reject_network) as socket_connect:
            with owner_api.open_behavior_inference(candidate['artifact']['path'], resource=resource,
                    expected_sha256=candidate['artifact']['sha256']) as model:
                assert model.artifact_sha256 == candidate['artifact']['sha256']
                complete = copy.deepcopy(original)
                complete['identity']['arm'] = model.capture_identity(1.)
                complete_path = OUT / 'complete-native.json'
                write_json(complete_path, complete)
                before = pin(complete_path)
                value = model.respond(text=record['question'], strength=1., capture_path=complete_path)
                assert value == complete and value['output'] == record['output']
                assert len(value['native']['output_ids']) > 64 and value['native']['output_ids'][-1] == 3
                assert value['identity']['decoding']['max_new_tokens'] == 4096
                assert pin(complete_path) == before and native.slider.strength == 1.
                assert native.legacy_calls == 0

                try:
                    model.respond(text=record['question'], strength=.5, capture_path=complete_path)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('A saved answer from a different strength was accepted')
                assert pin(complete_path) == before and native.slider.strength == 1.
                assert model.capture_identity(.1234561) != model.capture_identity(.1234562)

                unfinished = copy.deepcopy(complete)
                unfinished['native']['output_ids'] = unfinished['native']['output_ids'][:-1]
                unfinished['termination'] = 'max_new_tokens'
                unfinished_path = OUT / 'unfinished-native.json'
                write_json(unfinished_path, unfinished)
                before_unfinished = pin(unfinished_path)
                try:
                    model.respond(text=record['question'], strength=1., capture_path=unfinished_path)
                except AssertionError as error:
                    assert 'unfinished' in str(error)
                else:
                    raise AssertionError('An unfinished saved answer was returned')
                assert pin(unfinished_path) == before_unfinished and native.slider.strength == 1.
                try:
                    model.respond(text=record['question'], strength=float('nan'), capture_path=complete_path)
                except ValueError as error:
                    assert 'finite' in str(error)
                else:
                    raise AssertionError('Nonfinite strength was accepted')
            assert lifecycle == {'opened': 1, 'closed': 1}
            try:
                model.respond(text=record['question'], capture_path=complete_path)
            except RuntimeError as error:
                assert 'closed' in str(error)
            else:
                raise AssertionError('A closed owner was used')
            try:
                with owner_api.open_behavior_inference(candidate['artifact']['path'], resource=resource,
                        expected_sha256='0' * 64):
                    raise AssertionError('Wrong artifact entered the model context')
            except ValueError as error:
                assert 'SHA-256' in str(error)
            assert lifecycle['opened'] == 1
            fake_ema = {'selection': 'ema', 'specification': {'configuration': {'ema_decay': .9}}}
            with patch.object(owner_api, 'read_inference', return_value=fake_ema):
                try:
                    with owner_api.open_behavior_inference(candidate['artifact']['path'], resource=resource):
                        raise AssertionError('EMA artifact entered the model context')
                except ValueError as error:
                    assert 'Raw weights' in str(error)
            assert lifecycle['opened'] == 1 and native.legacy_calls == 0
            assert http.call_count == socket_connect.call_count == 0 and not torch.cuda.is_initialized()
            assert pin(candidate['artifact']['path']) == candidate['artifact']
    report = {'time': time.time(), 'state': 'public-fullanswer-owner-offline-CPU-checks-passed',
        'source': pin(__file__), 'owner_source': pin(Path(owner_api.__file__)),
        'declaration': pin(OUT / 'declaration-first.json'), 'native_tokens_preserved': len(value['native']['output_ids']),
        'legacy64_response_calls': native.legacy_calls, 'actual_cached_fullanswer_generator_exercised': True,
        'strength_cache_identity_and_scope_restoration': True, 'unfinished_capture_preserved_and_rejected': True,
        'wrong_artifact_and_EMA_rejected_before_model_load': True, 'closed_owner_rejected': True,
        'model_context_opens': lifecycle['opened'], 'model_context_closes': lifecycle['closed'],
        'new_model_calls': 0, 'new_Jev_calls': 0, 'network_calls': 0, 'CUDA_initialized': False,
        'fresh64_rows_read': 0, 'scope': 'CPU owner contract; actual GPU generation remains covered by existing native experiment admissions'}
    write_json(OUT / 'checks-passed-first.json', report)
    print(json.dumps({'report': pin(OUT / 'checks-passed-first.json'), 'checks_passed': True,
        'native_tokens_preserved': report['native_tokens_preserved'], 'new_model_or_Jev_calls': 0}))


if __name__ == '__main__':
    main()
