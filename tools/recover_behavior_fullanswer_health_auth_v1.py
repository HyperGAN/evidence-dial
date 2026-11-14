"""Retry a known HTTP401 health transport using the refreshed local credential.

Keep the original denied request/response at its exact path. No successful or
unknown judgment is rerolled; unchanged saved probes use the unchanged qualified
health observer and thresholds. No Tiny Aya operation occurs here.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--step', type=int, required=True)
    args = ap.parse_args()
    from followup_nanogpt_route import load_key
    from evidence_dial.behavior_fullanswer_health_v3 import BASE, ENDPOINT, JEV, QUESTIONS, ask, pin, passing_guard
    from evidence_dial.protocol import digest
    from evidence_dial.io import write_json
    os.environ['NANOGPT_API_KEY'] = load_key()
    out = args.candidate.resolve()
    step = args.step
    guard = out / f'guard-{step:06d}-fullanswer-Jev-v2.json'
    assert not guard.exists() and not (out / 'candidate-failed.json').exists()
    qualification = BASE / 'fullanswer-health-qualification-v3-first.json'
    q = json.loads(qualification.read_text())
    import evidence_dial.behavior_fullanswer_health_v3 as observer
    assert q['qualified'] and q['source'] == pin(observer.__file__)
    probes = out / f'probes-{step:06d}.json'
    rows = json.loads(probes.read_text())['probes']
    assert len(rows) == 4
    denied = []
    for row in rows:
        request = {'model': JEV, 'state': {'user_message': row['question'], 'response': row['output']}, 'questions': QUESTIONS}
        key = digest({'endpoint': ENDPOINT, 'request': request})
        path = out / 'health-Jev-firsts' / (key + '.json')
        commit = path.with_name(key + '-call-commit.json')
        if path.exists():
            value = json.loads(path.read_text())
            assert value['request'] == request and value['http_status'] == 401
            assert value['response'] == {'http_status': 401} and value['response_sha256'] == digest(value['response'])
            denied.append(pin(path))
        else:
            assert not commit.exists(), 'Unknown committed request cannot be retried'
    assert denied, 'This recovery is only for observed authentication rejection'
    observations = []
    for row in rows:
        assert row['native']['output_ids'][-1] == 3
        observations.append({'id': row['id'], **ask(row['question'], row['output'],
            out / f'health-Jev-auth-recovery-{step:06d}-firsts')})
    status = ('failed' if any(r['status'] == 'failed' for r in observations)
        else 'pass' if all(r['status'] == 'pass' for r in observations) else 'unmeasured')
    value = {'time': time.time(), 'status': status, 'checkpoint': pin(out / f'FULL-{step:06d}.pt'),
        'probes': pin(probes), 'qualification': pin(qualification), 'observations': observations,
        'effect_used_for_health': False, 'known_HTTP401_preserved': denied,
        'execution_recovery_source': pin(__file__), 'new_model_generations': 0,
        'successful_or_unknown_judgments_rerolled': False}
    write_json(guard, value)
    if status == 'failed':
        write_json(out / 'candidate-failed.json', value)
    assert status == 'pass', 'Health failed or unmeasured; no next update'
    assert passing_guard(out, step, out / f'FULL-{step:06d}.pt') == guard
    print(json.dumps({'status': status, 'guard': pin(guard), 'known401preserved': denied,
        'new_TinyAya_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
