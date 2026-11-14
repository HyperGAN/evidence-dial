"""Bind an exact behavior strength to four saved DEV outputs, without generation.

The unchanged qualified observer runs separately against these actual outputs.
The completed development search fixes the candidate identities before this
categorical check; no effect score or preservation criterion changes selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(closure_path, name, destination):
    from evidence_dial.io import read_jsonl, write_json
    from evidence_dial.behavior_fullanswer_health_v3 import passing_guard
    closure = json.loads(closure_path.read_text())
    assert closure['state'] == 'declared-two-rate-24-cell-DEV-search-complete'
    assert len(closure['all24_arms']) == 24 and closure['holdout_rows_read'] == 0
    maximum = max(a['effect_ON_minus_OFF'] for a in closure['all24_arms'])
    assert closure['all_maximum_ties'] == [a for a in closure['all24_arms'] if a['effect_ON_minus_OFF'] == maximum]
    arm = next(a for a in closure['all24_arms'] if a['name'] == name)
    display = closure['display_checkpoint_representative']
    assert arm in closure['all_maximum_ties'] or arm['artifact'] == display['artifact']
    for field in ['plan', 'artifact', 'FULL', 'quarter_admission', 'predictions', 'summary']:
        assert pin(arm[field]['path']) == arm[field]
    audit = json.loads(Path(arm['quarter_admission']['path']).read_text())
    assert audit['state'] == 'declared-complete-behavior-quarter-independently-readmitted'
    assert audit['all48_native_inputs_outputs_typed_Jev_scores_verified'] and audit['no_weight_EMA']
    rows = read_jsonl(arm['predictions']['path'])
    probes = [next(r for r in rows if r['case_type'] == kind)
        for kind in ['discouragement', 'unsupported_conclusion', 'reasonable_boundary', 'uncertainty']]
    assert len({p['id'] for p in probes}) == 4
    for p in probes:
        ids = p['native']['output_ids']
        assert 1 <= len(ids) <= 4096 and ids[-1] == 3
    run = Path(arm['FULL']['path']).parent
    if arm['strength'] == 1:
        guard = passing_guard(run, arm['step'], Path(arm['FULL']['path']))
        original = json.loads((run / f'probes-{arm["step"]:06d}.json').read_text())
        assert original['strength'] == 1.
        expected = [{'id': p['id'], 'output': p['output'], 'native': p['native'], 'question':p['question']} for p in probes]
        if guard is not None and original['probes'] == expected:
            return {'state': 'exact-DEV-strength-qualified-check-reused', 'arm': arm,
                'guard_folder': str(run), 'qualified_guard': pin(guard),
                'guard_session': pin(run / 'session.json'),
                'guard_probes': pin(run / f'probes-{arm["step"]:06d}.json'),
                'unchanged_outputs_verified': True, 'new_model_calls': 0, 'new_generation': 0}
    assert not destination.exists()
    destination.mkdir(parents=True)
    write_json(destination / 'session.json', {'time': time.time(), 'source': pin(__file__),
        'development_closure': pin(closure_path), 'arm': arm, 'strength': arm['strength'],
        'FULL': arm['FULL'], 'artifact': arm['artifact'], 'ema_decay': None,
        'probe_rule': 'First retained DEV question of each of four case types, irrespective of output',
        'new_model_calls': 0, 'new_training_updates': 0, 'holdout_rows_read': 0})
    write_json(destination / f'probes-{arm["step"]:06d}.json', {'checkpoint': arm['step'],
        'strength': arm['strength'], 'probes': probes, 'no_new_generation': True})
    return {'state': 'exact-DEV-strength-probes-prepared-observer-required', 'arm': arm,
        'guard_folder': str(destination.resolve()), 'guard_session': pin(destination / 'session.json'),
        'guard_probes': pin(destination / f'probes-{arm["step"]:06d}.json'),
        'new_model_calls': 0, 'new_generation': 0, 'qualified_observer_still_required': True}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--closure', type=Path, required=True)
    ap.add_argument('--name', required=True); ap.add_argument('--guard-folder', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    assert not args.output.exists(); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from evidence_dial.io import write_json
    result = prepare(args.closure, args.name, args.guard_folder)
    write_json(args.output, {'time': time.time(), 'source': pin(__file__), **result})
    print(json.dumps({'preparation': pin(args.output), 'state': result['state'], 'name': args.name,
        'new_model_calls': 0}))


if __name__ == '__main__':
    main()
