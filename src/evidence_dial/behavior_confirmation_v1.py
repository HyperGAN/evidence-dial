"""A locked, question-matched behavior confirmation with no test selection."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/behavior-dial-20261114-v1'
GPUS = ['GPU-72c1b506-891d-b8bc-b353-e020585e1c47', 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69']


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verified(record):
    assert pin(record['path']) == record, record['path']
    return Path(record['path'])


def read(path):
    return json.loads(Path(path).read_text())


def declared_arms(closure):
    assert closure['state'] == 'declared-two-rate-24-cell-DEV-search-complete'
    assert len(closure['all24_arms']) == 24 and closure['holdout_rows_read'] == 0
    maximum = max(a['effect_ON_minus_OFF'] for a in closure['all24_arms'])
    maxima = [a for a in closure['all24_arms'] if a['effect_ON_minus_OFF'] == maximum]
    assert closure['all_maximum_ties'] == maxima
    display = closure['display_checkpoint_representative']; assert display == maxima[0]
    grid = [a for a in closure['all24_arms'] if a['artifact'] == display['artifact']]
    assert [a['strength'] for a in grid] == [.5, 1., 2.]
    unique = []
    for arm in maxima + grid:
        if arm not in unique:
            unique.append(arm)
    return unique, maxima, display


def exact_strength_health(arm, preparation):
    from .qualified_text_guard_admission_v1 import passing_guard
    from .io import read_jsonl
    assert preparation['arm'] == arm
    for field in ['guard_session', 'guard_probes']:
        verified(preparation[field])
    for field in ['artifact', 'FULL', 'predictions', 'quarter_admission']:
        verified(arm[field])
    folder = Path(preparation['guard_folder'])
    probes = read(verified(preparation['guard_probes']))
    assert probes['checkpoint'] == arm['step'] and probes['strength'] == arm['strength']
    rows = read_jsonl(arm['predictions']['path'])
    expected = [next(r for r in rows if r['case_type'] == kind)
        for kind in ['discouragement', 'unsupported_conclusion', 'reasonable_boundary', 'uncertainty']]
    assert [(p['id'], p['output'], p['native']) for p in probes['probes']] == [
        (p['id'], p['output'], p['native']) for p in expected]
    guard = passing_guard(folder, arm['step'], Path(arm['FULL']['path']))
    assert guard is not None, 'Exact selected/display strength remains unmeasured'
    return {'guard': pin(guard), 'preparation': None,
        'session': preparation['guard_session'], 'probes': preparation['guard_probes']}


def validate_plan(plan):
    assert plan['schema'] == 'behavior-holdout32-locked-v1' and plan['ema_decay'] is None
    assert plan['cases'] == 32 and plan['max_new_tokens'] == 64
    assert plan['holdout_based_reselection'] is False
    assert plan['uncertainty'] == {'unit': 'matched_question', 'resamples': 10000, 'seed': 20261114}
    for item in plan['inputs'] + plan['sources']:
        verified(item)
    closure = read(verified(plan['development_closure']))
    particle, maxima, display = declared_arms(closure)
    assert plan['selected_development_maxima'] == [a['name'] for a in maxima]
    assert plan['display_checkpoint'] == display['artifact']
    assert plan['arms'][:2] == [{'name': 'off', 'kind': 'off', 'target': None, 'gpu': 0},
        {'name': 'prompt-only', 'kind': 'prompt-only', 'target_index': 0, 'gpu': 1}]
    assert len(plan['arms']) == len(particle) + 2
    for index, (arm, expected) in enumerate(zip(plan['arms'][2:], particle)):
        assert arm['kind'] == 'particle' and arm['candidate'] == expected
        assert arm['name'] == expected['name'] and arm['gpu'] == index % 2
        prep = read(verified(arm['health']['preparation']))
        actual = exact_strength_health(expected, prep)
        assert arm['health'] == {**actual, 'preparation': arm['health']['preparation']}
    return plan
