"""Fresh DEV-only maxima and complete-budget admission for round-two test."""
from copy import deepcopy
import json
from pathlib import Path

from .round2_complete_admission_v1 import (
    ROOT, BASE, FRESH, CONTROLS, TRAINING, RECIPE, CONTROL_ADMISSION,
    pin, verified, validate_arm, backend_for, complete_cells, raw_PG,
)

GUARD_FIELDS = {'selected_strength_guard', 'strength_guard_session', 'strength_guard_probes'}


def verify_source_qualification():
    path = BASE / 'round2-final-prospective-CPU-qualification-first.json'
    gate = json.loads(path.read_text())
    assert gate['state'] == 'pass-known-controls-and-premature-execution-refusals'
    verified(gate['source'])
    for item in gate['sources']:
        verified(item)
    return pin(path)


def selection():
    cells, sft, off, audits = complete_cells()
    verify_source_qualification()
    best = max(c['effect_ON_minus_OFF'] for c in cells)
    chosen = [deepcopy(c['arm']) for c in cells if c['effect_ON_minus_OFF'] == best]
    assert chosen and len(cells) == 23 and len(sft) == 4
    return {'cells': cells, 'SFT_cells': sft, 'OFF_correct': off, 'quarter_audits': audits,
        'PG_arms': chosen, 'SFT_arms': [deepcopy(c['arm']) for c in sft]}


def signature(value):
    fields = ['name', 'kind', 'step', 'strength', 'artifact', 'correct', 'cases', 'OFF_correct',
        'effect_ON_minus_OFF', 'predictions']
    return {'PG_scores': [{k: c[k] for k in fields if k in c} for c in value['cells']],
        'SFT_scores': [{k: c[k] for k in fields if k in c} for c in value['SFT_cells']],
        'OFF_correct': value['OFF_correct'], 'selected_PG_names': [a['name'] for a in value['PG_arms']],
        'fixed_SFT_names': [a['name'] for a in value['SFT_arms']]}


def match_expected(arm, expected):
    if 'legacy_qualified_descriptor' in expected:
        assert arm == expected, 'Preserve the original qualification and separate fresh DEV score'
    else:
        assert {k: v for k, v in arm.items() if k not in GUARD_FIELDS} == expected


def verify_health_probes(probes, *, fresh=False):
    """Verify four fixed ordinary captures; the observer judges their text only."""
    from .io import read_jsonl
    from .evaluate import tokenizer
    from .protocol import prompt
    data = FRESH / 'dev.jsonl' if fresh else ROOT / 'artifacts/answerable-followup-20261111-v1/data/dev.jsonl'
    if fresh:
        manifest = json.loads((FRESH / 'manifest-first.json').read_text())
        assert pin(data) == manifest['files']['dev']
    else:
        assert pin(data)['sha256'] == 'ddb1cdd340194cc5c653d2bd173b838040120a91df9cdd2a8e49406e2d266f6a'
    rows = read_jsonl(data)
    chosen = [next(r for r in rows if r['case_type'] == kind)
        for kind in ['answerable', 'missing', 'conflict', 'altered']]
    assert len(probes) == 4
    tok = tokenizer()
    for row, capture in zip(chosen, probes):
        assert capture['id'] == row['id']
        expected = tok.apply_chat_template([{'role': 'user', 'content': prompt(row)}], add_generation_prompt=True)
        assert capture['native']['input_ids'] == expected
        ids = capture['native']['output_ids']
        assert 1 <= len(ids) <= 64 and (ids[-1] == 3 or len(ids) == 64)
        assert capture['output'] == tok.decode(ids, skip_special_tokens=True)
    return pin(data)


def strength_guard(arm):
    """Reuse an exact existing verdict, or verify a separately captured check."""
    from .qualified_text_guard_admission_v1 import passing_guard
    if 'legacy_qualified_descriptor' in arm:
        legacy = arm['legacy_qualified_descriptor']
        assert all(arm[k] == legacy[k] for k in ['artifact', 'FULL', 'step', 'strength'])
        validate_arm(legacy)
        return True
    raw_PG(arm)
    if arm['strength'] == 1.:
        probes = json.loads((Path(arm['candidate']) / f'probes-{arm["step"]:06d}.json').read_text())
        assert probes['strength'] == 1.
        verify_health_probes(probes['probes'])
        for probe in probes['probes']:
            capture = json.loads((Path(arm['candidate']) / 'probe-native' /
                f'{arm["step"]:06d}-{probe["id"]}.json').read_text())
            assert capture['output'] == probe['output'] and capture['native'] == probe['native']
        return True  # The exact checkpoint guard already checked these four outputs.
    if not GUARD_FIELDS <= arm.keys():
        return False
    guard = verified(arm['selected_strength_guard'])
    session_path = verified(arm['strength_guard_session'])
    probes_path = verified(arm['strength_guard_probes'])
    assert session_path.parent == probes_path.parent == guard.parent
    assert session_path.name == 'session.json' and probes_path.name == f'probes-{arm["step"]:06d}.json'
    session = json.loads(session_path.read_text())
    assert all(session[k] == arm[k] for k in ['artifact', 'FULL', 'step', 'strength'])
    assert session['source'] == pin(ROOT / 'tools/prepare_round2_selected_strength_guard_v1.py')
    assert session['ema_decay'] is None and session['new_training_updates'] == session['new_model_calls'] == 0
    assert session['fresh_development_predictions'] == arm['fresh_development_predictions']
    assert session['fresh_development_audit'] == arm['fresh_development_audit']
    probes = json.loads(probes_path.read_text())
    assert probes['strength'] == arm['strength']
    from .io import read_jsonl
    records = read_jsonl(verified(arm['fresh_development_predictions']))
    expected = [next(r for r in records if r['case_type'] == kind)
        for kind in ['answerable', 'missing', 'conflict', 'altered']]
    assert probes['probes'] == expected
    assert verify_health_probes(probes['probes'], fresh=True) == session['probe_data']
    assert passing_guard(guard.parent, arm['step'], verified(arm['FULL'])) == guard
    return True


def apply_qualified_descriptors(selected, supplied):
    result, missing = [], []
    for expected in selected['PG_arms']:
        candidates = [a for a in supplied if a['name'] == expected['name']]
        assert len(candidates) <= 1
        arm = deepcopy(candidates[0] if candidates else expected)
        match_expected(arm, expected)
        if not strength_guard(arm):
            missing.append(arm['name'])
        result.append(arm)
    return result, missing


def input_pins(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            verified(value)
            return [value]
        return [p for v in value.values() for p in input_pins(v)]
    if isinstance(value, list):
        return [p for v in value for p in input_pins(v)]
    return []


def unique_pins(values):
    result = {}
    for item in values:
        verified(item)
        if item['path'] in result:
            assert result[item['path']] == item
        result[item['path']] = item
    return list(result.values())


def validate_confirmation_plan(path, *, preparing=False):
    """Rebuild all23 DEV choices; no test rows are needed for admission."""
    import torch
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    assert plan['schema'] == 'round2-complete-DEV-locked-confirmation-v1'
    assert plan['cases'] == 256 and plan['split'] == 'locked-confirmation'
    assert plan['ema_decay'] is None and plan['new_training_updates'] == plan['teacher_calls'] == 0
    assert plan['no_test_based_reselection']
    assert plan['gpu'] in [0, 1]
    assert plan['physical_gpu_uuid'] == {0: 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47',
        1: 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'}[plan['gpu']]
    for item in plan['inputs'] + plan['sources']:
        verified(item)
    assert plan['data_manifest'] == pin(FRESH / 'manifest-first.json')
    manifest = json.loads(verified(plan['data_manifest']).read_text())
    assert plan['data'] == manifest['files']['holdout']  # Hashes only; no row parsing here.
    assert plan['training_recipe'] == RECIPE
    assert plan['training_closure'] == pin(TRAINING / 'actual-root0-closure-first.json')
    lock = json.loads(verified(plan['selection_lock']).read_text())
    assert lock['state'] == 'round2-complete-budget-fresh-DEV-only-maxima-locked'
    assert lock['holdout_generated_before_lock'] is False and lock['test_based_reselection'] is False
    rebuilt = selection()
    assert lock['selection'] == signature(rebuilt)
    assert plan['arms'][0] == {'name': 'off', 'kind': 'off'}
    pg = [a for a in plan['arms'] if a['kind'] == 'particle']
    sft = [a for a in plan['arms'] if a['kind'] == 'sft']
    assert len(plan['arms']) == 1 + len(pg) + len(sft)
    assert len({a['name'] for a in plan['arms']}) == len(plan['arms'])
    assert [a['name'] for a in pg] == [a['name'] for a in rebuilt['PG_arms']]
    assert sft == rebuilt['SFT_arms'] and [a['step'] for a in sft] == [192, 384, 576, 768]
    admitted = {'off': {'kind': 'off', 'weights': 'raw-pretrained'}}
    for arm, expected in zip(pg, rebuilt['PG_arms']):
        match_expected(arm, expected)
        assert strength_guard(arm), 'Every DEV-maximum PG strength requires its exact qualified output check'
        admitted[arm['name']] = {'kind': 'particle', 'weights': 'raw', 'step': arm['step'], 'strength': arm['strength']}
    for arm in sft:
        validate_arm(arm['legacy_qualified_descriptor'])
        admitted[arm['name']] = {'kind': 'sft', 'weights': 'raw', 'step': arm['step']}
    if not preparing:
        gate = json.loads((path.parent / 'CPU-qualification-first.json').read_text())
        assert gate['state'] == 'pass-complete-round2-final-plan-CPU-admission'
        assert gate['plan'] == pin(path) and gate['PG_candidate_cells'] == 23
        for item in gate['sources']:
            verified(item)
    assert not torch.cuda.is_initialized()
    return plan, admitted
