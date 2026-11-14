"""Evidence admission for the complete finite round-two experiment.

Read saved RAW states and fresh DEV captures only. No model forward, training,
holdout-row access, averaging, or incomplete-budget selection is supported.
"""
import json
import fcntl
from pathlib import Path

from .long_measure_admission_v13 import pin, verified, validate_arm, require_root_identity, backend_for
from .io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/fixed-formulation-pursuit-20261112-v1'
FRESH = BASE / 'fresh-round2-evaluation192-256-v1'
CONTROLS = BASE / 'fresh-round2-DEV-controls-GPU0-v1'
TRAINING = Path('/mnt/nvme/lexica-evidence-dial-work/01a1c84b-5d17-7e32-80ef-34d8e4ee2e62/canonical-stop-TRAIN256-exact1024to2048-GPU1-round2-20261113-v1')
RECIPE = {'path': str(TRAINING / 'recipe.json'), 'sha256': 'ebd469eb841360ee7154ff1c5716e6c54a08bc67b95251f204de58b7e2820d53'}
CONTROL_ADMISSION = {'path': str(CONTROLS / 'independent-completion-audit-first.json'),
    'sha256': 'a2b9a86959aeb91ab91fb8ef55ed3f70aecf425235c247f44038b62e830fdaef'}
QUARTER_AUDITOR = {'path': str(ROOT / 'tools/audit_round2_pg_continue2048_quarter_v1.py'),
    'sha256': '8a2374e2ff3e68bc975695996ff99dfd4ef9950fd9d3573da4515a63b2231c44'}
QUARTERS = [1280, 1536, 1792, 2048]
STRENGTHS = [0.9, 0.975, 1.0, 1.05, 1.1]


def controls():
    proof = json.loads(verified(CONTROL_ADMISSION).read_text())
    verified(proof['source'])
    assert proof['state'] == 'complete-round2-DEV-controls-independently-readmitted'
    assert proof['all_own_processes_absent_and_private_leases_released']
    assert proof['all_locked_arms_executed_exactly_once'] and proof['execution_attempts'] == 8
    comparison = json.loads(verified(proof['comparison']).read_text())
    plan = json.loads(verified(comparison['plan']).read_text())
    from .round2_dev_admission_v1 import validate_DEV_plan
    validate_DEV_plan(Path(comparison['plan']['path']))
    assert comparison['complete'] and comparison['holdout_rows_read'] == 0
    assert len(comparison['results']) == 8 and len(comparison['PG_vs_SFT_all_control_pairs']) == 12
    return comparison, plan, proof


def DEV_verifier():
    from .evaluate import tokenizer
    from .protocol import prompt, digest, success
    manifest = json.loads((FRESH / 'manifest-first.json').read_text())
    admission = json.loads((FRESH / 'independent-data-admission-first.json').read_text())
    assert admission['manifest'] == pin(FRESH / 'manifest-first.json')
    assert admission['state'] == 'pass-offline-round2-data-construction-admission'
    rows = read_jsonl(verified(manifest['files']['dev']))
    assert len(rows) == len({r['id'] for r in rows}) == 192 and all(r['split'] == 'dev' for r in rows)
    tok = tokenizer()
    expected = {r['id']: tok.apply_chat_template([{'role': 'user', 'content': prompt(r)}],
        add_generation_prompt=True) for r in rows}
    checked = {}

    def verify(predictions, label):
        path = verified(predictions)
        key = (predictions['sha256'], label)
        if key in checked:
            return checked[key]
        records = read_jsonl(path)
        assert [r['id'] for r in records] == [r['id'] for r in rows]
        for row, record in zip(rows, records):
            capture = json.loads((path.parent / 'native' / f'{row["id"]}.json').read_text())
            ids = capture['native']['output_ids']
            assert 1 <= len(ids) <= 64 and (ids[-1] == 3 or len(ids) == 64)
            assert record['native']['input_ids'] == capture['native']['input_ids'] == expected[row['id']]
            assert record['native']['output_ids'] == ids
            assert record['output'] == capture['output'] == tok.decode(ids, skip_special_tokens=True)
            assert record['prompt_sha256'] == capture['prompt_sha256'] == digest(prompt(row))
            assert record['arm'] == label and not record['teacher']
            assert record['case_type'] == row['case_type']
            assert record['success'] == int(success(row, record['output']))
        summary = json.loads((path.parent / 'summary.json').read_text())
        correct = sum(r['success'] for r in records)
        assert summary['complete'] and summary['declared'] == summary['measured'] == 192
        assert summary['arm'] == label and summary['success_rate'] == correct / 192
        checked[key] = correct
        return correct

    return verify


def raw_PG(arm):
    import torch
    from particle_sliders.lifecycle import read_packet
    from particle_sliders.inference import read_inference, InferenceSpec
    from .train import tensor_equal
    from .qualified_text_guard_admission_v1 import passing_guard
    full, artifact = verified(arm['FULL']), verified(arm['artifact'])
    packet, native = read_packet(full), read_inference(artifact)
    game = packet['payload']['game']
    assert full.parent == artifact.parent == Path(arm['candidate'])
    assert game['completed_steps'] == native['origin']['completed_steps'] == arm['step']
    assert packet['payload']['ema'] == {'decay': None, 'values': None}
    assert game['config']['recipe']['ema_decay'] == 0. and native['selection'] == 'raw'
    assert tensor_equal(game['network'], native['network'])
    assert arm['backend'] == 'canonical_stop256'
    factory = backend_for(arm)
    assert native['specification']['backend'] == factory.name
    assert native['specification']['configuration'] == {'mode': 'first-answer', 'rank': 8,
        'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None}
    factory.validate({}, InferenceSpec(native['specification']))
    guard = passing_guard(artifact.parent, arm['step'], full)
    assert guard is not None and pin(guard) == arm['checkpoint_guard']
    assert not torch.cuda.is_initialized()


def parent_and_SFT_cells(verify):
    comparison, plan, _ = controls()
    by_name = {r['arm']: r for r in comparison['results']}
    off = next(r for r in comparison['results'] if r['kind'] == 'off')
    off_correct = verify(off['predictions'], 'off')
    assert off_correct == off['correct']
    pg, sft = [], []
    for original in plan['arms'][1:]:
        validate_arm(original)
        result = by_name[original['name']]
        label = (f'particle-raw-{original["artifact"]["sha256"]}-strength-{original["strength"]:g}'
            if original['kind'] == 'particle' else 'sft-raw-' + original['artifact']['sha256'])
        correct = verify(result['predictions'], label)
        assert correct == result['correct'] and result['cases'] == 192
        arm = {**original, 'fresh_development_predictions': result['predictions'],
            'fresh_development_audit': CONTROL_ADMISSION,
            'legacy_qualified_descriptor': original}
        cell = {'name': original['name'], 'artifact': original['artifact'], 'step': original['step'],
            'kind': original['kind'], 'correct': correct, 'cases': 192, 'OFF_correct': off_correct,
            'effect_ON_minus_OFF': (correct - off_correct) / 192, 'predictions': result['predictions'], 'arm': arm}
        if original['kind'] == 'particle':
            cell['strength'] = original['strength']
            pg.append(cell)
        else:
            sft.append(cell)
    assert len(pg) == 3 and [r['step'] for r in sft] == [192, 384, 576, 768]
    return pg, sft, off_correct


def complete_cells(*, closure_override=None):
    """Require real completion before reading any unfinished training state."""
    closure_path = TRAINING / 'actual-root0-closure-first.json'
    assert closure_override is not None or closure_path.exists(), 'Finish the declared2048 budget, all DEV grids and actual root closure first'
    closure = closure_override if closure_override is not None else json.loads(closure_path.read_text())
    verified(closure['source'])
    assert closure['recipe'] == RECIPE
    assert closure['actual_root_identity'] == pin(TRAINING / 'actual-live-controller-identity-first.json')
    identity = json.loads((TRAINING / 'actual-live-controller-identity-first.json').read_text())
    terminal = json.loads(verified(closure['actual_terminal_receipt']).read_text())
    require_root_identity(terminal, identity)
    assert terminal['actual_tool_terminal_chunk'] and not Path(f'/proc/{identity["pid"]}').exists()
    assert closure['private_leases_released'] and closure['new_updates'] == 1024
    recipe = json.loads(verified(RECIPE).read_text())
    for item in recipe['frozen_inputs'] + recipe['sources']:
        verified(item)
    declaration_path = TRAINING / 'queue-declaration-admitted-v2-first.json'
    declaration = json.loads(declaration_path.read_text())
    assert identity['declaration'] == pin(declaration_path)
    state = json.loads((TRAINING / (declaration['name'] + '-state.json')).read_text())
    assert state['state'] == 'training-and-development-complete'
    for path in [TRAINING / '.shared-queue-controller.lock', TRAINING / 'runs/first-answer/.shared-gpu-job.lock']:
        with path.open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    verified(QUARTER_AUDITOR)
    verify = DEV_verifier()
    cells, sft, off_correct = parent_and_SFT_cells(verify)
    assert recipe['development_checkpoints'] == QUARTERS and recipe['new_model_development_strengths'] == STRENGTHS
    audit_pins = []
    for step in QUARTERS:
        path = BASE / f'round2-canonical-TRAIN256-PG{step}-finished-DEV-independent-audit-first.json'
        audit = json.loads(path.read_text())
        assert audit['auditor_source_sha256'] == QUARTER_AUDITOR['sha256']
        assert audit['recipe'] == RECIPE and audit['milestone'] == step and audit['budget'] == 2048
        assert audit['inherited_updates'] == 1024 and audit['new_updates_at_milestone'] == step - 1024
        assert audit['all_declared_DEV_strengths_finished'] and audit['no_weight_EMA']
        assert audit['continuation_parent1024_prefix_and_exact_native_import_verified']
        assert audit['DEV_cases'] == 192 and audit['matched_OFF'] == off_correct
        assert audit['DEV_control_completion_admission'] == CONTROL_ADMISSION
        assert [a['strength'] for a in audit['DEV_arms']] == STRENGTHS
        for item in [audit['FULL'], audit['raw_export'], audit['native_artifact'], audit['qualified_guard']]:
            verified(item)
        for result in audit['DEV_arms']:
            name = f'pg-round2-step-{step}-alpha-{result["strength"]:g}'.replace('.', '_')
            correct = verify(result['predictions'],
                f'particle-raw-{audit["native_artifact"]["sha256"]}-strength-{result["strength"]:g}')
            assert correct == result['correct'] and result['cases'] == 192 and result['OFF_correct'] == off_correct
            assert result['effect_ON_minus_matched_OFF'] == (correct - off_correct) / 192
            arm = {'name': name, 'kind': 'particle', 'backend': 'canonical_stop256', 'step': step,
                'strength': result['strength'], 'artifact': audit['native_artifact'], 'FULL': audit['FULL'],
                'candidate': str(Path(audit['FULL']['path']).parent), 'checkpoint_guard': audit['qualified_guard'],
                'fresh_development_predictions': result['predictions'], 'fresh_development_audit': pin(path)}
            raw_PG(arm)
            cells.append({'name': name, 'kind': 'particle', 'step': step, 'strength': result['strength'],
                'artifact': arm['artifact'], 'correct': correct, 'cases': 192, 'OFF_correct': off_correct,
                'effect_ON_minus_OFF': (correct - off_correct) / 192, 'predictions': result['predictions'], 'arm': arm})
        audit_pins.append(pin(path))
    assert len(cells) == len({(c['artifact']['sha256'], c['strength']) for c in cells}) == 23
    return cells, sft, off_correct, audit_pins
