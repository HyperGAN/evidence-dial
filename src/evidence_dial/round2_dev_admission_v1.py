"""Admit the eight predeclared fresh-DEV controls; never open holdout rows."""
import json
from pathlib import Path

from .long_measure_admission_v13 import pin, verified, backend_for, validate_arm
from .io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/fixed-formulation-pursuit-20261112-v1'
FRESH = BASE / 'fresh-round2-evaluation192-256-v1'


def validate_DEV_plan(path, *, preparing=False):
    import torch
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    assert plan['schema'] == 'round2-fresh-DEV-controls-only-v1'
    assert plan['cases'] == 192 and plan['split'] == 'dev'
    assert plan['ema_decay'] is None and plan['new_training_updates'] == plan['teacher_calls'] == 0
    assert plan['gpu'] == 0 and plan['physical_gpu_uuid'] == 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
    assert plan['holdout_inference_forbidden'] and not plan['final_confirmation_plan']
    assert plan['data_manifest'] == pin(FRESH / 'manifest-first.json')
    manifest = json.loads(verified(plan['data_manifest']).read_text())
    assert plan['data'] == manifest['files']['dev'], 'Only the frozen192 DEV file is admitted'
    assert plan['data_admission'] == pin(FRESH / 'independent-data-admission-first.json')
    admission = json.loads(verified(plan['data_admission']).read_text())
    assert admission['manifest'] == plan['data_manifest']
    assert admission['state'] == 'pass-offline-round2-data-construction-admission'
    assert plan['control_declaration'] == pin(FRESH / 'prospective-pursuit-before-modelcalls-first.json')
    declaration = json.loads(verified(plan['control_declaration']).read_text())
    arms = plan['arms']
    assert len(arms) == len({a['name'] for a in arms}) == 8
    assert arms[0] == {'name': 'off', 'kind': 'off'}
    pg = [a for a in arms if a['kind'] == 'particle']
    sft = [a for a in arms if a['kind'] == 'sft']
    assert [a['strength'] for a in pg] == declaration['retained_parent_DEV_strengths'] == [0.975, 1.05, 1.1]
    assert [a['step'] for a in sft] == declaration['fixed_SFT_control_steps'] == [192, 384, 576, 768]
    completed = json.loads(verified(declaration['training_parent']).read_text())['completed_study']
    parent_run = verified(completed['recipe']).parent / 'runs/first-answer'
    assert all(a['step'] == 1024 and a['artifact'] == pin(parent_run / 'NATIVE-001024.pt')
        and a['FULL'] == pin(parent_run / 'FULL-001024.pt') and a['backend'] == 'canonical_stop256' for a in pg)
    sft_registry = json.loads(verified(declaration['SFT_registry']).read_text())
    expected = {a['step']: a['raw_artifact'] for a in sft_registry['scores']}
    assert all(a['artifact'] == expected[a['step']] for a in sft)
    for item in plan['inputs'] + plan['sources']:
        verified(item)
    source = ROOT / 'src/evidence_dial/round2_dev_admission_v1.py'
    assert pin(source) in plan['sources']
    rows = read_jsonl(verified(plan['data']))
    assert len(rows) == len({r['id'] for r in rows}) == 192
    assert all(r['split'] == 'dev' for r in rows)
    admitted = {a['name']: validate_arm(a) for a in arms}
    if not preparing:
        gate = json.loads((path.parent / 'CPU-qualification-first.json').read_text())
        assert gate['state'] == 'pass-CPU-round2-DEV-control-admission'
        assert gate['plan'] == pin(path) and gate['admission_source'] == pin(source)
        for item in gate['sources']:
            verified(item)
    assert not torch.cuda.is_initialized()
    return plan, admitted
