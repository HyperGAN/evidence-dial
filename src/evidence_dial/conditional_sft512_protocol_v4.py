"""CPU-only protocol for the prospective same-TRAIN512 SFT control."""
import json
from pathlib import Path
import sys

from .long_measure_admission_v6 import ROOT, pin, verified

BASE = ROOT / 'artifacts/fixed-formulation-pursuit-20261112-v1'
DIRECTORY = BASE / 'long-checkpoint-confirmation-execution-v1'
REGISTRY = DIRECTORY / 'required-six-studies-retained-stop256-alpha-registry-first.json'
QUALIFICATION = DIRECTORY / 'conditional-SFT512-retained-stop-alpha-CPU-qualification-v4/qualification-first.json'
HEALTH = [2, 32, 64, 384, 768, 1152, 1536]
QUARTERS = [384, 768, 1152, 1536]
NEW_SOURCES = [
    'src/evidence_dial/curated512_sft_shared_v1.py',
    'src/evidence_dial/conditional_sft512_protocol_v4.py',
    'tools/curated512_sft_phase_shared_v1.py',
    'tools/curated512_sft_queue_shared_v1.py',
    'tools/audit_curated512_sft_quarter_v1.py',
    'tools/prepare_conditional_SFT512_v4.py',
    'tools/finalize_conditional_SFT512_v4.py',
    'tools/qualify_conditional_SFT512_retained_stop_CPU_v4.py',
]


def qualified_sources():
    qualification = json.loads(QUALIFICATION.read_text())
    assert qualification['state'] == 'pass-CPU-static-data-resume-qualification-only'
    assert qualification['CUDA_initialized'] is False
    assert qualification['test_rows_parsed'] == qualification['new_model_calls'] == 0
    assert {p['path'] for p in qualification['sources']} == {str((ROOT / n).resolve()) for n in NEW_SOURCES}
    for item in qualification['sources'] + qualification['parent_sources']:
        verified(item)
    return qualification


def require_selected_TRAIN512(path):
    """Rebuild complete real DEV selection before permitting a control plan."""
    sys.path.insert(0, str(ROOT / 'tools'))
    from freeze_long_confirmation_v4 import complete_cells, selection_from_cells
    registry = json.loads(REGISTRY.read_text())
    selection = json.loads(Path(path).read_text())
    assert selection['required_registry'] == pin(REGISTRY)
    assert selection['selection_finally_locked'] is False and not selection['test_generated']
    assert selection['same512_data_control_required_and_missing'] is True
    studies = json.loads(verified(selection['completed_studies']).read_text())['completed_studies']
    cells, origins = complete_cells(registry, studies)
    rebuilt = selection_from_cells(cells, origins, registry)
    for key in ['particle_finished_DEV_scores', 'sft_finished_DEV_scores', 'selected_PG_ties', 'selected_SFT_ties']:
        assert selection[key] == rebuilt[key], key
    assert any(arm['backend'] == 'curated512' for arm in rebuilt['PG_descriptors'].values())
    return registry, rebuilt


def training_recipe(registry):
    return {**registry['conditional_TRAIN512_SFT_control']['training'],
        'train_cases': 512, 'dataloader_drop_last': False,
        'comparison': 'Same512 canonical TRAIN questions/gold as PG; same rank8 q/k/v/o baseline, original constant AdamW law. Steps do not imply equal compute or completion-token exposure.'}


def verify_plan(path):
    qualified_sources()
    spec = json.loads(Path(path).read_text())
    for item in spec['inputs'] + spec['sources']:
        verified(item)
    registry = json.loads(REGISTRY.read_text())
    assert spec['schema'] == 'matched-canonical-TRAIN512-SFT1536-v1'
    assert spec['training'] == registry['conditional_TRAIN512_SFT_control']['training']
    assert pin(Path(spec['training_data'])) == registry['conditional_TRAIN512_SFT_control']['training_data']
    assert spec['health_checkpoints'] == HEALTH and spec['development_checkpoints'] == QUARTERS
    return spec
