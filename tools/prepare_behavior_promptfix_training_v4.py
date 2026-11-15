"""Admit two fresh RAW/noEMA1024 cohorts after complete-answer TRAIN curation."""
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from mine_behavior_promptfix_v4 import BASE, DIAG, pin, read, GPUS


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from evidence_dial.io import write_json
    from evidence_dial.behavior_promptfix_binding_v4 import NAME, PROTOCOL, Native, initialize, validate
    from evidence_dial.binding import Native as OriginalNative, initialize as OriginalInitialize
    from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter, AudioResponseCritic
    from particle_sliders.pg09 import PG_VERSION
    from particle_sliders.lifecycle import SessionConfig
    from evidence_dial.protocol import SEED
    admission = BASE / 'independent-cache-audit-first.json'
    admitted = read(admission)
    assert admitted['state'] == 'all256-full-EOS-TRAIN-firsts-audited'
    assert pin(admitted['source']['path']) == admitted['source']
    for terminal in admitted['actual_outer_terminals']:
        assert pin(terminal['path']) == terminal
        assert read(terminal['path'])['exit_code'] == 0
    cache = BASE / 'cache'
    assert pin(cache / 'index.json') == admitted['cache_index'] and pin(cache / 'responses.pt') == admitted['responses']
    index = read(cache / 'index.json')
    n = index['cases']
    assert 16 <= n <= 64 and admitted['unique_training_questions'] == n
    assert Native._hidden is OriginalNative._hidden and Native._response is ARLastHiddenResponseAdapter._response
    assert Native.generated is ARLastHiddenResponseAdapter.generated and initialize is OriginalInitialize
    assert PG_VERSION == '0.9.5'
    critic = AudioResponseCritic(2048, 2052)
    assert tuple(critic.net[0].weight.shape) == (16, 4100) and type(critic.net[1]).__name__ == 'SiLU'
    assert tuple(critic.net[2].weight.shape) == (1, 16)
    # Root has space; NVMe's completed journals remain untouched.
    workspace = Path('/home/mikkel/lexica-evidence-dial-work/01a1c84b-5d17-7e32-80ef-34d8e4ee2e62')
    workspace.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(workspace).free > 55 * 1024**3
    train = ROOT / 'artifacts/behavior-dial-20261114-v1/data/train.jsonl'
    dev = ROOT / 'artifacts/behavior-dial-20261114-v1/data/dev.jsonl'
    assert pin(train) == admitted['source_train']
    original = ROOT / 'artifacts/first-answer-control-20261111-v1/runs/first-answer/FULL-000000.pt'
    parent = ROOT / 'artifacts/behavior-expanded-h1024-20261114-v2'
    qualification = parent / 'fullanswer-health-qualification-v3-first.json'
    assert read(qualification)['qualified']
    evaluator = parent / 'CPU-AuditKIT-evaluator-qualification-first.json'
    assert read(evaluator)['actual_AuditKIT_inferences'] == 4
    names = ['tools/prepare_behavior_promptfix_training_v4.py',
        'tools/behavior_promptfix_phase_worker_v4.py', 'tools/run_behavior_promptfix_training_v4.py',
        'tools/audit_behavior_promptfix_quarter_v4.py', 'tools/close_behavior_promptfix_training_v4.py',
        'tools/check_behavior_fullanswer_health_v2.py', 'src/evidence_dial/behavior_promptfix_binding_v4.py',
        'src/evidence_dial/behavior_promptfix_native_proof_v4.py',
        'src/evidence_dial/behavior_fullanswer_generate_v2.py', 'src/evidence_dial/behavior_fullanswer_health_v3.py',
        'src/evidence_dial/behavior_protocol_v1.py', 'src/evidence_dial/binding.py',
        'src/evidence_dial/first_answer_binding_v2.py', 'src/evidence_dial/evaluate.py',
        'src/evidence_dial/shared_gpu_resources_v1.py', 'src/evidence_dial/continuation_resources_v1.py',
        'src/evidence_dial/fresh_native_zero_proof_v2.py']
    sources = [pin(ROOT / n) for n in names]
    sources += [pin(importlib.import_module(n).__file__) for n in ['particle_sliders.pg09',
        'particle_sliders.pg09_audio', 'particle_sliders.lifecycle', 'particle_sliders.inference',
        'particle_sliders.routed_projection', 'particle_sliders.initialization']]
    controls = read(DIAG / 'plan-first.json')['controls_reused_without_new_calls']
    off = next(c for c in controls if c['name'] == 'off')
    prompt = DIAG / 'arms/target0-unrestricted'
    control_pins = [off['predictions'], off['summary'], pin(prompt / 'predictions.jsonl'), pin(prompt / 'summary.json')]
    inputs = [pin(train), pin(dev), pin(admission), pin(cache / 'index.json'), pin(cache / 'responses.pt'),
        pin(original), pin(qualification), pin(evaluator), pin(DIAG / 'independent-audit-first.json'),
        pin(BASE / 'declaration-first.json'), pin(BASE / 'runtime-source-derivatives-first.json')] + control_pins + admitted['actual_outer_terminals']
    declaration = read(BASE / 'declaration-first.json')
    prompt_target = declaration['selected_prompt_only_control']['target']
    assert not (BASE / 'training-declaration-first.json').exists()
    cohorts = []
    folders = [workspace / 'behavior-promptfix-h1024-GPU0-g00025-v4',
               workspace / 'behavior-promptfix-h1024-GPU1-g0000625-v4']
    for gpu, rate, folder in [(0, .00025, folders[0]), (1, .0000625, folders[1])]:
        assert not folder.exists()
        options = {'mode': 'first-answer', 'steps': 1024, 'batch': 8, 'g_lr': rate,
            'd_lr': .018, 'bank_lr': .03, 'ema_decay': None, 'base_precision': 'bfloat16', 'seed': SEED,
            'binding_sha256': pin(ROOT / 'src/evidence_dial/behavior_promptfix_binding_v4.py')['sha256'],
            'training_data': str(train), 'training_data_sha256': pin(train)['sha256'],
            'cache': str(cache), 'cache_index_sha256': pin(cache / 'index.json')['sha256'],
            'cache_admission': str(admission), 'cache_admission_sha256': pin(admission)['sha256'],
            'initialization_receipt': str(folder / 'run/initialization.json'), 'cohort_id': folder.name}
        validate(options)
        SessionConfig(NAME, folder.name, **options)
        plan = {'schema': 'behavior-promptfix-original-PG1024-batch8-fullanswers-v4', 'time': time.time(),
            'gpu': gpu, 'physical_gpu_uuid': GPUS[gpu], 'cohort': folder.name, 'session_options': options,
            'original_FULL0': pin(original), 'development': pin(dev), 'checkpoints': [256, 512, 768, 1024],
            'development_strengths': [.25, .5, 1.], 'ema_decay': None, 'elapsed_work_deadline': None,
            'training_rows': n, 'unique_training_questions': n, 'training_weighting': index['training_weighting'],
            'target_protocol': PROTOCOL, 'output_cap': 4096, 'effect_scoring_requires_native_EOS': True,
            'selection_metric': 'equal-case mean full finished Jev effect ON minus matched OFF only; report all maximum ties',
            'diagnostics_not_selectors': ['sympathy', 'sycophancy'],
            'catastrophic_health': 'Unchanged qualified full-text observer v3; four fixed DEV probes per quarter before continuation',
            'holdout_access': False, 'controls_GPU0_first': False, 'prompt_only_target': prompt_target,
            'completed_controls_reused': {'off': off['folder'], 'prompt-only': str(prompt)},
            'reuse_native_proof': None if gpu == 0 else str(folders[0] / 'run'),
            'initialization': 'Fresh original scoped initializer at0; preserve pretrained and zero-output LoRA; restores never initialize',
            'sources': sources, 'inputs': inputs}
        write_json(folder / 'plan.json', plan)
        cohorts.append({'gpu': gpu, 'g_lr': rate, 'plan': pin(folder / 'plan.json'), 'folder': str(folder)})
    write_json(BASE / 'training-declaration-first.json', {'time': time.time(), 'cohorts': cohorts,
        'declared_before_first_update': True, 'source': pin(__file__), 'training_pairs': n,
        'fresh_steps_each': [0, 1024], 'checkpoints': [256, 512, 768, 1024], 'strengths': [.25, .5, 1.],
        'both_GPUs': True, 'all24_DEV_cells_required': True, 'formula_changes': False,
        'no_weight_EMA': True, 'new_test_data_still_sealed': True,
        'storage': 'New root-filesystem artifact folders; all existing NVMe checkpoints retained',
        'native_zero_proof_pending': True})
    print(json.dumps({'cohorts': cohorts, 'training_pairs': n, 'updates': 0}), flush=True)


if __name__ == '__main__':
    main()
