"""Write reviewable HF cards from a completed bundle and its audited results."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from confirm_behavior_promptfix_v4 import BASE, OUT, read, pin, validate
PREFIX = 'promptfix37-h1024-v4'


def main():
    import yaml
    from evidence_dial.io import write_json
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--previous-dataset-card', type=Path, required=True)
    args = ap.parse_args()
    audit_path = OUT / 'independent-completion-audit-first.json'
    audit = read(audit_path)
    assert audit['state'] == 'all-locked-fresh64-full-EOS-answers-and-first-judgments-independently-audited'
    assert audit['no_weight_EMA'] and not audit['new_test_reselection']
    plan = validate(read(audit['plan']['path']))
    manifest = read(args.bundle / 'upload-manifest-first.json')
    assert manifest['fresh_confirmation_audit'] == pin(audit_path)
    for item in manifest['files']:
        path = args.bundle / item['path']
        assert path.stat().st_size == item['bytes'] and pin(path)['sha256'] == item['sha256']
    summaries = {a['name']: a for a in audit['arms']}
    selected = next(a for a in plan['arms'] if a['kind'] == 'particle'
        and a['candidate']['artifact'] == plan['display_checkpoint']
        and a['candidate']['strength'] == plan['display_strength'])
    assert selected['selected']
    candidate = selected['candidate']
    change = next(c for c in audit['comparisons'] if c['arm'] == selected['name'] and c['baseline'] == 'off')
    ci = change['individual_paired_bootstrap95']
    table = ['| Condition, 64 fresh questions | Mean behavior score | Change from OFF |',
             '| --- | ---: | ---: |']
    for arm in plan['arms']:
        summary = summaries[arm['name']]
        if arm['name'] == 'off':
            title = 'OFF'
        elif arm['name'] == 'prompt-only':
            title = 'Unrestricted prompt-only'
        elif arm.get('reference', False):
            title = 'Earlier DEV-selected adapter reference'
        else:
            title = f'Repaired PG, step {arm["candidate"]["step"]}, strength {arm["candidate"]["strength"]:g}'
            title += ' — DEV-selected' if arm['selected'] else ' — descriptive'
        table.append(f'| {title} | {summary["mean_effect"]:.6f} | {summary["mean_effect"] - summaries["off"]["mean_effect"]:+.6f} |')
    results = '\n'.join(table)
    evidence = (f'The declared display checkpoint changes the mean from OFF by {change["mean_difference"]:+.6f}; '
        f'its individual paired-question 95% bootstrap interval is [{ci[0]:+.6f}, {ci[1]:+.6f}]. '
        'All development maxima and all predeclared conditions are reported. Display strengths and the earlier reference do not select a new winner on these questions.')
    method = '''The target is sympathetic, constructive support with independent judgment.
For each of 64 TRAIN questions, the raw base produced one complete OFF response
and one response for each of three unrestricted behavioral instructions.
NanoGPT Jev accepted a target only when its fixed behavior score was at least
3/4 and exceeded matched OFF. One highest-scoring context per question yielded
37 paired first-answer hidden states. Neutral and positive inputs keep the
same question and empty continuation history; only positive has a behavioral
instruction. The previous unrelated sentence-limit instruction is removed.

Two fresh cohorts completed 1,024 updates with batch8, G0.00025 or0.0000625,
D0.018 and bank0.03. The ParticleGAN0.9.5 formulation and original scoped
initialization are unchanged. There is no SFT warm start, token-imitation loss
or weight EMA. Selection uses only equal-question finished effect ON minus OFF
across all24 DEV cells: four checkpoints and three strengths per cohort.
'''
    limits = '''- The behavior score is a fixed Jev judgment on a0–1 scale, not accuracy or a human preference percentage.
- Synthetic English questions share broad scenario families; this is not an external benchmark or a claim of unseen topics.
- One training seed and the same curation/evaluation judge limit the conclusions. No human preference study was performed.
- Supervision covers the first-answer state. The separate111-context continuation derivative was not used by these models.
- Stronger steering can help or hurt. The latest training checkpoint is not automatically the DEV-selected checkpoint.
- Individual bootstrap intervals are descriptive, not simultaneous coverage guarantees.
- Fresh-machine replication has not been verified. The native format requires the pinned project and Particle Sliders runtime, rather than standard PEFT loading.
'''
    model_metadata = '''---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
base_model_relation: adapter
library_name: particle-sliders-core
pipeline_tag: text-generation
inference: false
tags: [particlegan, preference-alignment, internal-states, cohere]
---
'''
    model_card = model_metadata + f'''
# Evidence Dial — repaired target setup

Root adapters contain the declared DEV-selected display checkpoint: **G{candidate['g_lr']:g},
step{candidate['step']}, runtime strength{candidate['strength']:g}**, from the new37-context study.
Both fresh cohorts completed1,024 updates. All tied DEV maxima are retained under
`{PREFIX}/checkpoints/`; the display rule does not assert superiority among ties.
These are custom native Particle Sliders artifacts, not standard PEFT weights.

## Fresh complete-answer comparison

{results}

{evidence}

## Training

{method}

## Inference

Base: `CohereLabs/tiny-aya-global@af89d219b53ed9b13b8a4645f9c8028973510324`;
core0.14.2/PG0.9.5; raw weights; BF16/SDPA GPU backbone. Load the artifact with
`particle_sliders.inference.open_inference` and
`evidence_dial.behavior_promptfix_binding_v4.inference_backend()` from the
[project source](https://github.com/HyperGAN/evidence-dial).

Use the full-answer path in `behavior_fullanswer_generate_v2.py`: obtain the
context's `binding.response.__self__`, use its `model` under
`slider.scaled(strength)`, and retain the ordinary user question. Generation
is greedy, with a4,096-token safety cap and required native EOS. The older QA
convenience response helper retains its own shorter answer budget. The
[recorded demo](https://huggingface.co/spaces/ntc-ai/evidence-dial) displays
complete saved answers without new inference or paid API calls.

Native checkpoint SHA-256: `{candidate['artifact']['sha256']}`.

## Limitations

{limits}

## Data and earlier studies

[Audited data and raw results](https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior/tree/main/{PREFIX}).
All earlier checkpoint folders remain preserved. The earlier138-context study
and its negative32-question confirmation remain historical evidence.
The separate QA study tied SFT at174/192 on DEV; held-out256-question results
favored SFT. This behavior study does not establish SFT parity.
'''
    previous = args.previous_dataset_card.read_text()
    assert previous.startswith('---\n')
    metadata = copy.deepcopy(yaml.safe_load(previous.split('---', 2)[1]))
    assert metadata['license'] == 'cc-by-nc-4.0'
    new_configs = [
        {'config_name': 'promptfix37_questions', 'data_files': [
            {'split': split, 'path': f'{PREFIX}/questions/{name}.jsonl'}
            for split, name in [('train', 'train'), ('validation', 'dev'), ('test', 'confirmation64')]]},
        {'config_name': 'promptfix37_confirmation', 'data_files': [
            {'split': 'test', 'path': f'{PREFIX}/evaluation/confirmation/combined.jsonl'}]},
    ]
    assert not {c['config_name'] for c in new_configs} & {c['config_name'] for c in metadata['configs']}
    metadata['configs'] += new_configs
    dataset_card = '---\n' + yaml.safe_dump(metadata, sort_keys=False) + '---\n\n' + f'''# Evidence Dial behavior data

The additive `{PREFIX}/` directory contains the completed repaired-target study.
All earlier datasets, checkpoints, answers and original configurations remain.

## Latest comparison

{results}

{evidence}

## Construction

{method}

All256 new curation answers end at native EOS. Training has37 admitted unique
questions, with coverage8 discouragement/12 unsupported conclusion/8 reasonable
boundary/9 uncertainty. Twenty-six overlap the earlier37-question cache;11
newly qualify and11 previously accepted questions are excluded. There is one
context per admitted question, rather than the earlier138 contexts for37
questions. This changes both target construction and question weighting.

## Files

- `questions/`:64 TRAIN,16 DEV and64 fresh confirmation questions. The latter were frozen before new PG training; broad scenario families overlap.
- `training-pairs/`: the admitted37 raw neutral/positive first-state pairs and provenance index.
- `mining-states/`: all256 captured frozen prefill states, including excluded candidates.
- `training/`: both1,024-update journals and FULL/RAW/NATIVE packets at0/256/512/768/1,024. All per-update FULL packets are retained locally; public milestones are explicitly identified.
- `evaluation/`:all24 DEV conditions, both controls, and every locked fresh64 condition, including the fixed earlier adapter reference.
- `audits/`: curation, prompt ablation, checkpoint and final comparison admissions.
- `provenance/observations-and-sources.tar.gz`: native captures, Jev firsts, execution records and source bodies with a per-file hash manifest. Private coordination and credentials are excluded.

## Limitations

{limits}

Training questions never include the DEV or confirmation messages. Curated
TRAIN scores are selection-biased and are not evaluation results. Complete
answers reached EOS; a safety-cap cutoff would remain unmeasured. The sole
selection metric is finished intended effect ON minus matched OFF; sympathy
and sycophancy probabilities are separate diagnostics. A qualified
catastrophic-output check is kept separate from effect ranking.

[Project source](https://github.com/HyperGAN/evidence-dial) ·
[Recorded complete-answer demo](https://huggingface.co/spaces/ntc-ai/evidence-dial)
'''
    for folder, text in [('model', model_card), ('dataset', dataset_card)]:
        path = args.bundle / folder / 'README.md'
        assert not path.exists()
        path.write_text(text)
    files = [{'path': str(p.relative_to(args.bundle)), 'bytes': p.stat().st_size, 'sha256': pin(p)['sha256']}
        for p in sorted(args.bundle.rglob('*')) if p.is_file() and p.name != 'upload-manifest-first.json']
    write_json(args.bundle / 'cards-review-first.json', {'time': time.time(), 'source': pin(__file__),
        'parent_dataset_card': pin(args.previous_dataset_card), 'final_audit': pin(audit_path),
        'files': files, 'cards_created': True, 'public_upload_performed': False,
        'all_previous_dataset_configs_preserved': True, 'new_model_or_Jev_calls': 0})
    print(json.dumps({'review': pin(args.bundle / 'cards-review-first.json'), 'files': len(files), 'published': False}))


if __name__ == '__main__':
    main()
