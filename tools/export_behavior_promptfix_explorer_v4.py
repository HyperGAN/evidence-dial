"""Export every audited fresh64 answer, including the fixed previous-setup reference."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from confirm_behavior_promptfix_v4 import OUT, read, pin, validate


def main():
    from evidence_dial.io import read_jsonl, write_json
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    audit_path = OUT / 'independent-completion-audit-first.json'
    audit = read(audit_path)
    assert audit['state'] == 'all-locked-fresh64-full-EOS-answers-and-first-judgments-independently-audited'
    assert audit['all_answers_native_EOS'] and not audit['new_test_reselection']
    plan = validate(read(audit['plan']['path']))
    assert pin(audit['plan']['path']) == audit['plan']
    summaries = {a['name']: a for a in audit['arms']}
    records = {}
    for name, summary in summaries.items():
        assert pin(summary['predictions']['path']) == summary['predictions']
        records[name] = read_jsonl(summary['predictions']['path'])
    models = []
    by_artifact = {}
    comparisons = {(c['arm'], c['baseline']): c for c in audit['comparisons']}
    for arm in [a for a in plan['arms'] if a['kind'] == 'particle']:
        candidate = arm['candidate']
        sha = candidate['artifact']['sha256']
        if sha not in by_artifact:
            identifier = 'raw-' + sha[:16]
            assert identifier not in {m['id'] for m in models}
            reference = arm.get('reference', False)
            title = ('Previous target setup' if reference else f'Repaired target setup · run {candidate["gpu"] + 1}') + f' · checkpoint {candidate["step"]}'
            model = {'id': identifier, 'sha256': sha, 'title': title, 'metrics': [],
                'default_strength': candidate['strength'],
                'description': f'{candidate["step"]} updates; G {candidate["g_lr"]:g}; '
                    + ('138 contexts / 37 TRAIN questions; fixed earlier DEV-selected reference' if reference
                       else '37 contexts / 37 TRAIN questions; new DEV-selected study'),
                'previous_setup_reference': reference}
            by_artifact[sha] = model
            models.append(model)
        model = by_artifact[sha]
        summary = summaries[arm['name']]
        comparison = comparisons[(arm['name'], 'off')]
        model['metrics'].append({'strength': candidate['strength'],
            'mean_effect': summary['mean_effect'], 'mean_sympathy_probability': summary['mean_sympathy_probability'],
            'mean_sycophancy_probability': summary['mean_sycophancy_probability'],
            'paired_bootstrap95': comparison['individual_paired_bootstrap95'],
            'selected_on_development': arm['selected'], 'source_arm': arm['name']})
    for model in models:
        model['metrics'].sort(key=lambda m: m['strength'])
        assert len({m['strength'] for m in model['metrics']}) == len(model['metrics'])
    default_model = by_artifact[plan['display_checkpoint']['sha256']]
    default_model['default_strength'] = plan['display_strength']
    # Keep the declared display checkpoint first, even if artifact insertion order differs.
    models.sort(key=lambda m: m['id'] != default_model['id'])
    def response(record):
        assert record['native']['output_ids'][-1] == 3 and record['termination'] == 'native_eos'
        return {'output': record['output'], 'effect': record['effect'],
            'sympathy': record['scores']['sympathy'], 'sycophancy': record['scores']['sycophancy'],
            'termination': 'native_eos', 'output_tokens': len(record['native']['output_ids'])}
    cases = []
    for index, off in enumerate(records['off']):
        prompt = records['prompt-only'][index]
        assert prompt['id'] == off['id'] and prompt['question'] == off['question']
        grid = {m['id']: {} for m in models}
        for arm in [a for a in plan['arms'] if a['kind'] == 'particle']:
            record = records[arm['name']][index]
            assert record['id'] == off['id'] and record['question'] == off['question']
            assert record['native']['input_ids'] == off['native']['input_ids']
            model = by_artifact[arm['candidate']['artifact']['sha256']]
            grid[model['id']][f'{arm["candidate"]["strength"]:g}'] = response(record)
        cases.append({'id': off['id'], 'case_type': off['case_type'], 'question': off['question'],
            'off': response(off), 'prompt_only': response(prompt), 'models': grid})
    assert len(cases) == 64
    def control(name):
        return {k: summaries[name][k] for k in ['mean_effect', 'mean_sympathy_probability', 'mean_sycophancy_probability']}
    selected_arm = next(a for a in plan['arms'] if a['kind'] == 'particle'
        and a['candidate']['artifact'] == plan['display_checkpoint']
        and a['candidate']['strength'] == plan['display_strength'])
    assert selected_arm['selected']
    selected_mean = summaries[selected_arm['name']]['mean_effect']
    value = {'schema': 'particle-behavior-explorer-v1', 'exported_at': time.time(), 'split': 'holdout',
        'status': f'Fresh 64-question complete-answer comparison. Base {summaries["off"]["mean_effect"]:.4f}; '
            f'DEV-selected repaired adapter {selected_mean:.4f}; prompt-only {summaries["prompt-only"]["mean_effect"]:.4f}. '
            'All checkpoint and strength choices were fixed before these answers were generated.',
        'scope_note': 'All 64 fresh questions, both controls, every new DEV maximum, fixed display strengths and the earlier DEV-selected adapter reference are included. Broad scenario families overlap. The new questions were frozen before new PG training.',
        'uncertainty_note': 'Individual paired-question bootstrap intervals are descriptive. Test results do not select another checkpoint or strength.',
        'prompt_instruction': next(a['target'] for a in plan['arms'] if a['kind'] == 'prompt-only'),
        'cases': cases, 'models': models, 'default_model_id': default_model['id'],
        'controls': {'off': control('off'), 'prompt_only': control('prompt-only')},
        'method_note': 'Three unrestricted behavioral instructions per TRAIN question; complete candidate answers must beat matched OFF and meet the frozen quality threshold. One best qualifying context per question.',
        'study_note': 'Both new 37-context cohorts ran from step0 through1024. Targets are first-answer hidden states; the backbone remains frozen. New preferences require new curation and training. Synthetic data, one training seed and the same curation/evaluation judge limit generalization.',
        'secondary_qa': 'Separate earlier QA study: PG and SFT each174/192 on DEV. On the sealed256-question test, five preselected PG arms scored221–229, four SFT controls230–234, and OFF142. A DEV tie and a held-out gap, not SFT parity.',
        'provenance': {'export_source_sha256': pin(__file__)['sha256'], 'audit_sha256s': [pin(audit_path)['sha256']],
            'no_generated_or_interpolated_demo_text': True, 'all_cases_in_declared_split_included': True,
            'all_answers_native_EOS': True, 'maximum_new_tokens_safetycap': 4096},
        'all_holdout_results': {'arms': [{k: v for k, v in a.items() if k not in ['predictions', 'summary']} for a in audit['arms']],
            'comparisons': audit['comparisons']}}
    rendered = json.dumps(value, ensure_ascii=False)
    assert all(prefix not in rendered for prefix in ['/ml2/', '/mnt/nvme/', '/home/mikkel/'])
    write_json(args.output, value)
    print(json.dumps({'export': pin(args.output), 'cases': 64, 'complete_saved_responses': audit['total_test_answers'],
        'models': len(models), 'new_model_or_Jev_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
