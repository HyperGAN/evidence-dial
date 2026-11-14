"""Export all32 audited full-answer confirmation cases for the existing gallery."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout-audit', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    from evidence_dial.behavior_fullanswer_confirmation_v2 import pin, verified, read, validate_plan
    from evidence_dial.behavior_protocol_v1 import TARGETS
    from evidence_dial.io import read_jsonl, write_json
    audit = read(args.holdout_audit)
    assert audit['state'] == 'all-locked-behavior-holdout-arms-independently-admitted'
    assert audit['cases_each'] == 32 and not audit['holdout_based_reselection']
    plan = validate_plan(read(verified(audit['plan'])))
    summaries = {arm['name']: arm for arm in audit['arms']}
    records = {name: read_jsonl(verified(summary['predictions'])) for name, summary in summaries.items()}
    display = sorted([arm for arm in plan['arms'] if arm['kind'] == 'particle'
        and arm['candidate']['artifact'] == plan['display_checkpoint']], key=lambda arm: arm['candidate']['strength'])
    assert [arm['candidate']['strength'] for arm in display] == [.5, 1., 2.]
    model_id = 'expanded138-development-selected-checkpoint'
    def record(row):
        assert row['native']['output_ids'][-1] == 3 and row['termination'] == 'native_eos'
        return {'output': row['output'], 'effect': row['effect'], 'sympathy': row['scores']['sympathy'],
            'sycophancy': row['scores']['sycophancy'], 'termination': 'native_eos',
            'output_tokens': len(row['native']['output_ids'])}
    cases = []
    for index, (off, prompt) in enumerate(zip(records['off'], records['prompt-only'])):
        assert off['id'] == prompt['id'] and off['question'] == prompt['question']
        grid = {}
        for arm in display:
            row = records[arm['name']][index]
            assert row['id'] == off['id'] and row['question'] == off['question']
            assert row['native']['input_ids'] == off['native']['input_ids']
            grid[f'{arm["candidate"]["strength"]:g}'] = record(row)
        cases.append({'id': off['id'], 'case_type': off['case_type'], 'question': off['question'],
            'off': record(off), 'prompt_only': record(prompt), 'models': {model_id: grid}})
    assert len(cases) == 32
    metrics = []
    for arm in display:
        summary = summaries[arm['name']]
        comparison = next(c for c in audit['comparisons'] if c['arm'] == arm['name'] and c['baseline'] == 'off')
        metrics.append({'strength': arm['candidate']['strength'],
            **{key: summary[key] for key in ['mean_effect', 'mean_sympathy_probability', 'mean_sycophancy_probability']},
            'paired_bootstrap95': comparison['paired_bootstrap95']})
    def control(name):
        return {key: summaries[name][key] for key in ['mean_effect', 'mean_sympathy_probability', 'mean_sycophancy_probability']}
    chosen = next(arm for arm in display if arm['name'] in plan['selected_development_maxima'])['candidate']
    value = {'schema': 'particle-behavior-explorer-v1', 'exported_at': time.time(), 'split': 'holdout',
        'status': 'Fresh32-question complete-answer confirmation. Both138-context training runs finished1024 updates; checkpoint and strength selection used development data only.',
        'scope_note': 'All32 fresh questions and all three fixed display strengths are shown. All tied development maxima are included in the downloadable comparison. These synthetic questions were frozen before reading any expanded-run trained development results; broader semantic families overlap.',
        'uncertainty_note': 'Individual matched-question bootstrap intervals are descriptive. Confirmation scores do not choose a new checkpoint or strength.',
        'prompt_instruction': TARGETS[0], 'cases': cases,
        'models': [{'id': model_id, 'title': 'Expanded-data checkpoint · complete answers',
            'description': f'{chosen["step"]}updates; G{chosen["g_lr"]:g};138context pairs/37TRAINquestions; raw weights; chosen on DEV',
            'sha256': plan['display_checkpoint']['sha256'], 'metrics': metrics}],
        'controls': {'off': control('off'), 'prompt_only': control('prompt-only')},
        'secondary_qa': 'Separate earlier QA study: PG and SFT each174/192 on DEV; on sealed256-question test, five preselected PG arms221–229, four SFT controls230–234, OFF142. A DEV tie and held-out gap, not SFT parity.',
        'provenance': {'export_source_sha256': pin(__file__)['sha256'], 'audit_sha256s': [pin(args.holdout_audit)['sha256']],
            'no_generated_or_interpolated_demo_text': True, 'all_cases_in_declared_split_included': True,
            'all_answers_native_EOS': True, 'maximum_new_tokens_safetycap': 4096},
        'all_holdout_results': {'arms': [{key: value for key, value in arm.items() if key not in ['predictions', 'summary']}
            for arm in audit['arms']], 'comparisons': audit['comparisons']}}
    assert '/ml2/' not in json.dumps(value) and '/mnt/nvme/' not in json.dumps(value)
    write_json(args.output, value)
    print(json.dumps({'export': pin(args.output), 'cases': 32, 'complete_displayed_responses': 160,
        'all_preselected_ties_preserved': True, 'new_model_or_Jev_calls': 0}))


if __name__ == '__main__':
    main()
