"""Export only audited recorded texts; never fabricate a demo response.

The preview consumes two complete32-update DEV quarters. The final path
requires the complete independently audited locked holdout comparison.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true')
    ap.add_argument('--holdout-audit',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert args.preview != bool(args.holdout_audit), 'Choose exactly one audited data scope'
    assert not args.output.exists()
    from evidence_dial.behavior_confirmation_v1 import BASE,pin,verified,read
    from evidence_dial.behavior_protocol_v1 import TARGETS
    from evidence_dial.io import read_jsonl,write_json
    audit_pins=[];models=[];model_records={}
    if args.preview:
        split='dev';scope='Development preview — training and held-out evaluation are still in progress.'
        controls_audit=BASE / 'DEV-controls-GPU0-v1/independent-completion-audit-first.json';controls=read(controls_audit)
        assert controls['state']=='all32-native-and-NanoGPT-Jev-control-records-readmitted'
        control_metrics=controls['metrics'];audit_pins.append(pin(controls_audit))
        for gpu in [0,1]:
            p=BASE / f'GPU{gpu}-quarter32-independent-admission-first.json';a=read(p)
            assert a['state']=='declared-complete-behavior-quarter-independently-readmitted' and a['step']==32
            assert a['all48_native_inputs_outputs_typed_Jev_scores_verified'] and a['no_weight_EMA']
            audit_pins.append(pin(p));mid=f'dev-GPU{gpu}-step32';metrics=[];model_records[mid]={}
            for arm in a['arms']:
                verified(arm['artifact']);verified(arm['summary'])
                model_records[mid][str(float(arm['strength'])).rstrip('0').rstrip('.')]=read_jsonl(verified(arm['predictions']))
                metrics.append({k:arm[k] for k in ['strength','mean_effect','mean_sympathy_probability','mean_sycophancy_probability']})
            models.append({'id':mid,'title':f'Development run {"A" if gpu==0 else "B"} · 32 updates',
                'description':f'G learning rate {a["actual_optimizer_rates"]["generator"]:g};32updates; strength grid0.5/1/2',
                'sha256':a['NATIVE']['sha256'],'metrics':metrics})
        scope_note='These are the first declared development checkpoints, not a final selected model or held-out result.'
        uncertainty_note='Development previews are used during model selection and do not establish held-out improvement.'
    else:
        split='holdout';a=read(args.holdout_audit)
        assert a['state']=='all-locked-behavior-holdout-arms-independently-admitted' and not a['holdout_based_reselection']
        plan=read(verified(a['plan']));audit_pins.append(pin(args.holdout_audit))
        controls_by_name={v['name']:v for v in a['arms']};control_metrics={'off':controls_by_name['off'],'prompt-only':controls_by_name['prompt-only']}
        display=[v for v in plan['arms'] if v['kind']=='particle' and v['candidate']['artifact']==plan['display_checkpoint']]
        assert sorted(v['candidate']['strength'] for v in display)==[.5,1.,2.]
        mid='development-chosen-display-checkpoint';model_records[mid]={};metrics=[]
        for arm in sorted(display,key=lambda v:v['candidate']['strength']):
            summary=controls_by_name[arm['name']];strength=arm['candidate']['strength'];model_records[mid][f'{strength:g}']=read_jsonl(verified(summary['predictions']))
            comparison=next(c for c in a['comparisons'] if c['arm']==arm['name'] and c['baseline']=='off')
            metrics.append({'strength':strength,**{k:summary[k] for k in ['mean_effect','mean_sympathy_probability','mean_sycophancy_probability']},'paired_bootstrap95':comparison['paired_bootstrap95']})
        representative=display[0]['candidate'];models=[{'id':mid,'title':'Checkpoint chosen on development data',
            'description':f'{representative["step"]}updates; G learning rate {representative["g_lr"]:g}; display identity frozen before held-out inference',
            'sha256':plan['display_checkpoint']['sha256'],'metrics':metrics}]
        scope='Held-out recorded comparison — checkpoint and strength selection used development data only.'
        scope_note='All tied development maxima are retained in the complete downloadable evaluation; this gallery displays the checkpoint fixed by the prospective display rule.'
        uncertainty_note='Individual paired question-bootstrap intervals are descriptive; the held-out results do not select a new model.'
    off=read_jsonl(verified(control_metrics['off']['predictions']));prompt=read_jsonl(verified(control_metrics['prompt-only']['predictions']))
    assert len(off)==len(prompt)==(16 if args.preview else 32)
    def record(r):
        ids=r['native']['output_ids'];assert ids and (ids[-1]==3 or len(ids)==64)
        return {'output':r['output'],'effect':r['effect'],'sympathy':r['scores']['sympathy'],'sycophancy':r['scores']['sycophancy'],
            'termination':'native_eos' if ids[-1]==3 else 'max_new_tokens'}
    cases=[]
    for i,(o,p) in enumerate(zip(off,prompt)):
        assert o['id']==p['id'] and o['question']==p['question']
        values={}
        for mid,strengths in model_records.items():
            values[mid]={}
            for strength,rows in strengths.items():
                r=rows[i];assert r['id']==o['id'] and r['question']==o['question'] and r['native']['input_ids']==o['native']['input_ids']
                values[mid][strength]=record(r)
        cases.append({'id':o['id'],'case_type':o['case_type'],'question':o['question'],'off':record(o),'prompt_only':record(p),'models':values})
    def metrics(v):return {k:v[k] for k in ['mean_effect','mean_sympathy_probability','mean_sycophancy_probability']}
    value={'schema':'particle-behavior-explorer-v1','exported_at':time.time(),'split':split,'status':scope,'scope_note':scope_note,
        'uncertainty_note':uncertainty_note,'prompt_instruction':TARGETS[0],'cases':cases,'models':models,
        'controls':{'off':metrics(control_metrics['off']),'prompt_only':metrics(control_metrics['prompt-only'])},
        'secondary_qa':'Secondary QA experiment: PG and SFT each174/192 on development. On a sealed256-question test, the five preselected PG arms scored221–229, four SFT controls230–234, and OFF142. Development tie; held-out results favor SFT.',
        'provenance':{'export_source_sha256':pin(__file__)['sha256'],'audit_sha256s':[p['sha256'] for p in audit_pins],
            'no_generated_or_interpolated_demo_text':True,'all_cases_in_declared_split_included':True},
        'all_holdout_results':None if args.preview else {'arms':a['arms'],'comparisons':a['comparisons']}}
    # Public data contains synthetic questions/text/metrics; omit private paths.
    if value['all_holdout_results']:
        value['all_holdout_results']['arms']=[{k:v for k,v in arm.items() if k not in ['predictions','summary']} for arm in a['arms']]
    assert '/ml2/' not in json.dumps(value) and '/mnt/nvme/' not in json.dumps(value)
    write_json(args.output,value)
    print(json.dumps({'export':pin(args.output),'split':split,'cases':len(cases),'models':len(models),'new_model_or_Jev_calls':0}))


if __name__=='__main__':main()
