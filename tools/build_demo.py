"""Build the static explorer exclusively from a complete retained release."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl,write_json


def build(release,out):
    result=json.loads((release/'results.json').read_text());selection=result['selection']
    cases=read_jsonl(release/'paired-cases.jsonl');selected=selection['selected_ties']
    if len({r['g_rate'] for r in selected})!=1:
        raise ValueError('Multiple tied rates require separate named demo cohorts')
    corrected='previous_recipe_selection' in selection
    experiments=({'curated':{'label':'Jev-curated successes','particle_prefix':'particle_','lora_key':'lora'},
                  'hinted':{'label':'Original answer-hinted targets','particle_prefix':'hinted_particle_','lora_key':'lora_original'}}
                 if corrected else {'main':{'label':'Original answer-hinted targets','particle_prefix':'particle_','lora_key':'lora'}})
    for variant,config in experiments.items():
        chosen=selection['previous_recipe_selection'] if variant=='hinted' else selection
        config['selected_strengths']=[r['strength'] for r in chosen['selected_ties']]
    experiment_strengths={};dataset_strengths={}
    for variant,config in experiments.items():
        experiment_strengths[variant]={}
        for name,dataset in result['datasets'].items():
            expected={'off',config['particle_prefix']+'1',config['lora_key']}
            if not expected.issubset(dataset['arms']):raise ValueError('Every planned comparison arm must be measured')
            if any(not a['effect']['complete'] for a in dataset['arms'].values()):
                raise ValueError('An incomplete benchmark is not a complete demo')
            strengths=sorted(int(a.removeprefix(config['particle_prefix'])) for a in dataset['arms']
                             if a.startswith(config['particle_prefix']))
            experiment_strengths[variant][name]=strengths
            if variant==next(iter(experiments)):dataset_strengths[name]=strengths
    strengths=sorted({s for datasets in experiment_strengths.values() for ss in datasets.values() for s in ss})
    public=[]
    for c in cases:
        row=dict(c);row['dataset']=c['benchmark']
        if c['benchmark']=='faith':
            row['question']='Source question '+c.get('source_id',c['id'])
            row['context']='The original passage is not redistributed. Read the pinned source: '+c.get('source_uri','')
            row['gold']='See the original source and frozen scorer.'
            row['source_url']='https://huggingface.co/datasets/'+c['source_uri'].removeprefix('hf://datasets/').split('@')[0]+'/tree/'+c['source_revision']
        public.append(row)
    previous=selection.get('previous_recipe_selection',selection);cohort=[]
    for rate in [.012,.004,.001]:
        measured=[r for r in previous['all_measured_development_results'] if r['g_rate']==rate]
        failed=next((r for r in previous['failed_cohorts'] if r['g_rate']==rate),None)
        stopped=next((r for r in previous.get('unmeasured_guard_cohorts',[]) if r['g_rate']==rate),None)
        cohort.append({'g_rate':f'Answer-hinted · {rate:g}','completed_steps':32 if failed else 128,
            'gain_1':next((r['effect_gain'] for r in measured if r['strength']==1),None),
            'gain_2':next((r['effect_gain'] for r in measured if r['strength']==2),None),
            'status':'Stopped: qualified repetitive collapse' if failed else
                     'Blocked: mathematical-notation guard scope uncertain' if stopped else
                     'RAW128 eligible; selected within original recipe'})
    if corrected:
        measured=selection['all_measured_development_results']
        cohort.append({'g_rate':'Jev-curated · 0.001','completed_steps':128,
            'gain_1':next(r['effect_gain'] for r in measured if r['strength']==1),
            'gain_2':next(r['effect_gain'] for r in measured if r['strength']==2),
            'status':'Separate fixed-rate retry; dev-selected strength(s) '+', '.join(str(r['strength']) for r in selected)})
    labels={'policy':'Original policy holdout','faith':'FaithEval transfer subset','cue_audit':'Cue intervention audit',
            'balanced':'Balanced grounding stress test','confirmation':'Fresh balanced confirmation'}
    aggregates=[]
    for name,dataset in result['datasets'].items():
        for variant,config in experiments.items():
            comparisons=[]
            for s in config['selected_strengths']:
                comparison=dataset['comparisons_vs_'+config['lora_key']][config['particle_prefix']+str(s)]['paired_success_difference']
                comparisons.append({'strength':s,'gain':comparison['effect_gain'],
                    'ci95':comparison['paired_family_bootstrap_95ci']})
            mapped={'off':dataset['arms']['off']['success_rate'],'lora':dataset['arms'][config['lora_key']]['success_rate']}
            mapped.update({'particle_'+str(s):dataset['arms'][config['particle_prefix']+str(s)]['success_rate']
                           for s in experiment_strengths[variant][name]})
            aggregates.append({'dataset':labels.get(name,name)+' · '+config['label'],'cases':dataset['n'],
                'success_rates':mapped,'selected_comparisons_vs_lora':comparisons})
    def fmt(x):return f'{100*x:+.1f} pp'
    primary=result['datasets']['confirmation' if corrected else 'policy']
    comparisons=[primary['comparisons_vs_lora']['particle_'+str(r['strength'])]['paired_success_difference'] for r in selected]
    stats=[{'label':'Fresh confirmation' if corrected else 'Original primary cases','value':str(primary['n']),
            'detail':'32 families; 384 order pairs' if corrected else '64 held-out families; shared question frames'},
           {'label':'Dev-selected strength(s)','value':', '.join(str(r['strength']) for r in selected),
            'detail':'Frozen before holdout; exact ties retained'},
           {'label':'Selected particle − matched LoRA','value':' / '.join(fmt(c['effect_gain']) for c in comparisons),
            'detail':'All selected ties, in strength order; paired intervals below'},
           {'label':'Matched LoRA strict success','value':f"{100*primary['arms']['lora']['success_rate']:.1f}%",
            'detail':'Same curated training-case pool' if corrected else 'Same64 source cases'}]
    summary='; '.join(f'G{r["g_rate"]:g} / strength {r["strength"]:g}' for r in selected)
    write_json(out,{'status':('Complete corrected and original retained benchmarks. ' if corrected else 'Complete original answer-hinted benchmark; corrected Jev-curated retry is pending. ')+'Development selection: '+summary+'. Every regression is retained.',
        'strengths':strengths,'selected_strengths':[r['strength'] for r in selected],
        'dataset_strengths':dataset_strengths,'dataset_labels':labels,'experiments':experiments,
        'experiment_strengths':experiment_strengths,'aggregates':aggregates,'stats':stats,'rates':cohort,'cases':public,
        'scope':'The original policy test shares question frames and has a document-count shortcut. Balanced tests break that cue; each 768-case cohort has32 families and384 order pairs. The corrected retry uses Jev-approved Tiny Aya successes and exact paired hidden states, with a same-case-pool LoRA control. These synthetic tests do not establish general grounding. FaithEval is a frozen external subset without verified matched answerable controls; its passages stay at pinned sources. Strict format gains count. G0.004 remains blocked for uncertain mathematical-notation guard coverage. Shared-GPU timing includes contention; one legacy diagnostic has an unverified-device exception documented in the article.',
        'links':[{'label':'Code and article','url':'https://github.com/HyperGAN/evidence-dial'},
                 {'label':'Model collection','url':'https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f'},
                 {'label':'Datasets and benchmark','url':'https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87'}]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();build(args.release,args.out)
