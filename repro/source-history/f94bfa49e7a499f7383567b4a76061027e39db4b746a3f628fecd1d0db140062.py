"""Build the static explorer exclusively from a complete retained release."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl,write_json


def build(release,out):
    result=json.loads((release/'results.json').read_text())
    selection=result['selection']
    cases=read_jsonl(release/'paired-cases.jsonl')
    selected=selection['selected_ties']
    # This declared cohort has a single promotable rate. Never choose between
    # tied rates using test performance to simplify a demo.
    rates={r['g_rate'] for r in selected}
    if len(rates)!=1:raise ValueError('Multiple tied rates require separate named demo cohorts')
    expected={'off','particle_1','particle_2','lora'}
    for name,dataset in result['datasets'].items():
        if not expected.issubset(dataset['arms']):raise ValueError('Every nominal demo arm must be measured')
        if any(not a['effect']['complete'] for a in dataset['arms'].values()):
            raise ValueError('An incomplete benchmark is not a complete demo')
    strengths=sorted({1,2,*(r['strength'] for r in selected)})
    public=[]
    for c in cases:
        row=dict(c)
        if c['dataset'].startswith('faith'):
            row['dataset']='faith'
            row['question']='Source question '+c.get('source_id',c['id'])
            row['context']='The original passage is not redistributed. Read the pinned source: '+c.get('source_uri','')
            row['gold']='See the original source and frozen scorer.'
            row['source_url']='https://huggingface.co/datasets/'+c['source_uri'].removeprefix('hf://datasets/').split('@')[0]+'/tree/'+c['source_revision']
        else:row['dataset']='policy'
        public.append(row)
    cohort=[]
    for rate in [.012,.004,.001]:
        measured=[r for r in selection['all_measured_development_results'] if r['g_rate']==rate]
        failed=next((r for r in selection['failed_cohorts'] if r['g_rate']==rate),None)
        stopped=next((r for r in selection.get('unmeasured_guard_cohorts',[]) if r['g_rate']==rate),None)
        cohort.append({'g_rate':rate,'completed_steps':32 if failed else 128,
            'gain_1':next((r['effect_gain'] for r in measured if r['strength']==1),None),
            'gain_2':next((r['effect_gain'] for r in measured if r['strength']==2),None),
            'status':'Stopped: qualified repetitive collapse' if failed else
                     'Blocked: mathematical-notation guard scope uncertain' if stopped else
                     'RAW128 eligible; development-selected rate'})
    policy=result['datasets']['policy'];faith=result['datasets']['faith']
    def fmt(x):return f'{100*x:+.1f} pp'
    # Selected ties remain listed; a display statistic never breaks them.
    summary='; '.join(f'G{r["g_rate"]:g} / strength {r["strength"]:g}' for r in selected)
    write_json(out,{'status':'Complete retained benchmark. Development selection: '+summary+'. Every regression is retained.',
        'strengths':strengths,'stats':[
            {'label':'Primary cases','value':str(policy['n']),'detail':'64 held-out families; four case types'},
            {'label':'FaithEval transfer','value':str(faith['n']),'detail':'Frozen subset; no verified matched controls'},
            {'label':'Particle strength 1 gain','value':fmt(policy['arms']['particle_1']['effect']['effect_gain']),'detail':'Primary holdout vs identical-prompt OFF'},
            {'label':'AlignTune LoRA gain','value':fmt(policy['arms']['lora']['effect']['effect_gain']),'detail':'Same training cases and matched OFF'}],
        'rates':cohort,'cases':public,
        'scope':'English fictional policies with held-out families and shared templates; external FaithEval subset. FaithEval passages stay at their original pinned sources. Strict answer-format gains count under the frozen protocol. Guard coverage is limited; G0.004 remains blocked for uncertain mathematical notation. Wall times include shared-GPU contention.',
        'links':[
            {'label':'Code and article','url':'https://github.com/HyperGAN/evidence-dial'},
            {'label':'Model collection','url':'https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f'},
            {'label':'Datasets and benchmark','url':'https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87'}]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();build(args.release,args.out)
