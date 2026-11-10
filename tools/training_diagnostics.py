"""Summarize committed exposures and losses; never select a candidate here."""
from pathlib import Path
from collections import Counter
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl,write_json
from evidence_dial.benchmark import file_sha


def particle_entry(name, cache, expected):
    targets=json.loads(cache.read_text())['rows']
    source=ROOT/'artifacts/runs'/name/'updates.jsonl';rows=read_jsonl(source)
    assert [r['step'] for r in rows]==list(range(1,expected+1)),name
    entry={'actual_updates':len(rows),'journal_sha256':file_sha(source),'weight_EMA':None,
           'cache_index_sha256':file_sha(cache),'response_targets':len(targets),
           'training_cases':len({r['id'] for r in targets})}
    for side in ['g','d']:
        indexes=[i for r in rows for i in r[side+'_rows']]
        assert len(indexes)==2*expected
        count=Counter(targets[i]['id'] for i in indexes)
        entry[side+'_response_exposures']=len(indexes)
        entry[side+'_unique_positions']=len(set(indexes))
        entry[side+'_case_exposures']=dict(sorted(count.items()))
    for key in ['g_loss','g_adv','prior_regularization','d_loss','d_adv','d_pen']:
        values=[r[key] for r in rows]
        entry[key]={'first':values[0],'last':values[-1],'mean':sum(values)/len(values)}
    return entry,rows


def lora_entry(name, cases, expected):
    run=ROOT/'artifacts/runs'/name
    journal=read_jsonl(run/'updates.jsonl')
    assert [r['step'] for r in journal]==list(range(1,expected+1))
    assert all(len(r['case_ids'])==2 for r in journal)
    exposure=Counter(k for r in journal for k in r['case_ids'])
    assert len(exposure)==cases and sum(exposure.values())==2*expected
    training=json.loads((run/'training-result.json').read_text())
    return {'actual_updates':expected,'journal_sha256':file_sha(run/'updates.jsonl'),
        'training_cases':cases,'total_case_exposures':sum(exposure.values()),
        'exposures_per_case_distribution':dict(sorted(Counter(exposure.values()).items())),
        'case_exposures':dict(sorted(exposure.items())),'weight_EMA':None,'metrics':training['metrics']}


def build(out, include_curated=False):
    cache=ROOT/'artifacts/cache/teacher-v1/index.json'
    targets=json.loads(cache.read_text())['rows']
    result={'diagnostic_only':True,'used_for_selection':False,'cache_index_sha256':file_sha(cache),
        'training_cases':64,'response_targets':len(targets),'arms':{},
        'sampling_note':'Particle sampling is over146 matched response positions, so case exposure is not uniform. LoRA uses each of the64 cases exactly4 times. Finished-output evaluation weights cases equally.'}
    series={}
    specs=[('particle-G012-v1',cache,32),('particle-G004-v1',cache,128),('particle-G001-v1',cache,128)]
    if include_curated:
        specs.append(('particle-curated-G001-v1',ROOT/'artifacts/cache/curated-success-v1/index.json',128))
        result['curated_sampling_note']='The corrected particle samples118 paired positions from48 accepted cases; same-pool LoRA samples48 cases. Both commit128 updates with batch2, but their exposure units and objectives differ. Neither training loss nor hidden SSE selects the model.'
    for name,arm_cache,expected in specs:
        entry,rows=particle_entry(name,arm_cache,expected)
        result['arms'][name]=entry;series[name]=rows
    result['arms']['aligntune-v2']=lora_entry('aligntune-v2',64,128)
    assert set(result['arms']['aligntune-v2']['case_exposures'].values())=={4}
    if include_curated:
        result['arms']['aligntune-curated-v1']=lora_entry('aligntune-curated-v1',48,128)
    out.mkdir(parents=True,exist_ok=True);write_json(out/'training-diagnostics.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncols=2 if include_curated else 3;nrows=(len(series)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(4.2*ncols,3.6*nrows),sharey=True,constrained_layout=True,squeeze=False)
    for ax,(name,rows) in zip(axes.ravel(),series.items()):
        ax.plot([r['step'] for r in rows],[r['g_loss'] for r in rows],label='Generator',lw=1,color='#1e6b4c')
        ax.plot([r['step'] for r in rows],[r['d_loss'] for r in rows],label='Critic',lw=1,color='#b58612')
        ax.set(title=name.replace('particle-','').replace('-v1',''),xlabel='Actual committed update')
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    for ax in axes[:,0]:ax.set_ylabel('Training loss (diagnostic only)')
    fig.suptitle('Loss curves do not establish finished-answer improvement',fontsize=11)
    fig.savefig(out/'particle-training-losses.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--include-curated',action='store_true')
    a=p.parse_args();build(a.out,a.include_curated)
