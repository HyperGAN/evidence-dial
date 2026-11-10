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


def build(out):
    cache=ROOT/'artifacts/cache/teacher-v1/index.json'
    targets=json.loads(cache.read_text())['rows']
    result={'diagnostic_only':True,'used_for_selection':False,'cache_index_sha256':file_sha(cache),
        'training_cases':64,'response_targets':len(targets),'arms':{},
        'sampling_note':'Particle sampling is over146 matched response positions, so case exposure is not uniform. LoRA uses each of the64 cases exactly4 times. Finished-output evaluation weights cases equally.'}
    series={}
    for name,expected in [('particle-G012-v1',32),('particle-G004-v1',128),('particle-G001-v1',128)]:
        source=ROOT/'artifacts/runs'/name/'updates.jsonl';rows=read_jsonl(source)
        assert [r['step'] for r in rows]==list(range(1,expected+1)),name
        entry={'actual_updates':len(rows),'journal_sha256':file_sha(source),'weight_EMA':None}
        for side in ['g','d']:
            indexes=[i for r in rows for i in r[side+'_rows']]
            count=Counter(targets[i]['id'] for i in indexes)
            entry[side+'_response_exposures']=len(indexes)
            entry[side+'_unique_positions']=len(set(indexes))
            entry[side+'_case_exposures']=dict(sorted(count.items()))
        for key in ['g_loss','g_adv','prior_regularization','d_loss','d_adv','d_pen']:
            values=[r[key] for r in rows]
            entry[key]={'first':values[0],'last':values[-1],'mean':sum(values)/len(values)}
        result['arms'][name]=entry;series[name]=rows
    run=ROOT/'artifacts/runs/aligntune-v2'
    journal=read_jsonl(run/'updates.jsonl');assert [r['step'] for r in journal]==list(range(1,129))
    exposure=Counter(k for r in journal for k in r['case_ids'])
    assert len(exposure)==64 and set(exposure.values())=={4}
    training=json.loads((run/'training-result.json').read_text())
    result['arms']['aligntune-v2']={'actual_updates':128,'journal_sha256':file_sha(run/'updates.jsonl'),
        'case_exposures':dict(sorted(exposure.items())),'weight_EMA':None,'metrics':training['metrics']}
    out.mkdir(parents=True,exist_ok=True);write_json(out/'training-diagnostics.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True,constrained_layout=True)
    for ax,(name,rows) in zip(axes,series.items()):
        ax.plot([r['step'] for r in rows],[r['g_loss'] for r in rows],label='Generator',lw=1,color='#1e6b4c')
        ax.plot([r['step'] for r in rows],[r['d_loss'] for r in rows],label='Critic',lw=1,color='#b58612')
        ax.set(title=name.replace('particle-','').replace('-v1',''),xlabel='Actual committed update')
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('Training loss (diagnostic only)')
    fig.suptitle('Loss curves do not establish finished-answer improvement',fontsize=11)
    fig.savefig(out/'particle-training-losses.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
