"""Plot frozen paired effects and clustered intervals from retained outputs."""
from pathlib import Path
import argparse
import json


def plot(release,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((release/'results.json').read_text())
    preferred=['particle_1','particle_2','lora','hinted_particle_1','hinted_particle_2','lora_original']
    arms=[a for a in preferred if any(a in d['arms'] for d in report['datasets'].values())]
    corrected='previous_recipe_selection' in report['selection']
    datasets=list(report['datasets'])
    ncols=min(2,len(datasets));nrows=(len(datasets)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(6*ncols,max(3.9,.55*len(arms)+1.8)*nrows),sharey=True,constrained_layout=True,squeeze=False)
    axes=axes.ravel()
    names={'particle_1':'Particle · curated ×1' if corrected else 'Particle ×1',
           'particle_2':'Particle · curated ×2' if corrected else 'Particle ×2',
           'lora':'LoRA · curated cases' if corrected else 'AlignTune LoRA',
           'hinted_particle_1':'Particle · hinted ×1','hinted_particle_2':'Particle · hinted ×2',
           'lora_original':'LoRA · original64'}
    labels=[names[a] for a in arms]
    titles={'policy':'Original policies · repeated' if corrected else 'Original primary policies',
            'faith':'FaithEval transfer','cue_audit':'Cue intervention audit','balanced':'Balanced grounding stress',
            'confirmation':'Fresh balanced confirmation'}
    for ax,dataset in zip(axes,datasets):
        data=report['datasets'][dataset]
        for i,arm in enumerate(arms):
            if arm not in data['arms']:
                ax.text(0,i,'Not planned',fontsize=8,color='#66706c',va='center')
                continue
            effect=data['arms'][arm]['effect'];assert effect['complete']
            lo,hi=[100*x for x in effect['paired_family_bootstrap_95ci']]
            value=100*effect['effect_gain'];color='#1e6b4c' if 'lora' in arm else '#a27d12' if arm.startswith('particle_') else '#66706c'
            ax.plot([lo,hi],[i,i],color=color,lw=2)
            ax.scatter([value],[i],color=color,s=28,zorder=3)
        ax.axvline(0,color='#66706c',ls='--',lw=1)
        ax.set(title=f'{titles.get(dataset,dataset)} · n={data["n"]}',
               xlabel='ON − OFF strict success (percentage points)',yticks=range(len(arms)),yticklabels=labels)
        ax.grid(axis='x',alpha=.2);ax.set_ylim(len(arms)-.5,-.5)
    for ax in axes[len(datasets):]:ax.set_visible(False)
    fig.suptitle('Finished-answer effects with paired cluster-bootstrap 95% intervals',fontsize=11)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180)
    fig.savefig(out.with_suffix('.svg'))
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();plot(args.release,args.out)
