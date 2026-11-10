"""Plot frozen paired effects and clustered intervals from retained outputs."""
from pathlib import Path
import argparse
import json


def plot(release,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((release/'results.json').read_text())
    arms=[a for a in report['datasets']['policy']['arms'] if a!='off']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True,constrained_layout=True)
    labels=['AlignTune LoRA' if a=='lora' else 'Particle ×'+a.removeprefix('particle_') for a in arms]
    for ax,dataset in zip(axes,['policy','faith']):
        data=report['datasets'][dataset]
        for i,arm in enumerate(arms):
            effect=data['arms'][arm]['effect'];assert effect['complete']
            lo,hi=[100*x for x in effect['paired_family_bootstrap_95ci']]
            value=100*effect['effect_gain'];color='#1e6b4c' if arm=='lora' else '#a27d12'
            ax.plot([lo,hi],[i,i],color=color,lw=2)
            ax.scatter([value],[i],color=color,s=28,zorder=3)
        ax.axvline(0,color='#66706c',ls='--',lw=1)
        ax.set(title=f'{"Primary policies" if dataset=="policy" else "FaithEval transfer"} · n={data["n"]}',
               xlabel='ON − OFF strict success (percentage points)',yticks=range(len(arms)),yticklabels=labels)
        ax.grid(axis='x',alpha=.2);ax.set_ylim(len(arms)-.5,-.5)
    fig.suptitle('Finished-answer effects with paired cluster-bootstrap95% intervals',fontsize=11)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180)
    fig.savefig(out.with_suffix('.svg'))
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();plot(args.release,args.out)
