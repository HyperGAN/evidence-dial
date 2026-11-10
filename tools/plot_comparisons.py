"""Plot every selected particle/LoRA comparison, including all transfer strata."""
from pathlib import Path
import argparse,json


def plot(release,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((release/'results.json').read_text());selection=r['selection']
    corrected='previous_recipe_selection' in selection
    labels={'confirmation':'Fresh confirmation','policy':'Original policies',
            'faith':'FaithEval','cue_audit':'Cue audit','balanced':'Balanced stress'}
    if corrected:labels.update({k:v+' · repeated' for k,v in labels.items() if k!='confirmation'})
    entries=[]
    for name in ['confirmation','policy','faith','cue_audit','balanced']:
        if name not in r['datasets']:continue
        entries.append((name,None,f"{labels[name]} (n={r['datasets'][name]['n']})"))
        if name=='faith':
            entries.extend((name,s,f"    {s.removeprefix('faith-')} (n=300)")
                           for s in ['faith-counterfactual','faith-inconsistent','faith-unanswerable'])
    groups=[('Jev-curated particle vs same-pool LoRA','comparisons_vs_lora','particle_',selection['selected_ties'])]
    if corrected:
        groups.append(('Original hinted particle vs original64-case LoRA',
                       'comparisons_vs_lora_original','hinted_particle_',selection['previous_recipe_selection']['selected_ties']))
    else:groups[0]=('Original particle vs original LoRA',*groups[0][1:])
    fig,axes=plt.subplots(1,len(groups),figsize=(7*len(groups),max(4.5,.58*len(entries)+1.5)),
                          constrained_layout=True,squeeze=False)
    for ax,(title,key,prefix,ties) in zip(axes[0],groups):
        for j,tie in enumerate(ties):
            strength=tie['strength'];arm=prefix+f'{strength:g}';color=['#a27d12','#1e6b4c'][j%2]
            for i,(dataset,stratum,label) in enumerate(entries):
                comparison=r['datasets'][dataset][key][arm]
                estimate=comparison['by_stratum'][stratum] if stratum else comparison['paired_success_difference']
                assert estimate['complete']
                v=100*estimate['effect_gain'];lo,hi=[100*x for x in estimate['paired_family_bootstrap_95ci']]
                y=i+(j-(len(ties)-1)/2)*.15
                ax.plot([lo,hi],[y,y],color=color,lw=2)
                ax.scatter([v],[y],s=30,color=color,label=f'Dev-selected strength{strength:g}' if i==0 else None)
                ax.annotate(f'{v:+.1f}',(hi,y),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
        ax.axvline(0,color='#66706c',ls='--',lw=1);ax.grid(axis='x',alpha=.15)
        ax.set(title=title,yticks=range(len(entries)),yticklabels=[e[2] for e in entries],
               xlabel='Particle − matched LoRA strict success (percentage points)',ylim=(len(entries)-.5,-.5))
        ax.legend(loc='lower left',fontsize=8)
        ax.margins(x=.12)
    fig.suptitle('Paired finished-answer differences · 95% family/source-question bootstrap intervals',fontsize=11)
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180);fig.savefig(out.with_suffix('.svg'));plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.release,a.out)
