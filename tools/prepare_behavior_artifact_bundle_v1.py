"""Prepare model/data upload trees from completed, independently audited runs.

Explicit files only. No credentials, private Git/Babel history, new model
calls, training, or score-based choices on held-out data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--closure',type=Path,required=True)
    ap.add_argument('--holdout-audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    from evidence_dial.behavior_confirmation_v1 import BASE,pin,verified,read,validate_plan
    from evidence_dial.io import read_jsonl,write_json,write_jsonl
    closure=read(args.closure);assert closure['state']=='declared-two-rate-24-cell-DEV-search-complete'
    audit=read(args.holdout_audit);assert audit['state']=='all-locked-behavior-holdout-arms-independently-admitted'
    plan=validate_plan(read(verified(audit['plan'])));assert plan['development_closure']==pin(args.closure)
    assert not audit['holdout_based_reselection'] and audit['cases_each']==32
    model=args.out/'model';dataset=args.out/'dataset';manifest=[]
    assert len(closure['all24_arms'])==24
    def copy(source,destination):
        source=Path(source);data=source.read_bytes()
        if source.suffix in ['.json','.jsonl','.txt','.md','.py']:
            assert not re.search(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})',data),source.name
        destination.parent.mkdir(parents=True,exist_ok=True);assert not destination.exists();destination.write_bytes(data)
        manifest.append({'path':str(destination.relative_to(args.out)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source':str(source.resolve())})
    def document(path,text):
        path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();path.write_text(text)
        manifest.append({'path':str(path.relative_to(args.out)),'bytes':path.stat().st_size,'sha256':pin(path)['sha256'],'generated_from_audited_records':True})
    summary_by_name={a['name']:a for a in audit['arms']};model_table=[];weight_artifacts={}
    for chosen in closure['all_maximum_ties']:
        candidate=next(a for a in plan['arms'] if a['name']==chosen['name']);assert candidate['candidate']==chosen
        scores=summary_by_name[chosen['name']];comparison=next(c for c in audit['comparisons'] if c['arm']==chosen['name'] and c['baseline']=='off')
        model_table.append('| '+chosen['name']+f' | {chosen["mean_effect"]:.6f} | {scores["mean_effect"]:.6f} | {comparison["mean_effect_difference"]:+.6f} | '+str(comparison['paired_bootstrap95'])+' |')
        key=chosen['artifact']['sha256']
        if key not in weight_artifacts:
            folder=model/'checkpoints'/key[:16];copy(verified(chosen['artifact']),folder/'adapter-native.pt')
            quarter=read(verified(chosen['quarter_admission']));copy(verified(quarter['RAW']),folder/'adapter-raw.pt')
            weight_artifacts[key]={'native_file':str((folder/'adapter-native.pt').relative_to(model)),
                'raw_file':str((folder/'adapter-raw.pt').relative_to(model)),'native_sha256':key}
    display=plan['display_checkpoint'];assert display['sha256'] in weight_artifacts
    first=weight_artifacts[display['sha256']]
    copy(model/first['native_file'],model/'adapter-native.pt');copy(model/first['raw_file'],model/'adapter-raw.pt')
    write_json(model/'selection.json',{'selection':'all maximum ties on equally weighted DEV effect ON minus OFF',
        'selected_candidates':[{k:a[k] for k in ['name','step','g_lr','strength','mean_effect','effect_ON_minus_OFF']} for a in closure['all_maximum_ties']],
        'weight_artifacts':weight_artifacts,'display_native_sha256':display['sha256'],
        'root_files':'Stable presentation representative chosen before holdout; all maximum ties retained',
        'weight_EMA':False,'base_model':'CohereLabs/tiny-aya-global','base_revision':'af89d219b53ed9b13b8a4645f9c8028973510324'})
    manifest_data=read(BASE/'data/manifest-first.json')
    for split in ['train','dev','holdout']:copy(verified(manifest_data['files'][split]),dataset/'questions'/f'{split}.jsonl')
    by_id={r['id']:r for r in read_jsonl(manifest_data['files']['train']['path'])}
    attempts=[]
    for gpu in [0,1]:
        shard=BASE/f'mining-GPU{gpu}-v1';mining=read(shard/'plan.json')
        for case in mining['assigned_ids']:
            for trial in range(16):
                d=read(shard/'decisions'/f'{case}-{trial:02d}.json');g=read(d['generation'])
                assert pin(d['generation'])['sha256']==d['generation_sha256'] and pin(d['state'])['sha256']==g['state_sha256']
                attempts.append({'id':f'{case}-target-{trial:02d}','source_case_id':case,'split':'train',
                    'case_type':by_id[case]['case_type'],'question':by_id[case]['question'],
                    'instruction':g['identity']['system_instruction'],'trial':trial,'seed':g['identity']['seed'],
                    'response':g['output'],'output_ids':g['output_ids'],'positive_input_ids':g['identity']['input_ids'],
                    'positive_state_sha256':g['state_sha256'],'effect_score':d['scores'].get('effect'),
                    'sympathy_probability':d['scores'].get('sympathy'),'sycophancy_probability':d['scores'].get('sycophancy'),
                    'measured':d['scores']['measured'],'passes_original_threshold':d['scores']['measured'] and d['scores']['effect']>=3.})
                copy(d['state'],dataset/'curation/native-states'/f'{case}-target-{trial:02d}.pt')
        copy(shard/'plan.json',dataset/'curation'/f'mining-GPU{gpu}-plan.json')
    assert len(attempts)==1024
    write_jsonl(dataset/'curation/attempts-1024.jsonl',attempts)
    old=read(BASE/'cache/index.json');selected_ids={f'{r["id"]}-target-{r["selected_trial"]:02d}' for r in old['rows']}
    selected=[r for r in attempts if r['id'] in selected_ids];passing=[r for r in attempts if r['passes_original_threshold']]
    assert len(selected)==37 and len(passing)==138
    write_jsonl(dataset/'curation/selected-37.jsonl',selected);write_jsonl(dataset/'curation/passing-138.jsonl',passing)
    copy(BASE/'cache/responses.pt',dataset/'training-pairs/selected-37.pt')
    richer=ROOT/'artifacts/behavior-all-passing-20261114-v1';copy(richer/'cache/responses.pt',dataset/'training-pairs/all-passing-138.pt')
    copy(richer/'independent-cache-audit-first.json',dataset/'audits/all-passing-cache-admission.json')
    copy(BASE/'independent-cache-audit-first.json',dataset/'audits/original-cache-admission.json')
    copy(BASE/'Jev-fixture-qualification-first.json',dataset/'audits/Jev-fixtures.json')
    for cohort in closure['cohorts']:
        train_plan=read(verified(cohort['plan']));folder=Path(cohort['plan']['path']).parent;label=train_plan['cohort']
        copy(folder/'plan.json',dataset/'training'/label/'plan.json')
        copy(folder/'run/updates.jsonl',dataset/'training'/label/'updates.jsonl')
        copy(folder/'run/initialization.json',dataset/'training'/label/'initialization.json')
        copy(folder/'run/native-check.json',dataset/'training'/label/'native-check.json')
        for step in [0,32,64,96,128]:
            copy(folder/'run'/f'FULL-{step:06d}.pt',dataset/'training'/label/f'FULL-{step:06d}.pt')
        for quarter_pin in cohort['independent_quarters']:
            quarter=read(verified(quarter_pin));step=quarter['step']
            copy(verified(quarter_pin),dataset/'audits'/f'{label}-quarter{step}.json')
            copy(verified(quarter['NATIVE']),dataset/'training'/label/f'NATIVE-{step:06d}.pt')
            for arm in quarter['arms']:
                target=dataset/'evaluation/dev'/label/f'step{step}-alpha{arm["strength"]:g}'
                copy(verified(arm['predictions']),target/'predictions.jsonl');copy(verified(arm['summary']),target/'summary.json')
    controls=BASE/'DEV-controls-GPU0-v1'
    for arm in ['off','prompt-only']:
        copy(controls/arm/'predictions.jsonl',dataset/'evaluation/dev'/arm/'predictions.jsonl')
        copy(controls/arm/'summary.json',dataset/'evaluation/dev'/arm/'summary.json')
    for arm in audit['arms']:
        copy(verified(arm['predictions']),dataset/'evaluation/holdout'/arm['name']/'predictions.jsonl')
        copy(verified(arm['summary']),dataset/'evaluation/holdout'/arm['name']/'summary.json')
    copy(args.closure,dataset/'audits/development-search-closure.json');copy(args.holdout_audit,dataset/'audits/heldout-comparison.json')
    copy(verified(audit['plan']),dataset/'evaluation/holdout/plan.json')
    qa=ROOT/'artifacts/fixed-formulation-pursuit-20261112-v1/round2-locked-all5-PG-fourSFT-confirmation-v1'
    for name in ['independent-completion-audit-first.json','round2-confirmation-comparison-first.json','confirmation-plan-first.json']:
        copy(qa/name,dataset/'secondary-QA'/name)
    model_card='''---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
base_model_relation: adapter
library_name: particle-sliders-core
pipeline_tag: text-generation
inference: false
tags: [particlegan, preference-alignment, internal-states, cohere]
---
# Evidence Dial behavior adapter

Raw ParticleGAN adapters for Tiny Aya Global3.35B, targeting sympathetic support with independent judgment. The backbone is frozen. The training pairs are37 selected positive/neutral first-answer hidden states from64 training questions; no SFT/text-imitation loss or weight EMA is used. A separate138-pair dataset is prepared but was not used to train these adapters.

All development maximum ties are retained. Root adapter files identify the presentation representative fixed before held-out inference; they do not select a new winner on test. These are custom Particle Sliders artifacts, not standard PEFT weights.

| Preselected candidate | DEV behavior score | Held-out behavior score | Held-out ON minus OFF | Individual paired95% interval |
| --- | ---: | ---: | ---: | --- |
'''+ '\n'.join(model_table)+'''

Scores are Jev model judgments, not accuracy percentages. Complete OFF, prompt-only, all selected ties and display strengths are in the companion dataset. Small synthetic corpus, one training seed, a shared curation/evaluation judge and64-token generation limit restrict generalization. Stronger steering may regress. See the project README for the exact conclusion and the earlier QA study's held-out gap to SFT.

Base revision: `af89d219b53ed9b13b8a4645f9c8028973510324`. Core0.14.2/PG0.9.5; full BF16/SDPA GPU backbone. Install the pinned project dependencies and download that base snapshot. Load `adapter-native.pt` with `particle_sliders.inference.open_inference`, using `evidence_dial.behavior_binding_v1.inference_backend()`. Within the context, `binding.response.__self__` exposes `model` and `slider`; use `slider.scaled(strength)` with the unchanged user chat message. The source evaluator `behavior_generate_v1.py` records the actual native greedy64 path.

Code: https://github.com/HyperGAN/evidence-dial
Data and logs: https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior
Retain Tiny Aya's CC-BY-NC-4.0 noncommercial license and upstream code licenses.
'''
    document(model/'README.md',model_card)
    dataset_card='''---
license: cc-by-nc-4.0
language: [en]
tags: [particlegan, preference-alignment, internal-states, cohere]
configs:
  - config_name: questions
    data_files:
      - split: train
        path: questions/train.jsonl
      - split: validation
        path: questions/dev.jsonl
      - split: test
        path: questions/holdout.jsonl
  - config_name: mining
    data_files:
      - split: train
        path: curation/attempts-1024.jsonl
---
# Evidence Dial behavior data and experiment record

Track1section7 preference/style alignment on Tiny Aya Global3.35B. Questions are synthetic:64 TRAIN,16 DEV,32 held-out. The original behavior models train on37 selected context pairs. Each pair keeps the same question in the neutral and positive contexts, adds a behavioral system instruction only to the positive context, and has empty continuation history. Targets are frozen final hidden states before the first response token.

`curation/attempts-1024.jsonl` preserves all64questions×16instruction trials and model-judged effects, including failures. `selected-37.jsonl` contains the original max-per-question contexts meeting the frozen3/4 threshold. `passing-138.jsonl` retains all138 contexts meeting that same threshold across the same37questions; it is a separate prepared data derivative, not138 distinct questions and not training data used by the original models. Sampling those context pairs uniformly gives unequal question exposure.

`training-pairs` contains actual saved positive/neutral states. `curation/native-states` retains positive first states and sampling RNG state for all1024 trials. `training` preserves raw/full snapshots, native exports, exact update journals and scoped initialization records. `evaluation` contains every finished development and locked held-out arm. `audits` binds the source, model, data, first observations and decisions. Original host paths in archived provenance identify recorded executions; port them using an explicit new execution configuration.

The complete held-out comparison includes OFF, a fixed prompt-only instruction, all development-selected maximum ties, and a three-strength grid from the display checkpoint frozen before test. The display grid is descriptive; test scores do not select another model. The primary selector is equal-question mean intended-behavior score ON minus matched OFF. Sympathy and sycophancy judgments are separate diagnostics.

Jev is `typesafe/jev-1.14` through NanoGPT. Scores are model judgments, not accuracy percentages or human preference rates. Curation uses128-token sampled replies; all evaluation arms use the same64-token greedy cap and native EOS, retaining truncation. Weight EMA is disabled; all adapter, critic and bank weights are raw. The small synthetic corpus, one training seed, shared curation/evaluation judge and first-state supervision limit generalization.

`secondary-QA` contains a separate completed confirmation: PG and SFT tie at174/192 on DEV, while all five preselected PG arms score221–229/256, all four SFT controls230–234/256, and OFF142/256 on sealed test. This establishes a development tie and held-out gap, not SFT parity. Earlier grounding/FaithEval results remain separate in the public project.

Code: https://github.com/HyperGAN/evidence-dial
Adapter: https://huggingface.co/ntc-ai/evidence-dial-behavior-particle
Model-derived outputs and states retain CC-BY-NC-4.0; retain upstream code licenses. No credentials or private Git/Babel history are included.
'''
    document(dataset/'README.md',dataset_card)
    public_manifest=[]
    for tree in [model,dataset]:
        for path in sorted(tree.rglob('*')):
            if path.is_file():
                public_manifest.append({'path':str(path.relative_to(args.out)),
                    'bytes':path.stat().st_size,'sha256':pin(path)['sha256']})
    write_json(args.out/'upload-manifest-first.json',{'time':time.time(),'files':public_manifest,
        'development_closure':pin(args.closure),'heldout_audit':pin(args.holdout_audit),'new_model_calls':0,'new_Jev_calls':0,
        'heldout_based_selection':False,'original138_pair_cache_not_trained':True})
    write_json(BASE/'artifact-bundle-preparation-first.json',{'time':time.time(),'source':pin(__file__),'bundle':str(args.out.resolve()),
        'manifest':pin(args.out/'upload-manifest-first.json'),'files':len(public_manifest),'copy_sources':manifest,
        'public_upload_not_yet_performed':True,'new_model_calls':0,'new_Jev_calls':0})
    print(json.dumps({'bundle':str(args.out.resolve()),'files':len(public_manifest),'bytes':sum(v['bytes'] for v in public_manifest),
        'new_model_calls':0,'public_upload_still_required':True}))


if __name__=='__main__':main()
