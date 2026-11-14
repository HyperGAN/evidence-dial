"""Retain every already-mined context passing the original frozen threshold.

No new model/Jev calls or examined-test inputs. This is a separately identified
training-data derivative, preserving the original37-pair cache and all runs.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def pin(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def verified(value):
    assert pin(value['path'])==value
    return Path(value['path'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists();os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from particle_sliders.pg09_audio import AudioRow,PreparedAudioResponse
    from particle_sliders.lifecycle import atomic_save
    from evidence_dial.behavior_protocol_v1 import TARGETS,QUESTIONS,JEV,ENDPOINT,messages
    from evidence_dial.protocol import digest,MODEL,REVISION
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.io import read_jsonl,write_json
    base=ROOT/'artifacts/behavior-dial-20261114-v1'
    upstream_path=base/'independent-cache-audit-first.json';upstream=json.loads(upstream_path.read_text())
    assert upstream['state']=='all1024-saved-native-and-Jev-firsts-readmitted'
    assert upstream['native_generations_verified']==1024 and upstream['all_own_processes_absent_and_leases_released']
    old=json.loads(verified(upstream['cache_index']).read_text());verified(upstream['responses'])
    assert old['cases']==37 and len(old['rows'])==37 and old['shared_history']==[]
    manifest_path=base/'data/manifest-first.json';manifest=json.loads(manifest_path.read_text())
    rows=read_jsonl(verified(manifest['files']['train']));assert len(rows)==64 and all(r['split']=='train' for r in rows)
    qualified_path=base/'Jev-fixture-qualification-first.json';qualified=json.loads(qualified_path.read_text())
    assert qualified['passed'] and qualified['judge_questions_sha256']==digest(QUESTIONS)
    tok=tokenizer();assignments={};plan_pins=[]
    for gpu in [0,1]:
        path=base/f'mining-GPU{gpu}-v1/plan.json';plan=json.loads(path.read_text())
        assert plan['target_variants_per_case']==16 and plan['target_acceptance_effect_min']==3.
        for item in plan['inputs']+plan['sources']:verified(item)
        for case_id in plan['assigned_ids']:
            assert case_id not in assignments;assignments[case_id]=(path,plan)
        plan_pins.append(pin(path))
    assert set(assignments)=={r['id'] for r in rows}
    prepared=[];index=[];decision_pins=[];coverage=Counter();case_counts=Counter();native_count=0
    old_by_id={r['id']:r for r in old['rows']};all_case_audits=[]
    for row in rows:
        plan_path,plan=assignments[row['id']];shard=plan_path.parent
        neutral_ids=tok.apply_chat_template(messages(row['question']),add_generation_prompt=True)
        np=shard/'neutral-firsts'/(row['id']+'.pt');n=torch.load(np,map_location='cpu',weights_only=False)
        assert n['input_ids']==neutral_ids and n['hidden'].shape==(1,2048) and n['hidden'].dtype==torch.float32 and torch.isfinite(n['hidden']).all()
        measured=[];accepted=[]
        for trial,target in enumerate(TARGETS):
            path=shard/'decisions'/f'{row["id"]}-{trial:02d}.json';decision=json.loads(path.read_text())
            gp=Path(decision['generation']);sp=Path(decision['state']);generation=json.loads(gp.read_text())
            assert decision['id']==row['id'] and decision['trial']==trial and decision['neutral']==str(np)
            assert decision['generation_sha256']==pin(gp)['sha256'] and generation['state_sha256']==pin(sp)['sha256']
            positive=tok.apply_chat_template(messages(row['question'],target),add_generation_prompt=True)
            seed=plan['seed']+int(digest({'id':row['id'],'target':trial})[:8],16)
            assert generation['identity']=={'case':row['id'],'trial':trial,'seed':seed,'system_instruction':target,
                'question':row['question'],'input_ids':positive,'model':MODEL,'revision':REVISION,'plan_sha256':pin(plan_path)['sha256']}
            ids=generation['output_ids'];assert 1<=len(ids)<=128 and (ids[-1]==3 or len(ids)==128)
            assert generation['output']==tok.decode(ids,skip_special_tokens=True)
            assert len(neutral_ids)<=512 and len(positive)<=512
            scores=decision['scores']
            if generation['output'].strip():
                request={'model':JEV,'state':{'user_message':row['question'],'response':generation['output']},'questions':QUESTIONS}
                key=digest({'endpoint':ENDPOINT,'request':request});jp=shard/'jev-firsts'/(key+'.json');j=json.loads(jp.read_text())
                assert j['request']==request and j['http_status']==200 and j['response_sha256']==digest(j['response']) and j['response']['model']==JEV
                answers=j['response']['answers'];refused=any(answers[k].get('type')=='refusal' for k in QUESTIONS)
                assert scores['measured']==(not refused)
                if not refused:
                    for key,kind,field,maximum in [('effect','score','score',4),('sympathy','noul','noul',1),('sycophancy','noul','noul',1)]:
                        assert answers[key]['type']==kind and scores[key]==answers[key][field]
                        assert not isinstance(scores[key],bool) and math.isfinite(scores[key]) and 0<=scores[key]<=maximum
                    assert scores['source']==str(jp);measured.append(decision)
            else:assert not scores['measured']
            native_count+=1;decision_pins.append(pin(path))
            if not scores['measured'] or scores['effect']<3.:continue
            tensors=torch.load(sp,map_location='cpu',weights_only=False);target_hidden=tensors['positive_first_hidden'].clone()
            assert target_hidden.shape==(1,2048) and target_hidden.dtype==torch.float32 and torch.isfinite(target_hidden).all()
            assert tensors['rng_cpu_before'].dtype==tensors['rng_cuda_before'].dtype==torch.uint8
            prepared.append(PreparedAudioResponse(AudioRow(tuple(neutral_ids),tuple(positive),0,1.),target_hidden,n['hidden'].clone()))
            index.append({'id':f'{row["id"]}-target-{trial:02d}','source_case_id':row['id'],'case_type':row['case_type'],
                'history_ids':[],'neutral_ids':neutral_ids,'positive_ids':positive,'target_instruction':target,
                'selected_trial':trial,'effect':scores['effect'],'decision':pin(path),'generation':pin(gp),'positive_state':pin(sp),'neutral_state':pin(np),
                'target_hidden_sha256':hashlib.sha256(target_hidden.contiguous().numpy().tobytes()).hexdigest(),
                'target_from_raw_positive_first_hidden':True})
            accepted.append(trial);case_counts[row['id']]+=1;coverage[row['case_type']]+=1
        best=max(measured,key=lambda v:v['scores']['effect']) if measured else None
        eligible=best is not None and best['scores']['effect']>=3.
        assert eligible==(row['id'] in old_by_id)
        if eligible:
            original=old_by_id[row['id']]
            assert original['selected_trial']==best['trial'] and original['generation']==pin(best['generation'])
        all_case_audits.append({'id':row['id'],'passing_trials':accepted,'original_best_trial':best['trial'] if eligible else None})
    assert native_count==1024 and len(prepared)==len(index)==138 and len(case_counts)==37
    assert all(coverage[k]>=4 for k in ['discouragement','unsupported_conclusion','reasonable_boundary','uncertainty'])
    args.out.mkdir(parents=True);cache=args.out/'cache';cache.mkdir();atomic_save(cache/'responses.pt',prepared)
    write_json(cache/'index.json',{'mode':'behavior-all-passing-MetaGlimmer-empty-history-v1','cases':len(prepared),
        'training_pairs':138,'unique_training_questions':37,'rows':index,'responses_sha256':pin(cache/'responses.pt')['sha256'],
        'training_data':manifest['files']['train'],'upstream_original_cache':upstream['cache_index'],
        'curation_method':'Every measured saved target with unchanged Jev effect>=3; no per-case maximum filter',
        'training_weighting':'Uniform over138 passing context pairs; question exposure varies with passing-context count',
        'cases_per_source_question':dict(case_counts),'pair_coverage':dict(coverage),'neutral_behavior_instruction':'',
        'neutral_system_message_present':False,'shared_history':[],'model_calls_in_merger':0,'weight_EMA':False})
    write_json(args.out/'independent-cache-audit-first.json',{'time':time.time(),'source':pin(__file__),
        'state':'all1024-saved-firsts-reverified-all138-passing-contexts-admitted','cache_index':pin(cache/'index.json'),
        'responses':pin(cache/'responses.pt'),'upstream_original_admission':pin(upstream_path),'mining_plans':plan_pins,
        'Jev_qualification':pin(qualified_path),'source_train':manifest['files']['train'],
        'native_generations_verified':1024,'candidate_decisions':decision_pins,'case_audits':all_case_audits,
        'training_pairs':138,'unique_training_questions':37,'all_original_training_questions_preserved':True,
        'threshold_unchanged':3.,'pair_coverage':dict(coverage),'new_native_model_calls':0,'new_Jev_calls':0,
        'new_training_updates':0,'DEV_or_holdout_rows_read':0,'fresh_training_binding_and_native_proof_still_required':True})
    print(json.dumps({'cache':pin(cache/'index.json'),'pairs':138,'unique_questions':37,'pair_coverage':dict(coverage),
        'new_native_or_Jev_calls':0,'new_training_updates':0}))


if __name__=='__main__':main()
