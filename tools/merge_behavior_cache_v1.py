"""Readmit every saved target and build Meta-Glimmer-style first-state pairs.

No model/teacher/Jev call; completed owned processes and released leases required.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def pin(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def verified(value):
    path=Path(value['path']);assert pin(path)==value;return path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True)
    ap.add_argument('--terminal-receipt',type=Path,action='append',required=True)
    args=ap.parse_args();folder=args.folder.resolve();assert len(args.terminal_receipt)==2
    assert not (folder/'cache').exists() and not (folder/'independent-cache-audit-first.json').exists()
    from evidence_dial.behavior_protocol_v1 import TARGETS,QUESTIONS,messages,JEV,ENDPOINT
    from evidence_dial.protocol import digest,MODEL,REVISION
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.io import read_jsonl,write_json
    from evidence_dial.long_measure_admission_v13 import require_root_identity
    import torch
    from particle_sliders.pg09_audio import AudioRow,PreparedAudioResponse
    from particle_sliders.lifecycle import atomic_save
    manifest=json.loads((folder/'data/manifest-first.json').read_text());rows=read_jsonl(verified(manifest['files']['train']))
    assert len(rows)==64 and all(r['split']=='train' for r in rows)
    q=json.loads((folder/'Jev-fixture-qualification-first.json').read_text());assert q['passed'] and q['judge_questions_sha256']==digest(QUESTIONS)
    tok=tokenizer();receipts=[json.loads(p.read_text()) for p in args.terminal_receipt]
    by_id={r['id']:r for r in rows};seen=set();accepted={};all_audits=[];native_count=0;judgments=0;unique_judgments=set()
    for gpu in [0,1]:
        shard=folder/f'mining-GPU{gpu}-v1';plan_path=shard/'plan.json';plan=json.loads(plan_path.read_text())
        for item in plan['sources']+plan['inputs']+[plan['training_data']]:verified(item)
        identity=json.loads((shard/'actual-tool-identity-first.json').read_text());terminal=next(r for r in receipts if r['tool_handle']==identity['tool_handle'])
        require_root_identity(terminal,identity);assert terminal['actual_tool_terminal_chunk'] and not Path(f'/proc/{identity["pid"]}').exists()
        resource=json.loads((shard/'resources-first.completed.json').read_text());assert resource['state']=='complete' and resource['failure'] is None
        with (shard/'.shared-gpu-job.lock').open('r+') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(h,fcntl.LOCK_UN)
        complete=json.loads((shard/'completion-first.json').read_text());assert complete['complete'] and complete['cases']==32 and complete['target_variants_per_case']==16
        selection={x['id']:x for x in complete['selected']};excluded={x['id'] for x in complete['ineligible']}
        assert set(selection).isdisjoint(excluded) and set(selection)|excluded==set(plan['assigned_ids'])
        for case_id in plan['assigned_ids']:
            assert case_id not in seen;seen.add(case_id);row=by_id[case_id]
            neutral_ids=tok.apply_chat_template(messages(row['question']),add_generation_prompt=True)
            np=shard/'neutral-firsts'/(case_id+'.pt');neutral=torch.load(np,map_location='cpu',weights_only=False)
            assert neutral['input_ids']==neutral_ids and neutral['hidden'].shape==(1,2048) and neutral['hidden'].dtype==torch.float32 and torch.isfinite(neutral['hidden']).all()
            trials=[]
            for trial,target in enumerate(TARGETS):
                gp=shard/'native-firsts'/f'{case_id}-target-{trial:02d}.json';sp=gp.with_suffix('.pt')
                generation=json.loads(gp.read_text());i=generation['identity'];ids=generation['output_ids']
                positive_ids=tok.apply_chat_template(messages(row['question'],target),add_generation_prompt=True)
                seed=plan['seed']+int(digest({'id':case_id,'target':trial})[:8],16)
                assert i=={'case':case_id,'trial':trial,'seed':seed,'system_instruction':target,'question':row['question'],'input_ids':positive_ids,'model':MODEL,'revision':REVISION,'plan_sha256':pin(plan_path)['sha256']}
                assert 1<=len(ids)<=128 and (ids[-1]==3 or len(ids)==128)
                assert generation['output']==tok.decode(ids,skip_special_tokens=True)
                assert generation['state_sha256']==pin(sp)['sha256']
                tensors=torch.load(sp,map_location='cpu',weights_only=False);target_hidden=tensors['positive_first_hidden']
                assert target_hidden.shape==(1,2048) and target_hidden.dtype==torch.float32 and torch.isfinite(target_hidden).all()
                assert tensors['rng_cpu_before'].dtype==tensors['rng_cuda_before'].dtype==torch.uint8
                decision=json.loads((shard/'decisions'/f'{case_id}-{trial:02d}.json').read_text())
                assert decision['generation']==str(gp) and decision['generation_sha256']==pin(gp)['sha256']
                assert decision['state']==str(sp) and decision['neutral']==str(np) and decision['trial']==trial
                scores=decision['scores']
                if generation['output'].strip():
                    request={'model':JEV,'state':{'user_message':row['question'],'response':generation['output']},'questions':QUESTIONS}
                    key=digest({'endpoint':ENDPOINT,'request':request});jp=shard/'jev-firsts'/(key+'.json');raw=json.loads(jp.read_text())
                    assert raw['http_status']==200 and raw['request']==request and raw['response_sha256']==digest(raw['response']) and raw['response']['model']==JEV
                    answers=raw['response']['answers'];refused=any(answers[k].get('type')=='refusal' for k in QUESTIONS)
                    assert scores['measured']==(not refused)
                    if not refused:
                        expected={'effect':answers['effect']['score'],'sympathy':answers['sympathy']['noul'],'sycophancy':answers['sycophancy']['noul']}
                        assert answers['effect']['type']=='score' and answers['sympathy']['type']==answers['sycophancy']['type']=='noul'
                        assert all(not isinstance(x,bool) and math.isfinite(x) and 0<=x<=(4 if k=='effect' else 1) for k,x in expected.items())
                        assert all(scores[k]==v for k,v in expected.items()) and scores['source']==str(jp)
                    judgments+=1;unique_judgments.add(str(jp))
                else:assert not scores['measured']
                if scores['measured']:trials.append(decision)
                native_count+=1
            best=max(trials,key=lambda x:x['scores']['effect']) if trials else None
            should_accept=best is not None and best['scores']['effect']>=3.
            assert (case_id in selection)==should_accept
            if should_accept:
                assert selection[case_id]==best;accepted[case_id]=best
            all_audits.append({'id':case_id,'trials':16,'accepted':should_accept,'selected_trial':best['trial'] if should_accept else None})
        for quarter,count in enumerate([8,16,24,32],1):
            marker=json.loads((shard/f'quarter-{quarter:02d}-finished-first.json').read_text());assert marker['finished_cases']==count and marker['declared_cases']==32
        for pending in (shard/'native-firsts').glob('*.pending.json'):assert pending.with_name(pending.name.replace('.pending.json','.json')).exists()
        for pending in (shard/'neutral-firsts').glob('*.pending.json'):assert pending.with_name(pending.name.replace('.pending.json','.pt')).exists()
    assert seen==set(by_id) and native_count==1024
    coverage=Counter(by_id[k]['case_type'] for k in accepted)
    assert len(accepted)>=32 and all(coverage[k]>=4 for k in ['discouragement','unsupported_conclusion','reasonable_boundary','uncertainty']), 'Insufficient declared broad coverage; no training'
    prepared,index=[],[]
    for row in rows:
        if row['id'] not in accepted:continue
        best=accepted[row['id']];gp=Path(best['generation']);generation=json.loads(gp.read_text());n=torch.load(best['neutral'],map_location='cpu',weights_only=False)
        target=torch.load(best['state'],map_location='cpu',weights_only=False)['positive_first_hidden'].clone()
        prepared.append(PreparedAudioResponse(AudioRow(tuple(n['input_ids']),tuple(generation['identity']['input_ids']),0,1.),target,n['hidden'].clone()))
        index.append({'id':row['id'],'case_type':row['case_type'],'history_ids':[],'neutral_ids':n['input_ids'],'positive_ids':generation['identity']['input_ids'],
            'target_instruction':generation['identity']['system_instruction'],'selected_trial':best['trial'],'effect':best['scores']['effect'],
            'generation':pin(gp),'positive_state':pin(best['state']),'neutral_state':pin(best['neutral']),
            'target_hidden_sha256':hashlib.sha256(target.contiguous().numpy().tobytes()).hexdigest(),'target_from_raw_positive_first_hidden':True})
    cache=folder/'cache';cache.mkdir();atomic_save(cache/'responses.pt',prepared)
    write_json(cache/'index.json',{'mode':'behavior-MetaGlimmer-empty-history-v1','cases':len(prepared),'rows':index,'responses_sha256':pin(cache/'responses.pt')['sha256'],
        'training_data':manifest['files']['train'],'manifest':pin(folder/'data/manifest-first.json'),'curation_method':'maxJev effect among16, minimum3','coverage':dict(coverage),
        'neutral_behavior_instruction':'','neutral_system_message_present':False,'shared_history':[],'model_calls_in_merger':0,'weight_EMA':False})
    write_json(folder/'independent-cache-audit-first.json',{'time':time.time(),'source':pin(__file__),'state':'all1024-saved-native-and-Jev-firsts-readmitted',
        'cache_index':pin(cache/'index.json'),'responses':pin(cache/'responses.pt'),'native_generations_verified':native_count,'Jev_candidate_judgment_records_verified':judgments,'unique_Jev_firsts_verified':len(unique_judgments),
        'cases':len(prepared),'ineligible_cases':[r['id'] for r in rows if r['id'] not in accepted],'coverage':dict(coverage),'case_audits':all_audits,
        'both_actual_terminal_receipts':[pin(p) for p in args.terminal_receipt],'all_own_processes_absent_and_leases_released':True,
        'new_model_calls':0,'new_Jev_calls':0,'new_training_updates':0,'DEV_or_holdout_rows_read':0})
    print(json.dumps({'cache':pin(cache/'index.json'),'accepted':len(prepared),'verified_generations':1024,'new_model_calls':0}))


if __name__=='__main__':main()
