"""Admit complete-answer remeasurement of fixed oldweights, without new model/judge calls."""
import argparse,fcntl,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def verified(v):assert pin(v['path'])==v;return Path(v['path'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists();os.environ['CUDA_VISIBLE_DEVICES']=''
 from evidence_dial.io import read_jsonl,write_json
 from evidence_dial.behavior_protocol_v1 import messages,QUESTIONS,JEV,ENDPOINT
 from evidence_dial.behavior_fullanswer_health_v3 import passing_guard
 from evidence_dial.evaluate import tokenizer
 from evidence_dial.protocol import digest
 plan=read(a.plan);folder=a.plan.resolve().parent
 for p in plan['inputs']+plan['sources']:verified(p)
 identity=read(folder/'actual-owned-process-first.json');terminal=read(folder/'actual-tool-terminal-receipt-first.json');assert terminal['actual_tool_terminal_collected'] and terminal['actual_tool_terminal_chunk'] and terminal['exit_code']==0 and terminal['reaped'];assert terminal['pid']==identity['pid'] and terminal['birth']==identity['birth'] and terminal['command']==identity['argv'];assert not Path('/proc/'+str(identity['pid'])).exists()
 complete=read(folder/'completion-first.json');assert complete['complete'] and complete['cases_each']==16 and complete['strengths']==[.5,1.,2.] and not complete['selection_performed'] and complete['new_training_updates']==0;assert not (folder/'measurement-failed-first.json').exists()
 resource=read(folder/'resources-first.completed.json');assert resource['state']=='complete' and resource['failure'] is None
 with (folder/'.shared-gpu-job.lock').open('r+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
 old=ROOT/'artifacts/behavior-dial-20261114-v1';closure=read(old/'development-search-closure-first.json');candidate=plan['candidate'];assert candidate==closure['display_checkpoint_representative'];quarter=read(verified(candidate['quarter_admission']));assert quarter['no_weight_EMA'] and quarter['RAW_FULL_NATIVE_network_equality'];assert quarter['NATIVE']==candidate['artifact'];verified(quarter['NATIVE'])
 controls=read(verified(plan['controls']));assert controls['all_answers_native_EOS'];off=read_jsonl(verified(controls['metrics']['off']['predictions']));prompt=read_jsonl(verified(controls['metrics']['prompt-only']['predictions']));rows=read_jsonl(verified(plan['data']));tok=tokenizer();assert len(rows)==len(off)==len(prompt)==16;arms=[]
 import numpy as np
 rng=np.random.default_rng(20261114);indices=rng.integers(0,16,size=(10000,16))
 for alpha in plan['strengths']:
  out=folder/f'alpha-{alpha:g}';records=read_jsonl(out/'predictions.jsonl');assert [r['id'] for r in records]==[r['id'] for r in rows]
  oldarm=next(v for v in quarter['arms'] if v['strength']==alpha);oldrows=read_jsonl(verified(oldarm['predictions']));label=f'old-selected-raw-{candidate["artifact"]["sha256"]}-strength-{alpha:g}'
  for row,record,before,base in zip(rows,records,oldrows,off):
   native=read(out/'native'/(row['id']+'.json'));expected=tok.apply_chat_template(messages(row['question']),add_generation_prompt=True);ids=record['native']['output_ids'];assert native['identity']['arm']==record['arm']==label;assert record['question']==row['question'] and record['case_type']==row['case_type'];assert native['identity']['messages']==messages(row['question'])
   assert record['native']['input_ids']==native['native']['input_ids']==native['identity']['input_ids']==base['native']['input_ids']==expected
   assert ids==native['native']['output_ids'] and 1<=len(ids)<=4096 and ids[-1]==3;assert native['identity']['decoding']=={'do_sample':False,'max_new_tokens':4096,'eos_token_id':3,'use_cache':True};assert record['termination']==native['termination']=='native_eos';assert record['output']==native['output']==tok.decode(ids,skip_special_tokens=True);assert ids[:len(before['native']['output_ids'])]==before['native']['output_ids']
   request={'model':JEV,'state':{'user_message':row['question'],'response':record['output']},'questions':QUESTIONS};key=digest({'endpoint':ENDPOINT,'request':request});judge=read(out/'jev-firsts'/(key+'.json'));assert judge['request']==request and judge['http_status']==200 and judge['response_sha256']==digest(judge['response']) and judge['response']['model']==JEV
   for k,typ,field,maximum in [('effect','score','score',4),('sympathy','noul','noul',1),('sycophancy','noul','noul',1)]:
    answer=judge['response']['answers'][k];assert answer['type']==typ and record['scores'][k]==answer[field] and math.isfinite(answer[field]) and 0<=answer[field]<=maximum
   assert record['scores']['measured'] and record['effect']==record['scores']['effect']/4
  summary=read(out/'summary.json');assert summary['complete'] and summary['cases']==16 and summary['mean_effect']==sum(v['effect'] for v in records)/16
  for k,key in [('sympathy','mean_sympathy_probability'),('sycophancy','mean_sycophancy_probability')]:assert summary[key]==sum(v['scores'][k] for v in records)/16
  guard=passing_guard(out,32,Path(candidate['FULL']['path']));assert guard is not None;comparisons=[]
  for baseline_name,baseline in [('off',off),('prompt-only',prompt)]:
   delta=np.asarray([x['effect']-y['effect'] for x,y in zip(records,baseline)]);comparisons.append({'baseline':baseline_name,'mean_effect_difference':float(delta.mean()),'individual_paired_bootstrap95':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist(),'unit':'matched_DEV_question','resamples':10000,'seed':20261114})
  arms.append({'strength':alpha,'old64_mean_effect':oldarm['mean_effect'],'full_EOS_mean_effect':summary['mean_effect'],'full_sympathy_probability':summary['mean_sympathy_probability'],'full_sycophancy_probability':summary['mean_sycophancy_probability'],'cases':16,'mean_new_tokens':sum(len(v['native']['output_ids']) for v in records)/16,'maximum_new_tokens':max(len(v['native']['output_ids']) for v in records),'all16_native_EOS_and_exact_saved64_prefix':True,'comparisons':comparisons,'predictions':pin(out/'predictions.jsonl'),'summary':pin(out/'summary.json'),'qualified_health':pin(guard)})
 write_json(a.output,{'time':time.time(),'state':'fixed-old-selected-checkpoint-full-answer-DEV-remeasurement-independently-admitted','source':pin(__file__),'plan':pin(a.plan),'actual_root_terminal':pin(folder/'actual-tool-terminal-receipt-first.json'),'candidate':candidate,'controls':plan['controls'],'arms':arms,'new_training_updates':0,'new_model_calls':0,'new_Jev_calls':0,'selection_performed':False,'fresh_holdout_access':False,'scope':'All3displaystrengths reported; oldselectedalpha1 fixed before complete-answer measurement. DEV-only, not untouched confirmation; no new winner chosen from these outputs.'});print(json.dumps({'audit':pin(a.output),'arms':arms,'new_model_calls':0}))
if __name__=='__main__':main()
