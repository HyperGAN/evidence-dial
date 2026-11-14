"""Export all audited complete-answer DEV responses from the fixed published checkpoint."""
from pathlib import Path
import argparse,json,time,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def verified(v):assert pin(v['path'])==v;return Path(v['path'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
 from evidence_dial.io import read_jsonl,write_json
 from evidence_dial.behavior_protocol_v1 import TARGETS
 audit=read(a.audit);assert audit['state']=='fixed-old-selected-checkpoint-full-answer-DEV-remeasurement-independently-admitted' and not audit['selection_performed'];controls=read(verified(audit['controls']));off=read_jsonl(verified(controls['metrics']['off']['predictions']));prompt=read_jsonl(verified(controls['metrics']['prompt-only']['predictions']));records={str(v['strength']).rstrip('0').rstrip('.'):read_jsonl(verified(v['predictions'])) for v in audit['arms']};mid='published-step32-complete-answer-DEV'
 def record(r):
  assert r['native']['output_ids'][-1]==3
  return {'output':r['output'],'effect':r['effect'],'sympathy':r['scores']['sympathy'],'sycophancy':r['scores']['sycophancy'],'termination':'native_eos','output_tokens':len(r['native']['output_ids'])}
 cases=[]
 for i,(o,p) in enumerate(zip(off,prompt)):
  assert o['id']==p['id'] and o['question']==p['question'];values={}
  for alpha,rows in records.items():
   r=rows[i];assert r['id']==o['id'] and r['question']==o['question'] and r['native']['input_ids']==o['native']['input_ids'];values[alpha]=record(r)
  cases.append({'id':o['id'],'case_type':o['case_type'],'question':o['question'],'off':record(o),'prompt_only':record(p),'models':{mid:values}})
 assert len(cases)==16
 metrics=[{'strength':v['strength'],'mean_effect':v['full_EOS_mean_effect'],'mean_sympathy_probability':v['full_sympathy_probability'],'mean_sycophancy_probability':v['full_sycophancy_probability'],'paired_bootstrap95':next(c for c in v['comparisons'] if c['baseline']=='off')['individual_paired_bootstrap95']} for v in audit['arms']]
 def cm(v):return {k:v[k] for k in ['mean_effect','mean_sympathy_probability','mean_sycophancy_probability']}
 data={'schema':'particle-behavior-explorer-v1','exported_at':time.time(),'split':'dev','status':'Complete-answer development remeasurement of the published32-update adapter. All80 displayed responses end naturally. Fresh138-pair training from0 through1024 updates is in progress on bothGPUs; this gallery is not the new model or an untouched-test result.','scope_note':'The checkpoint and strength1 were selected under the previous64-token development protocol. This remeasurement reports all16DEVcases and three fixed display strengths, without choosing another winner.','uncertainty_note':'Individual paired question-bootstrap intervals are descriptive DEV diagnostics; they do not establish a held-out advantage.','prompt_instruction':TARGETS[0],'cases':cases,'models':[{'id':mid,'title':'Published checkpoint · complete answers','description':'32updates; G.001;37training pairs; raw weights; previously selected under64-token DEV protocol','sha256':audit['candidate']['artifact']['sha256'],'metrics':metrics}],'controls':{'off':cm(controls['metrics']['off']),'prompt_only':cm(controls['metrics']['prompt-only'])},'secondary_qa':'Separate earlier QA study: PG and SFT each174/192 on DEV; on sealed256-question test, all five preselectedPG arms221–229, four SFT controls230–234, OFF142. This is a DEV tie and held-out gap, not SFT parity.','provenance':{'export_source_sha256':pin(__file__)['sha256'],'audit_sha256s':[pin(a.audit)['sha256'],audit['controls']['sha256']],'no_generated_or_interpolated_demo_text':True,'all_cases_in_declared_split_included':True,'all_answers_native_EOS':True,'maximum_new_tokens_safetycap':4096,'same_saved64_token_prefixes_independently_verified':True},'all_holdout_results':None}
 assert '/ml2/' not in json.dumps(data) and '/mnt/nvme/' not in json.dumps(data);write_json(a.output,data);print(json.dumps({'export':pin(a.output),'cases':16,'complete_responses':80,'new_model_or_Jev_calls':0}))
if __name__=='__main__':main()
