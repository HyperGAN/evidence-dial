"""Declare the owner's fresh138-pair1024-update study before any update."""
import copy,hashlib,importlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.environ['CUDA_VISIBLE_DEVICES']=''

def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
 from evidence_dial.io import write_json
 from evidence_dial.behavior_expanded_binding_v2 import validate,Native,initialize,PROTOCOL,NAME
 from evidence_dial.binding import Native as OriginalNative,initialize as OriginalInitialize
 from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter,AudioResponseCritic
 from particle_sliders.pg09 import PG_VERSION
 from particle_sliders.lifecycle import SessionConfig
 from evidence_dial.protocol import SEED
 folder=ROOT/'artifacts/behavior-expanded-h1024-20261114-v2';old=ROOT/'artifacts/behavior-dial-20261114-v1';expanded=ROOT/'artifacts/behavior-all-passing-20261114-v1';cache=expanded/'cache';admission=expanded/'independent-cache-audit-first.json'
 a=json.loads(admission.read_text());assert a['state']=='all1024-saved-firsts-reverified-all138-passing-contexts-admitted' and a['training_pairs']==138 and a['unique_training_questions']==37
 assert a['cache_index']==pin(cache/'index.json') and a['responses']==pin(cache/'responses.pt')
 index=json.loads((cache/'index.json').read_text());assert index['cases']==len(index['rows'])==138
 manifest=json.loads((old/'data/manifest-first.json').read_text());data=manifest['files']['train'];dev=manifest['files']['dev'];assert pin(data['path'])==data and pin(dev['path'])==dev
 assert Native._hidden is OriginalNative._hidden and Native._response is ARLastHiddenResponseAdapter._response and Native.generated is ARLastHiddenResponseAdapter.generated and initialize is OriginalInitialize and PG_VERSION=='0.9.5'
 critic=AudioResponseCritic(2048,2052);assert tuple(critic.net[0].weight.shape)==(16,4100) and type(critic.net[1]).__name__=='SiLU' and tuple(critic.net[2].weight.shape)==(1,16)
 evaluator=folder/'CPU-AuditKIT-evaluator-qualification-first.json';e=json.loads(evaluator.read_text());assert e['actual_AuditKIT_inferences']==4 and e['generation_source_sha256']==pin(ROOT/'src/evidence_dial/behavior_fullanswer_generate_v2.py')['sha256']
 health=folder/'fullanswer-health-qualification-v3-first.json';assert json.loads(health.read_text())['qualified']
 original=ROOT/'artifacts/first-answer-control-20261111-v1/runs/first-answer/FULL-000000.pt'
 names=['tools/prepare_behavior_expanded_training_v2.py','tools/behavior_expanded_phase_worker_v2.py','tools/run_behavior_expanded_training_v2.py','tools/check_behavior_fullanswer_health_v2.py','src/evidence_dial/behavior_expanded_binding_v2.py','src/evidence_dial/behavior_expanded_native_proof_v2.py','src/evidence_dial/behavior_fullanswer_generate_v2.py','src/evidence_dial/behavior_fullanswer_health_v3.py','src/evidence_dial/behavior_protocol_v1.py','src/evidence_dial/binding.py','src/evidence_dial/first_answer_binding_v2.py','src/evidence_dial/evaluate.py','src/evidence_dial/shared_gpu_resources_v1.py','src/evidence_dial/continuation_resources_v1.py','src/evidence_dial/fresh_native_zero_proof_v2.py']
 sources=[pin(ROOT/n) for n in names];runtime=[pin(importlib.import_module(n).__file__) for n in ['particle_sliders.pg09','particle_sliders.pg09_audio','particle_sliders.lifecycle','particle_sliders.inference','particle_sliders.routed_projection','particle_sliders.initialization']]
 workspace=Path('/mnt/nvme/lexica-evidence-dial-work/01a1c84b-5d17-7e32-80ef-34d8e4ee2e62');assert shutil.disk_usage(workspace).free>55*1024**3,'Need observed room for2x1024 exact FULL journals plus retained history'
 g0=workspace/'behavior-expanded138-h1024-GPU0-g001-v2';g1=workspace/'behavior-expanded138-h1024-GPU1-g00025-v2';assert not g0.exists() and not g1.exists() and not (folder/'training-declaration-first.json').exists()
 inputs=[data,dev,pin(admission),pin(cache/'index.json'),pin(cache/'responses.pt'),pin(original),pin(old/'data/manifest-first.json'),pin(old/'Jev-fixture-qualification-first.json'),pin(evaluator),pin(health),pin(folder/'USER-REQUEST-first.json')]
 declarations=[];negative_checks=0
 for gpu,rate,out in [(0,.001,g0),(1,.00025,g1)]:
  options={'mode':'first-answer','steps':1024,'batch':8,'g_lr':rate,'d_lr':.018,'bank_lr':.03,'ema_decay':None,'base_precision':'bfloat16','seed':SEED,'binding_sha256':pin(ROOT/'src/evidence_dial/behavior_expanded_binding_v2.py')['sha256'],'training_data':data['path'],'training_data_sha256':data['sha256'],'cache':str(cache),'cache_index_sha256':pin(cache/'index.json')['sha256'],'cache_admission':str(admission),'cache_admission_sha256':pin(admission)['sha256'],'initialization_receipt':str(out/'run/initialization.json'),'cohort_id':out.name}
  validate(options);SessionConfig(NAME,out.name,**options)
  for key,value in [('ema_decay',.9),('batch',2),('steps',128),('g_lr',.0003),('binding_sha256','0'*64),('cache_index_sha256','0'*64)]:
   bad=copy.deepcopy(options);bad[key]=value
   try:validate(bad)
   except (AssertionError,ValueError):negative_checks+=1
   else:raise AssertionError('Invalid configuration accepted:'+key)
  out.mkdir();(out/'run').mkdir()
  plan={'schema':'behavior-expanded-original-PG1024-batch8-fullanswers-v2','time':time.time(),'gpu':gpu,'physical_gpu_uuid':['GPU-72c1b506-891d-b8bc-b353-e020585e1c47','GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'][gpu],'cohort':out.name,'session_options':options,'original_FULL0':pin(original),'development':dev,'checkpoints':[256,512,768,1024],'development_strengths':[.5,1.,2.],'ema_decay':None,'elapsed_work_deadline':None,'initialization':'Fresh scoped original initialize at0; preserve pretrained and zero-UP; no SFT/oldadapter warmstart; restore never initializes','training_rows':138,'unique_training_questions':37,'training_weighting':index['training_weighting'],'target_protocol':PROTOCOL,'output_cap':4096,'effect_scoring_requires_native_EOS':True,'mining_cap_separately':128,'selection_metric':'equal-case mean full finished Jev effect ON minus matched OFF only; keep all maximum ties','diagnostics_not_selectors':['sympathy','sycophancy'],'catastrophic_health':'Qualified full-text NanoGPTJev Noul v3;4fixedDEV alpha1 nativeEOS probes before further quarter; uncertain=unmeasured, weak effect is not failure','holdout_access':False,'old_behavior_holdout32_already_spent':True,'old64token_results_separate':True,'controls_GPU0_first':gpu==0,'reuse_native_proof':None if gpu==0 else str(g0/'run'),'sources':sources+runtime,'inputs':inputs}
  write_json(out/'plan.json',plan);declarations.append({'gpu':gpu,'g_lr':rate,'plan':pin(out/'plan.json'),'folder':str(out)})
 write_json(folder/'training-CPU-qualification-first.json',{'time':time.time(),'state':'pass-original-formula-new138data-h1024-EOS-validation','sources':sources+runtime,'negative_config_refusals':negative_checks,'same_native_hidden_response_generated_and_initializer_functions':True,'original_critic16_SiLU_response2048_condition2052':True,'EMA_disabled':True,'new_TinyAya_calls':0,'native0_proof_pending':True})
 write_json(folder/'training-declaration-first.json',{'time':time.time(),'cohorts':declarations,'CPU_qualification':pin(folder/'training-CPU-qualification-first.json'),'declared_before_first_update':True,'both_GPUs':True,'formula':'Unchanged ConditionalPG09 Gaussian sigma1; real epsilon; fake epsilon+(generated-target)/1; official optimizer laws','new_data':138,'fresh_from':0,'total_updates_each':1024,'quarters':[256,512,768,1024],'full_answer_output_cap':4096,'effect_requires_EOS':True,'development_only_selection':True,'all24_candidate_cells_required':True,'old_holdout32_spent_no_untouched_claim':True})
 print(json.dumps({'declared':declarations,'CPU_negative_refusals':negative_checks,'training_updates':0,'native_proof_pending':True}))
if __name__=='__main__':main()
