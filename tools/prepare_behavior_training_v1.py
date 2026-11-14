"""Declare two finite original-PG behavior cohorts after cached-target admission."""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def pin(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,default=ROOT/'artifacts/behavior-dial-20261114-v1');args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from evidence_dial.io import write_json
    from evidence_dial.behavior_binding_v1 import validate,Native,initialize,PROTOCOL
    from evidence_dial.binding import Native as OriginalNative,initialize as OriginalInitialize
    from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter,AudioResponseCritic
    from particle_sliders.pg09 import PG_VERSION
    from particle_sliders.lifecycle import SessionConfig
    from evidence_dial.protocol import SEED
    folder=args.folder.resolve();cache=folder/'cache';admission=folder/'independent-cache-audit-first.json'
    a=json.loads(admission.read_text());assert a['state']=='all1024-saved-native-and-Jev-firsts-readmitted'
    assert a['native_generations_verified']==1024 and a['all_own_processes_absent_and_leases_released']
    assert a['cache_index']==pin(cache/'index.json') and a['responses']==pin(cache/'responses.pt')
    index=json.loads((cache/'index.json').read_text());assert 32<=index['cases']<=64
    manifest=json.loads((folder/'data/manifest-first.json').read_text());data=manifest['files']['train'];dev=manifest['files']['dev']
    assert pin(data['path'])==data and pin(dev['path'])==dev
    assert Native._hidden is OriginalNative._hidden and Native._response is ARLastHiddenResponseAdapter._response and Native.generated is ARLastHiddenResponseAdapter.generated
    assert initialize is OriginalInitialize and PG_VERSION=='0.9.5'
    critic=AudioResponseCritic(2048,2052)
    assert tuple(critic.net[0].weight.shape)==(16,4100) and type(critic.net[1]).__name__=='SiLU' and tuple(critic.net[2].weight.shape)==(1,16)
    evaluator_gate=folder/'CPU-AuditKIT-evaluator-qualification-first.json';e=json.loads(evaluator_gate.read_text())
    assert e['generation_source_sha256']==pin(ROOT/'src/evidence_dial/behavior_generate_v1.py')['sha256']
    assert e['actual_AuditKIT_inferences']==4 and e['new_native_model_calls']==e['new_Jev_calls']==0
    original=ROOT/'artifacts/first-answer-control-20261111-v1/runs/first-answer/FULL-000000.pt'
    names=['tools/prepare_behavior_training_v1.py','tools/behavior_phase_worker_v1.py','tools/run_behavior_training_v1.py',
        'tools/merge_behavior_cache_v1.py','src/evidence_dial/behavior_binding_v1.py','src/evidence_dial/behavior_native_proof_v1.py',
        'src/evidence_dial/behavior_generate_v1.py','src/evidence_dial/behavior_protocol_v1.py','src/evidence_dial/binding.py',
        'src/evidence_dial/first_answer_binding_v2.py','src/evidence_dial/evaluate.py','src/evidence_dial/shared_gpu_resources_v1.py',
        'src/evidence_dial/continuation_resources_v1.py','src/evidence_dial/qualified_text_guard_admission_v1.py',
        'tools/check_native_text_guard_gpu0_shared_v1.py','tools/check_native_text_guard_gpu1_continued_v1.py']
    sources=[pin(ROOT/n) for n in names]
    runtime=[pin(importlib.import_module(n).__file__) for n in ['particle_sliders.pg09','particle_sliders.pg09_audio','particle_sliders.lifecycle',
        'particle_sliders.inference','particle_sliders.routed_projection','particle_sliders.initialization']]
    root=Path('/mnt/nvme/lexica-evidence-dial-work/01a1c84b-5d17-7e32-80ef-34d8e4ee2e62')
    assert shutil.disk_usage(root).free>12*1024**3,'Insufficient observed space for own checkpoints; preserve other work'
    g0=root/'behavior-sympathy-h128-GPU0-g001-v1';g1=root/'behavior-sympathy-h128-GPU1-g00025-v1'
    assert not g0.exists() and not g1.exists() and not (folder/'training-declaration-first.json').exists()
    inputs=[data,dev,pin(admission),pin(cache/'index.json'),pin(cache/'responses.pt'),pin(original),
        pin(folder/'data/manifest-first.json'),pin(folder/'Jev-fixture-qualification-first.json'),pin(evaluator_gate),
        pin(ROOT/'artifacts/native-text-qualification-gpu0-v1/observer-qualification.json'),
        pin(ROOT/'artifacts/observer-native-text-v1e/qualification.json'),
        pin(ROOT/'artifacts/native-text-qualification-gpu0-v1/launch-context.json'),pin(ROOT/'configs/qwen-launch-context.json')]
    declarations=[];negative_checks=0
    for gpu,rate,out in [(0,.001,g0),(1,.00025,g1)]:
        options={'mode':'first-answer','steps':128,'batch':8,'g_lr':rate,'d_lr':.018,'bank_lr':.03,'ema_decay':None,
            'base_precision':'bfloat16','seed':SEED,'binding_sha256':pin(ROOT/'src/evidence_dial/behavior_binding_v1.py')['sha256'],
            'training_data':data['path'],'training_data_sha256':data['sha256'],'cache':str(cache),
            'cache_index_sha256':pin(cache/'index.json')['sha256'],'cache_admission':str(admission),
            'cache_admission_sha256':pin(admission)['sha256'],'initialization_receipt':str(out/'run/initialization.json'),
            'cohort_id':out.name}
        validate(options);SessionConfig('tiny-aya-behavior-MetaGlimmer-v1',out.name,**options)
        for key,value in [('ema_decay',.9),('batch',2),('steps',256),('g_lr',.0003),('binding_sha256','0'*64),('cache_index_sha256','0'*64)]:
            bad=copy.deepcopy(options);bad[key]=value
            try:validate(bad)
            except (AssertionError,ValueError):negative_checks+=1
            else:raise AssertionError('Invalid configuration accepted: '+key)
        out.mkdir();(out/'run').mkdir()
        plan={'schema':'behavior-original-PG128-batch8-v1','time':time.time(),'gpu':gpu,
            'physical_gpu_uuid':['GPU-72c1b506-891d-b8bc-b353-e020585e1c47','GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'][gpu],
            'cohort':out.name,'session_options':options,'original_FULL0':pin(original),'development':dev,
            'checkpoints':[32,64,96,128],'development_strengths':[.5,1.,2.],'ema_decay':None,'elapsed_work_deadline':None,
            'initialization':'Scoped original binding.initialize only for fresh state; never on restore',
            'training_rows':index['cases'],'target_protocol':PROTOCOL,'output_cap':64,'mining_cap_separately':128,
            'selection_metric':'equal-case mean finished Jev effect ON minus matched OFF only; keep all maximum ties',
            'diagnostics_not_selectors':['sympathy','sycophancy'],'catastrophic_health':'Existing qualified native64 prose observer; exact FULL-bound four fixed DEV probes before further updates or promotion',
            'holdout_access':False,'reuse_native_proof':None if gpu==0 else str(g0/'run'),
            'sources':sources+runtime,'inputs':inputs}
        write_json(out/'plan.json',plan);declarations.append({'gpu':gpu,'g_lr':rate,'plan':pin(out/'plan.json'),'folder':str(out)})
    gate={'time':time.time(),'state':'pass-original-PG-native-methods-config-and-AuditKIT-CPU-admission','sources':sources+runtime,
        'negative_config_refusals':negative_checks,'same_native_hidden_response_generated_and_initializer_functions':True,
        'critic':'original AudioResponseCritic16-SiLU, response2048/condition2052','weight_EMA':False,
        'new_native_model_calls':0,'new_Jev_calls':0,'training_updates':0,'native_first_proof_and_training_still_required':True}
    write_json(folder/'training-CPU-qualification-first.json',gate)
    write_json(folder/'training-declaration-first.json',{'time':time.time(),'cohorts':declarations,'CPU_qualification':pin(folder/'training-CPU-qualification-first.json'),
        'declared_before_first_update':True,'both_GPUs':True,'formula':'Unchanged ConditionalPG09 sigma1; real epsilon; fake epsilon+(generated-target)/1; official optimizer laws',
        'development_only_selection':True,'all24_candidate_cells_required':True,'behavior_holdout32_still_sealed':True})
    print(json.dumps({'declared':declarations,'CPU_negative_refusals':negative_checks,'training_updates':0,'native_proof_pending':True}))


if __name__=='__main__':main()
