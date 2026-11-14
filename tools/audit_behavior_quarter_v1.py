"""Independently admit native0 and declared complete behavior DEV quarters."""
import argparse
from itertools import islice
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def pin(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def native0(plan,folder):
    from particle_sliders.lifecycle import read_packet
    from particle_sliders.inference import read_inference
    from evidence_dial.train import tensor_equal
    from evidence_dial.behavior_native_proof_v1 import compare_original_initial
    out=folder/'run';full=read_packet(out/'FULL-000000.pt');raw=read_packet(out/'RAW-000000.pt');artifact=read_inference(out/'NATIVE-000000.pt')
    zero=full['payload']['game'];assert zero['completed_steps']==0 and full['payload']['ema']=={'decay':None,'values':None}
    allowed=compare_original_initial(read_packet(plan['original_FULL0']['path'])['payload']['game'],zero)
    assert raw['payload']['selection']==artifact['selection']=='raw'
    assert tensor_equal(zero['network'],raw['payload']['adapter']['network']) and tensor_equal(zero['network'],artifact['network'])
    proof=json.loads((out/'native-check.json').read_text())
    assert proof['initial_ON_equals_OFF'] and proof['OFF_matches_new_cache'] and proof['positive_sample_hidden_matches_direct_frozen_hidden']
    assert proof['connected_tensors']==2593 and proof['exact_FULL_restore'] and proof['native_inference_factory_reload_matches_raw_zero'] and proof['no_weight_EMA']
    assert proof['current_FULL0_sha256']==pin(out/'FULL-000000.pt')['sha256']
    # A reused zero/gradient proof has no new operations inside run(); the
    # worker separately performs one actual new inference-factory reload.
    operations={'native_zero_and_gradient_response_operations':0 if plan.get('reuse_native_proof') else 4,
                'separate_inference_factory_reload_forward':1,'new_training_updates_in_native_check':0}
    return {'proof':pin(out/'native-check.json'),'FULL0':pin(out/'FULL-000000.pt'),'RAW0':pin(out/'RAW-000000.pt'),'NATIVE0':pin(out/'NATIVE-000000.pt'),
        'initial_network_critic_prior_match_original':True,'allowed_recipe_differences':allowed,'operation_scope_clarification':operations}


def audit(plan_path,step):
    from particle_sliders.lifecycle import read_packet
    from particle_sliders.inference import read_inference
    from evidence_dial.train import tensor_equal
    from evidence_dial.qualified_text_guard_admission_v1 import passing_guard
    from evidence_dial.behavior_protocol_v1 import messages,QUESTIONS,JEV,ENDPOINT
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.protocol import digest
    from evidence_dial.io import read_jsonl
    import torch
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text());folder=plan_path.parent;out=folder/'run'
    for p in plan['sources']+plan['inputs']:assert pin(p['path'])==p
    initial=native0(plan,folder)
    if step==0:return {'state':'native-zero-independently-readmitted','native0':initial,'new_model_calls':0,'training_updates':0}
    assert step in [32,64,96,128]
    marker=json.loads((folder/f'quarter-{step//32:02d}-finished-first.json').read_text());assert marker['step']==step and marker['budget']==128 and marker['finished_DEV_strengths']==[.5,1.,2.]
    full,raw,artifact=[out/f'{k}-{step:06d}.pt' for k in ['FULL','RAW','NATIVE']]
    packet,exported,native=read_packet(full),read_packet(raw),read_inference(artifact);game=packet['payload']['game'];zero=read_packet(out/'FULL-000000.pt')['payload']['game']
    assert game['completed_steps']==exported['payload']['adapter']['completed_steps']==native['origin']['completed_steps']==step
    assert game['config']==zero['config'] and game['config']['recipe']['ema_decay']==0.
    assert packet['payload']['ema']=={'decay':None,'values':None} and exported['payload']['selection']==native['selection']=='raw'
    assert tensor_equal(game['network'],exported['payload']['adapter']['network']) and tensor_equal(game['network'],native['network'])
    assert all(torch.isfinite(v).all() for c in ['network','critic','prior'] for v in game[c].values() if isinstance(v,torch.Tensor))
    rates={g['role']:g['lr'] for g in game['opt_g']['param_groups']+game['opt_d']['param_groups']}
    for role,want in [('generator',plan['session_options']['g_lr']),('critic',.018),('prior',.03)]:assert abs(rates[role]-want)<1e-14
    for k in ['opt_g','opt_d','optimizer_history_schema','spread','streams','row_orders','row_cursors','exposures','model_cpu_rng','model_cuda_rng']:assert game[k] is not None
    with (out/'updates.jsonl').open() as f:journal=[json.loads(x) for x in islice(f,step)]
    assert [j['step'] for j in journal]==list(range(1,step+1))
    guard=passing_guard(out,step,full);assert guard is not None
    rows=read_jsonl(plan['development']['path']);assert len(rows)==16 and all(r['split']=='dev' for r in rows)
    base=ROOT/'artifacts/behavior-dial-20261114-v1/DEV-controls-GPU0-v1';control_audit=json.loads((base/'independent-completion-audit-first.json').read_text())
    assert control_audit['state']=='all32-native-and-NanoGPT-Jev-control-records-readmitted'
    off_path=Path(control_audit['metrics']['off']['predictions']['path']);assert pin(off_path)==control_audit['metrics']['off']['predictions']
    off=read_jsonl(off_path);assert [r['id'] for r in off]==[r['id'] for r in rows]
    tok=tokenizer();arms=[]
    for alpha in plan['development_strengths']:
        f=out/f'dev-step-{step:06d}-alpha-{alpha:g}';records=read_jsonl(f/'predictions.jsonl');assert [r['id'] for r in records]==[r['id'] for r in rows]
        label=f'particle-raw-{pin(artifact)["sha256"]}-strength-{alpha:g}'
        for row,r,b in zip(rows,records,off):
            v=json.loads((f/'native'/f'{row["id"]}.json').read_text());expected=tok.apply_chat_template(messages(row['question']),add_generation_prompt=True)
            assert r['arm']==v['identity']['arm']==label and v['identity']['messages']==messages(row['question'])
            assert r['native']['input_ids']==v['native']['input_ids']==v['identity']['input_ids']==b['native']['input_ids']==expected
            ids=r['native']['output_ids'];assert ids==v['native']['output_ids'] and 1<=len(ids)<=64 and (ids[-1]==3 or len(ids)==64)
            assert r['output']==v['output']==tok.decode(ids,skip_special_tokens=True) and r['question']==row['question']
            request={'model':JEV,'state':{'user_message':row['question'],'response':r['output']},'questions':QUESTIONS}
            key=digest({'endpoint':ENDPOINT,'request':request});saved=json.loads((f/'jev-firsts'/(key+'.json')).read_text())
            assert saved['request']==request and saved['http_status']==200 and saved['response_sha256']==digest(saved['response']) and saved['response']['model']==JEV
            a=saved['response']['answers'];assert a['effect']['type']=='score' and a['sympathy']['type']==a['sycophancy']['type']=='noul'
            assert r['scores']['effect']==a['effect']['score'] and r['scores']['sympathy']==a['sympathy']['noul'] and r['scores']['sycophancy']==a['sycophancy']['noul']
            assert r['effect']==a['effect']['score']/4. and math.isfinite(r['effect']) and 0<=r['effect']<=1
        summary=json.loads((f/'summary.json').read_text());mean=sum(r['effect'] for r in records)/16;off_mean=sum(r['effect'] for r in off)/16
        assert summary['complete'] and summary['cases']==16 and summary['mean_effect']==mean and summary['arm']==label
        arms.append({'step':step,'strength':alpha,'cases':16,'mean_effect':mean,'OFF_mean_effect':off_mean,
            'effect_ON_minus_OFF':sum(r['effect']-b['effect'] for r,b in zip(records,off))/16,
            'mean_sympathy_probability':summary['mean_sympathy_probability'],'mean_sycophancy_probability':summary['mean_sycophancy_probability'],
            'predictions':pin(f/'predictions.jsonl'),'summary':pin(f/'summary.json'),'artifact':pin(artifact),'FULL':pin(full)})
    return {'state':'declared-complete-behavior-quarter-independently-readmitted','step':step,'budget':128,'plan':pin(plan_path),'native0':initial,
        'FULL':pin(full),'RAW':pin(raw),'NATIVE':pin(artifact),'qualified_checkpoint_guard':pin(guard),
        'all48_native_inputs_outputs_typed_Jev_scores_verified':True,'exact_declared_journal_prefix_verified':True,
        'RAW_FULL_NATIVE_network_equality':True,'original_config_and_optimizer_laws_preserved':True,'actual_optimizer_rates':rates,
        'no_weight_EMA':True,'arms':arms,'control_admission':pin(base/'independent-completion-audit-first.json'),
        'new_model_calls':0,'new_training_updates':0,'holdout_rows_read':0,'selection_performed':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--step',type=int,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists();os.environ['CUDA_VISIBLE_DEVICES']=''
    from evidence_dial.io import write_json
    result=audit(args.plan,args.step);write_json(args.output,{'time':time.time(),'source':pin(__file__),**result})
    print(json.dumps({'audit':pin(args.output),'step':args.step,'arms':result.get('arms',[]),'new_model_calls':0}))


if __name__=='__main__':main()
