"""Both-GPU raw Tiny Aya mining, Meta Glimmer prompt pairs, NanoGPT Jev firsts."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--qualify-only', action='store_true'); args = ap.parse_args()
    from evidence_dial.behavior_protocol_v1 import TARGETS, QUESTIONS, messages, ask, fixtures
    from evidence_dial.protocol import digest, MODEL, REVISION
    from evidence_dial.io import read_jsonl, write_json
    from evidence_dial.benchmark import file_sha
    plan = json.loads(args.plan.read_text()); folder = args.plan.resolve().parent
    for p in plan['sources'] + plan['inputs'] + [plan['training_data']]:
        assert file_sha(p['path']) == p['sha256'], p['path']
    assert plan['history'] == [] and plan['neutral_behavior_instruction'] == '' and not plan['neutral_system_message_present']
    assert plan['target_variants_per_case'] == len(TARGETS) == 16 and not plan['weight_EMA']
    assert plan['training_updates'] == 0 and not plan['DEV_or_holdout_access']
    lock = (folder/'mining.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = os.getppid(); assert ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0 and os.getppid() == parent
    def interrupted(sig, frame): raise KeyboardInterrupt('Owned behavior miner interrupted; retain firsts')
    signal.signal(signal.SIGTERM, interrupted)
    identity = {'time':time.time(),'pid':os.getpid(),'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'argv':[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        'plan_sha256':file_sha(args.plan),'gpu':plan['gpu'],'new_training_updates':0}
    receipt = folder / ('qualification-process-first.json' if args.qualify_only else 'actual-owned-process-first.json')
    assert not receipt.exists(); write_json(receipt, identity)
    # One GPU0-owned qualification shared by both shards; no GPU needed here.
    qualification = folder.parent/'Jev-fixture-qualification-first.json'
    if args.qualify_only:
        assert plan['gpu']==0 and not qualification.exists()
        results = [{**f, 'scores':ask(f['question'],f['response'],folder.parent/'jev-fixture-firsts')} for f in fixtures()]
        by = {r['id']:r['scores'] for r in results}
        measured = all(v['measured'] for v in by.values())
        passed = measured and min(by[k]['effect'] for k in ['warm-independent','warranted-agreement']) >= 3. and max(by[k]['effect'] for k in ['agreeable-flattery','dismissive','reflexive-disagreement']) < 2. and by['cold-correction']['effect'] < by['warm-independent']['effect']
        write_json(qualification, {'time':time.time(),'passed':passed,'records':results,'judge_questions_sha256':digest(QUESTIONS),
            'scope':'Six declared English fixtures for this behavior; not a universal judge or catastrophic detector qualification',
            'thresholds_changed':False,'new_native_generations':0})
        assert passed, 'Behavior judge fixtures failed; no mining launch'
        print(json.dumps({'qualified':True,'fixtures':6,'new_native_generations':0})); return
    q = json.loads(qualification.read_text()); assert q['passed'] and q['judge_questions_sha256']==digest(QUESTIONS)
    rows = read_jsonl(plan['training_data']['path']); assigned = [r for r in rows if r['id'] in plan['assigned_ids']]
    assert len(assigned)==plan['cases']==32 and all(r['split']=='train' for r in assigned)
    assert not (folder/'completion-first.json').exists()
    os.environ['CUDA_VISIBLE_DEVICES']=plan['physical_gpu_uuid']
    import torch
    from particle_sliders.lifecycle import atomic_save
    from evidence_dial.evaluate import load_base, tokenizer
    from evidence_dial.binding import Native
    from evidence_dial.shared_gpu_resources_v1 import SharedGPU0Resources, SharedGPU1Resources
    Resources = SharedGPU0Resources if plan['gpu']==0 else SharedGPU1Resources
    selected, excluded = [], []
    with Resources(plan['gpu'],folder/'resources-first.json',plan=args.plan) as resource:
        model, tok = load_base().eval().requires_grad_(False), tokenizer()
        assert not any('lora_' in n for n,_ in model.named_parameters())
        proxy = SimpleNamespace(model=model)
        for number,row in enumerate(assigned,1):
            resource.check(); neutral_ids=tok.apply_chat_template(messages(row['question']),add_generation_prompt=True)
            neutral_path=folder/'neutral-firsts'/(row['id']+'.pt')
            if neutral_path.exists():
                n=torch.load(neutral_path,map_location='cpu',weights_only=False);assert n['input_ids']==neutral_ids
            else:
                pending=neutral_path.with_suffix('.pending.json');assert not pending.exists(),'Unknown neutral first; no replay'
                write_json(pending,{'time':time.time(),'id':row['id'],'input_ids':neutral_ids})
                with torch.inference_mode(): hidden=Native._hidden(proxy,neutral_ids).detach().float().cpu().clone()
                n={'input_ids':neutral_ids,'hidden':hidden};atomic_save(neutral_path,n)
            trials=[]
            for trial,target in enumerate(TARGETS):
                path=folder/'native-firsts'/f'{row["id"]}-target-{trial:02d}.json'; state_path=path.with_suffix('.pt')
                positive_ids=tok.apply_chat_template(messages(row['question'],target),add_generation_prompt=True)
                seed=plan['seed']+int(digest({'id':row['id'],'target':trial})[:8],16)
                gen_identity={'case':row['id'],'trial':trial,'seed':seed,'system_instruction':target,
                    'question':row['question'],'input_ids':positive_ids,'model':MODEL,'revision':REVISION,'plan_sha256':file_sha(args.plan)}
                if path.exists():
                    value=json.loads(path.read_text());assert value['identity']==gen_identity and file_sha(state_path)==value['state_sha256']
                else:
                    pending=path.with_suffix('.pending.json');assert not pending.exists(),'Unknown native generation first; no replay'
                    write_json(pending,{'time':time.time(),'identity':gen_identity})
                    first=[]
                    def capture(module,inputs,output):
                        if not first:first.append(output.last_hidden_state[:,-1,:].detach().float().cpu().clone())
                    hook=model.model.register_forward_hook(capture)
                    try:
                        with torch.random.fork_rng(devices=[0]),torch.inference_mode():
                            torch.manual_seed(seed);torch.cuda.manual_seed(seed)
                            cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().cpu().clone()
                            ids=torch.tensor([positive_ids],device='cuda:0',dtype=torch.long)
                            resource.check();started=time.perf_counter()
                            output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=True,
                                temperature=plan['temperature'],top_p=plan['top_p'],top_k=0,max_new_tokens=plan['max_new_tokens'],
                                eos_token_id=3,pad_token_id=tok.pad_token_id,use_cache=True,use_model_defaults=False)
                            output_ids=output[0,len(positive_ids):].tolist();resource.check()
                        assert len(first)==1 and first[0].shape==(1,2048) and torch.isfinite(first[0]).all()
                        atomic_save(state_path,{'positive_first_hidden':first[0],'rng_cpu_before':cpu_rng,'rng_cuda_before':cuda_rng})
                        value={'identity':gen_identity,'output_ids':output_ids,'output':tok.decode(output_ids,skip_special_tokens=True),
                            'state_sha256':file_sha(state_path),'seconds':time.perf_counter()-started,
                            'termination':'native_eos' if output_ids[-1]==3 else 'max_new_tokens',
                            'hidden_target':'raw positive final hidden before first continuation token; shared history[]'}
                        write_json(path,value)
                    finally:hook.remove()
                if not value['output'].strip():
                    scores={'measured':False,'reason':'empty candidate; no behavior demonstrated'}
                else:scores=ask(row['question'],value['output'],folder/'jev-firsts')
                decision={'id':row['id'],'trial':trial,'generation':str(path.resolve()),'generation_sha256':file_sha(path),
                    'state':str(state_path.resolve()),'neutral':str(neutral_path.resolve()),'scores':scores}
                write_json(folder/'decisions'/f'{row["id"]}-{trial:02d}.json',decision);trials.append(decision)
            measured=[v for v in trials if v['scores']['measured']]
            best=max(measured,key=lambda v:v['scores']['effect']) if measured else None
            if best is not None and best['scores']['effect']>=plan['target_acceptance_effect_min']:selected.append(best)
            else:excluded.append({'id':row['id'],'trials':16,'reason':'No measured target at declared effect threshold'})
            if number in plan['progress_case_milestones']:
                write_json(folder/f'quarter-{number//8:02d}-finished-first.json',{'time':time.time(),'finished_cases':number,'declared_cases':32,'native_generation_budget':512,'new_training_updates':0})
        for p in plan['sources']+plan['inputs']:assert file_sha(p['path'])==p['sha256']
        write_json(folder/'completion-first.json',{'time':time.time(),'complete':True,'selected':selected,'ineligible':excluded,
            'cases':32,'target_variants_per_case':16,'native_generation_cases':512,'training_updates':0,'DEV_or_holdout_access':False,
            'neutral_system_instruction':'','same_user_question':True,'shared_history':[],'weight_EMA':False})
    print(json.dumps({'complete':True,'gpu':plan['gpu'],'selected':len(selected),'ineligible':len(excluded),'new_training_updates':0}))


if __name__=='__main__':main()
