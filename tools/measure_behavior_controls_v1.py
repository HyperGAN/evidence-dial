"""Frozen OFF and explicit prompt-only controls on behavior DEV, no training."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);args=ap.parse_args()
    from evidence_dial.io import read_jsonl,write_json
    from evidence_dial.benchmark import file_sha
    from evidence_dial.behavior_protocol_v1 import TARGETS
    plan=json.loads(args.plan.read_text());folder=args.plan.resolve().parent
    assert plan['schema']=='behavior-DEV-OFF-prompt-only-controls-v1' and plan['ema_decay'] is None
    for p in plan['inputs']+plan['sources']:assert file_sha(p['path'])==p['sha256']
    rows=read_jsonl(plan['data']['path']);assert len(rows)==16 and all(r['split']=='dev' for r in rows)
    assert not (folder/'completion-first.json').exists()
    parent=os.getppid();assert ctypes.CDLL(None).prctl(1,signal.SIGTERM)==0 and os.getppid()==parent
    def stop(sig,frame):raise KeyboardInterrupt('Owned behavior controls interrupted; preserve firsts')
    signal.signal(signal.SIGTERM,stop)
    write_json(folder/'actual-owned-process-first.json',{'time':time.time(),'pid':os.getpid(),
        'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'argv':[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        'plan_sha256':file_sha(args.plan),'training_updates':0})
    os.environ['CUDA_VISIBLE_DEVICES']=plan['physical_gpu_uuid']
    from evidence_dial.shared_gpu_resources_v1 import SharedGPU0Resources
    from evidence_dial.evaluate import load_base,tokenizer
    from evidence_dial.behavior_generate_v1 import evaluate,MAX_NEW_TOKENS
    assert plan['gpu']==0 and plan['max_new_tokens']==MAX_NEW_TOKENS==64
    with SharedGPU0Resources(0,folder/'resources-first.json',plan=args.plan) as resource:
        model,tok=load_base(),tokenizer();assert not any('lora_' in n for n,_ in model.named_parameters())
        for arm,target in [('off',None),('prompt-only',TARGETS[0])]:
            out=folder/arm
            if not (out/'summary.json').exists():evaluate(rows,model,tok,out,resource,arm=arm,target=target)
        for p in plan['inputs']+plan['sources']:assert file_sha(p['path'])==p['sha256']
        write_json(folder/'completion-first.json',{'time':time.time(),'complete':True,'arms':['off','prompt-only'],'cases_each':16,
            'new_training_updates':0,'selection_performed':False,'behavior_holdout_access':False,'raw_base_no_adapters':True})
    print(json.dumps({'complete':True,'arms':2,'cases_each':16,'new_training_updates':0}))


if __name__=='__main__':main()
