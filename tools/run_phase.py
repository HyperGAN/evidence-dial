"""Capture an immutable execution derivative before a bounded GPU phase."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from evidence_dial.io import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['train','fresh','native-check','recover-outputs'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stop',type=int)
    p.add_argument('--rate',type=float)
    p.add_argument('--reference',type=Path)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--attempt',default='first',help='Stable ID for this bounded execution; new ID only after a verified terminal attempt')
    args=p.parse_args()
    uuid=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    capture=root/'artifacts/execution-sources'
    capture.mkdir(parents=True,exist_ok=True)
    files={}
    for name in ['train','binding','evaluate','faith_diagnostics','protocol','resources','io','guard_scope','benchmark']:
        source=root/f'src/evidence_dial/{name}.py'
        content=source.read_bytes();sha=hashlib.sha256(content).hexdigest()
        dest=capture/f'{name}-{sha}.py'
        if dest.exists():
            assert dest.read_bytes()==content
        else:
            with dest.open('xb') as f:f.write(content)
        files[name]={'original':str(source),'capture':str(dest),'sha256':sha}
    launcher=Path(__file__).read_bytes()
    launcher_sha=hashlib.sha256(launcher).hexdigest()
    token=hashlib.sha256(json.dumps({'phase':args.phase,'run':str(args.run.resolve()),'stop':args.stop,'sources':files,'attempt':args.attempt,'launcher_sha256':launcher_sha},sort_keys=True).encode()).hexdigest()
    receipt=root/f'artifacts/execution-receipts/{token}.json'
    if receipt.exists():
        raise ValueError('Execution already attempted; inspect/reap it, never blindly restart')
    started=time.time()
    write_json(receipt,{'state':'started','pid':os.getpid(),'gpu':args.gpu,'uuid':uuid,'phase':args.phase,'run':str(args.run.resolve()),'stop':args.stop,'started':started,'sources':files,'attempt':args.attempt,'launcher_sha256':launcher_sha})
    name='evidence_dial.captured_train_'+files['train']['sha256'][:12]
    spec=importlib.util.spec_from_file_location(name,files['train']['capture'])
    module=importlib.util.module_from_spec(spec);module.__package__='evidence_dial';sys.modules[name]=module;spec.loader.exec_module(module)
    try:
        if args.phase=='train':
            module.train_chunk(args.run,args.stop,root/'artifacts/data/policy-v1',args.gpu)
        elif args.phase=='fresh':
            module.fresh_arm(args.run,root/'artifacts/cache/teacher-v1',args.rate,args.gpu,args.reference)
        elif args.phase=='native-check':
            module.native_check(args.run,root/'artifacts/cache/teacher-v1',root/'artifacts/data/policy-v1',args.rate,args.gpu)
        else:
            module.recover_outputs(args.run,args.gpu)
    except BaseException as error:
        write_json(receipt.with_suffix('.terminal.json'),{'state':'failed','error':repr(error),'seconds':time.time()-started,'time':time.time()})
        raise
    write_json(receipt.with_suffix('.terminal.json'),{'state':'complete','seconds':time.time()-started,'time':time.time()})


if __name__=='__main__':main()
