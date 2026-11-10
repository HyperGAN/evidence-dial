"""Capture evaluation sources and execute one bounded, explicitly named attempt."""
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
    p.add_argument('--attempt',required=True)
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    args=p.parse_args()
    values=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    if not values or values[0] not in ['evaluate','dev-particle']:
        raise ValueError('One explicit evaluation phase required')
    gpu=int(values[values.index('--gpu')+1]) if '--gpu' in values else 1
    os.environ['CUDA_VISIBLE_DEVICES']=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
        '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    names=['protocol','io','resources','faith_diagnostics','evaluate','report','benchmark','guard_scope','holdout','cli']
    capture=root/'artifacts/execution-sources';capture.mkdir(parents=True,exist_ok=True)
    files={}
    for name in names+['binding']:
        data=(root/f'src/evidence_dial/{name}.py').read_bytes();sha=hashlib.sha256(data).hexdigest()
        dest=capture/f'{name}-{sha}.py'
        if not dest.exists():
            with dest.open('xb') as f:f.write(data)
        assert dest.read_bytes()==data
        files[name]={'capture':str(dest),'sha256':sha}
    key=hashlib.sha256(json.dumps({'arguments':values,'attempt':args.attempt,'sources':files},sort_keys=True).encode()).hexdigest()
    receipt=root/f'artifacts/execution-receipts/eval-{key}.json'
    if receipt.exists():raise ValueError('Attempt already exists; inspect terminal state before a new named retry')
    started=time.time()
    write_json(receipt,{'pid':os.getpid(),'started':started,'arguments':values,'attempt':args.attempt,
                        'sources':files,'physical_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']})
    # Binding stays at its original source path because original native-v2
    # checkpoints deliberately bind its exact source/runtime identity.
    for name in names:
        module_name='evidence_dial.'+name
        spec=importlib.util.spec_from_file_location(module_name,files[name]['capture'])
        module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
    sys.argv=['evidence-dial',*values]
    try:
        sys.modules['evidence_dial.cli'].main()
    except BaseException as error:
        write_json(receipt.with_suffix('.terminal.json'),{'state':'exception','error':repr(error),'seconds':time.time()-started})
        raise
    write_json(receipt.with_suffix('.terminal.json'),{'state':'complete','seconds':time.time()-started})


if __name__=='__main__':main()
