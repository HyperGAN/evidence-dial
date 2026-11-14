"""Own one bounded PG cohort; checkpoint, qualified health, then fixed DEV grid."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);args=ap.parse_args()
    from evidence_dial.io import write_json
    from evidence_dial.benchmark import file_sha
    from evidence_dial.qualified_text_guard_admission_v1 import passing_guard
    plan=json.loads(args.plan.read_text());folder=args.plan.resolve().parent;out=folder/'run'
    for p in plan['inputs']+plan['sources']:assert file_sha(p['path'])==p['sha256']
    assert plan['session_options']['steps']==128 and plan['checkpoints']==[32,64,96,128]
    lock=(folder/'.controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    first=folder/'controller-first.json';assert not first.exists()
    parent=os.getppid();assert ctypes.CDLL(None).prctl(1,signal.SIGTERM)==0 and os.getppid()==parent
    def stop(sig,frame):raise KeyboardInterrupt('Owned behavior controller interrupted')
    signal.signal(signal.SIGTERM,stop)
    write_json(first,{'time':time.time(),'pid':os.getpid(),'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'argv':[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],'plan_sha256':file_sha(args.plan)})
    executions=folder/'controller-executions';counter=0
    def child(command,env=None):
        nonlocal counter
        counter+=1;receipt=executions/f'phase-{counter:03d}.json';assert not receipt.exists()
        write_json(receipt,{'time':time.time(),'command':command,'env_names':sorted((env or {}).keys())})
        process=None
        with receipt.with_suffix('.log').open('x') as log:
            try:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1',**(env or {})})
                write_json(receipt,{'time':time.time(),'pid':process.pid,'command':command})
                code=process.wait()
            finally:
                if process is not None:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    write_json(receipt.with_suffix('.terminal.json'),{'time':time.time(),'returncode':process.returncode,'reaped':True})
        assert code==0,f'Owned phase failed{code}; preserve completed FULL and firsts, review before any new execution'
    def phase(name,step):
        child([str(ROOT/'.venv/bin/python'),'tools/behavior_phase_worker_v1.py','--plan',str(args.plan.resolve()),'--phase',name,'--stop',str(step),'--attempt','first'])
    try:
        if plan.get('reuse_native_proof'):
            parent_proof=Path(plan['reuse_native_proof'])/'native-check.json'
            while not parent_proof.exists():
                if (Path(plan['reuse_native_proof']).parent/'controller-failed.json').exists():raise RuntimeError('Reference native proof failed; do not wait indefinitely or fake reuse')
                time.sleep(2)
        phase('initialize',0)
        for quarter,step in enumerate(plan['checkpoints'],1):
            phase('train',step)
            context=json.loads((ROOT/('artifacts/native-text-qualification-gpu0-v1/launch-context.json' if plan['gpu']==0 else 'configs/qwen-launch-context.json')).read_text())
            observer='/ml2/particlesliders-working/lr-work/output-effect-scoring/ming-layer1-qwen3vl8b-cuda-env-v1/bin/python'
            script='tools/check_native_text_guard_gpu0_shared_v1.py' if plan['gpu']==0 else 'tools/check_native_text_guard_gpu1_continued_v1.py'
            child([observer,script,'--candidate',str(out),'--step',str(step),'--checkpoint',str(out/f'FULL-{step:06d}.pt'),'--attempt','first'],context)
            assert passing_guard(out,step,out/f'FULL-{step:06d}.pt') is not None
            phase('evaluate',step)
            write_json(folder/f'quarter-{quarter:02d}-finished-first.json',{'time':time.time(),'step':step,'budget':128,
                'finished_DEV_strengths':plan['development_strengths'],'partial_scores_reported':False})
        for p in plan['inputs']+plan['sources']:assert file_sha(p['path'])==p['sha256']
        write_json(folder/'completion-first.json',{'time':time.time(),'complete':True,'steps':128,'all4quarters_and3strengths_finished':True,
            'weight_EMA':False,'selection_performed':False,'holdout_access':False})
        print(json.dumps({'complete':True,'steps':128,'gpu':plan['gpu']}))
    except BaseException as error:
        write_json(folder/'controller-failed.json',{'time':time.time(),'error':repr(error),'completed_work_preserved':True});raise


if __name__=='__main__':main()
