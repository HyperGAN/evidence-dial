"""Metadata-only exact milestone/exit waiter, including future directories.

Linux pidfds report exact owned process exits. Inotify reports only declared
milestone/completion/failure file events. No model work, counter polling, job
restarts or signals. Each invocation ends after its first substantive event.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import platform
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True)
    args=ap.parse_args()
    from evidence_dial.benchmark import file_sha
    from evidence_dial.io import write_json
    plan=args.plan.resolve()
    spec=json.loads(plan.read_text())
    for item in spec['inputs']+spec['sources']:
        assert file_sha(ROOT/item['path'])==item['sha256'],item['path']
    ready=plan.parent/'ready-first.json'
    event=plan.parent/'event-first.json'
    assert not ready.exists() and not event.exists()
    handles={}
    proc=[]
    for item in spec['owned_processes']:
        pid=item['pid']
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert any(item['required_argument'].encode() == x for x in cmd),item
        if hasattr(os,'pidfd_open'):
            fd=os.pidfd_open(pid)
        else:
            runtime=ctypes.CDLL(None,use_errno=True)
            if hasattr(runtime,'pidfd_open'):
                fd=runtime.pidfd_open(pid,0)
            else:
                assert platform.machine()=='x86_64','No guessed syscall number for another architecture'
                fd=runtime.syscall(434,pid,0)
            if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
        handles[fd]=item
        proc.append({'pid':pid,'label':item['label'],'pidfd_opened':True})
    libc=ctypes.CDLL(None,use_errno=True)
    notify=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if notify<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    watches={}
    installed=set()
    targets={}
    declared=[]
    for item in spec['directories']:
        folder=(ROOT/item['path']).resolve()
        base=folder
        while not base.exists():base=base.parent
        declared.append((folder,base,item))
        for name,kind in item['exact_files'].items():
            targets[folder/name]=kind
    def refresh_watches():
        # Only exact declared ancestor chains; never list checkpoints/counters.
        for folder,base,item in declared:
            current=base
            candidates=[current]
            for part in folder.relative_to(base).parts:
                current=current/part
                if not current.exists():break
                candidates.append(current)
            for current in candidates:
                if current in installed:continue
                wd=libc.inotify_add_watch(notify,os.fsencode(current),0x100|0x80|0x8)
                if wd<0:raise OSError(ctypes.get_errno(),str(current))
                watches[wd]=current
                installed.add(current)
    refresh_watches()
    write_json(ready,{'time':time.time(),'pid':os.getpid(),'owned_pidfds':proc,
        'routine_progress_schedule':spec['progress_schedule'],
        'intermediate_FULL_or_journal_reads':0,'model_calls':0,'signals':0,
        'monitoring':'pidfd+inotify; only exact declared quarter milestones, completion and catastrophe/exit events'})
    def finish(value):
        write_json(event,{**value,'time':time.time(),'new_model_calls':0,'job_signals':0})
        print(json.dumps({'event':str(event.relative_to(ROOT)),**value}),flush=True)
    # A declared target may have appeared during watcher installation. Check
    # only those exact files; never list, stat or read intermediate checkpoints.
    for path,kind in targets.items():
        if path.exists():
            finish({'kind':kind,'source':str(path),'already_present_at_admission':True})
            return
    while True:
        readable,_,_=select.select([notify,*handles],[],[])
        for fd in readable:
            if fd in handles:
                finish({'kind':'owned_process_exit','owned_process':handles[fd]})
                return
            raw=os.read(notify,1024*1024)
            offset=0
            while offset<len(raw):
                wd,mask,cookie,length=struct.unpack_from('iIII',raw,offset)
                offset+=16
                name=raw[offset:offset+length].split(b'\0',1)[0].decode()
                offset+=length
                if wd not in watches:continue
                folder=watches[wd]
                path=folder/name
                kind=targets.get(path)
                if name=='candidate-failed.json' or (name.startswith('candidate-confirmed-failed-') and name.endswith('.json')):
                    kind='qualified_failure_flag'
                if mask & 0x40000000: # An exact declared ancestor may now exist.
                    refresh_watches()
                    for target,target_kind in targets.items():
                        if target.exists():
                            finish({'kind':target_kind,'source':str(target)})
                            return
                # CREATE alone can precede an atomic payload commit. Scientific
                # files are only actionable after close or rename into place.
                if kind and mask&(0x8|0x80):
                    finish({'kind':kind,'source':str((folder/name).resolve())})
                    return


if __name__=='__main__':
    main()
