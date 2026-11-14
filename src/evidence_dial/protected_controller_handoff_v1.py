"""Exact-owned serial-controller handoff; never signal a model child.

The admitted child finishes naturally while only its controller is stopped.
Any uncertain phase/exit refuses handoff and resumes the original controller.
No training counters are read to choose a boundary: the already-running child's
declared --stop supplies the boundary. An observation timeout is not completion.
"""
import contextlib
import ctypes
import fcntl
import json
import os
from pathlib import Path
import platform
import select
import signal
import time

from .benchmark import file_sha
from .io import write_json


def proc(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'state':fields[0],'parent':int(fields[1]),
        'birth':fields[19],'raw_exit_status':int(fields[49]),
        'argv':[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]}


def pidfd(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    libc=ctypes.CDLL(None,use_errno=True)
    if hasattr(libc,'pidfd_open'):fd=libc.pidfd_open(pid,0)
    else:
        assert platform.machine()=='x86_64'
        fd=libc.syscall(434,pid,0)
    if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
    return fd


def exited(fd):
    return bool(select.select([fd],[],[],0)[0])


def same_process(identity,*,allow_zombie=False):
    actual=proc(identity['pid'])
    assert actual['birth']==identity['birth'],'PID identity changed'
    if not allow_zombie:
        assert actual['argv']==identity['argv'] and actual['state']!='Z','Process argv/state changed'
    return actual


def children(pid):
    return [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]


def option(argv,name):
    assert argv.count(name)==1,(name,argv)
    return argv[argv.index(name)+1]


def admitted_child(plan):
    """Bind only one recognized native phase with an actual admitted receipt."""
    root=same_process(plan['controller'])
    ids=children(root['pid'])
    assert len(ids)==1,'Require the same single direct phase child'
    child=proc(ids[0]);argv=child['argv']
    assert child['parent']==root['pid'] and child['state']!='Z'
    assert plan['phase_tool'] in argv,'Current child is not the declared phase tool'
    recipe=Path(option(argv,'--recipe')).resolve()
    assert str(recipe)==str(Path(plan['recipe']).resolve())
    assert option(argv,'--mode')=='first-answer'
    phase=option(argv,'--phase');assert phase in ['train','evaluate']
    stop=int(option(argv,'--stop'));assert stop in plan['declared_phase_stops']
    attempt=option(argv,'--attempt');assert '/' not in attempt and '\\' not in attempt
    run=recipe.parent/'runs/first-answer'
    resource_path=run/(f'eval-resources-{attempt}.json' if phase=='evaluate' else f'{phase}-{stop}-{attempt}-resources.json')
    admission=json.loads(resource_path.read_text())
    assert admission['state']=='admitted' and admission['pid']==child['pid']
    assert str(admission['pid_birth_ticks'])==child['birth'] and admission['parent_pid']==root['pid']
    return {'identity':child,'phase':phase,'stop':stop,'attempt':attempt,
        'resource_path':str(resource_path),'resource_sha256':file_sha(resource_path),
        'driver_terminal':str(recipe.parent/'executions'/f'first-answer-{phase}-{stop}-{attempt}.terminal.json'),
        'controller_terminal':str(recipe.parent/'executions'/f'{attempt}.terminal.json')}


def perform(plan,out):
    """Called by the same tested guardian for fixtures and actual controllers."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=out/'handoff-first.json';assert not result.exists()
    paused=False;terminated=False;root=plan['controller']
    root_fd=child_fd=None
    old_handlers={}
    def interrupted(sig,frame):raise KeyboardInterrupt(f'Guardian signal{sig}; resume exact old controller')
    for sig in [signal.SIGTERM,signal.SIGINT]:
        old_handlers[sig]=signal.signal(sig,interrupted)
    try:
        for item in plan['inputs']+plan['sources']:
            assert file_sha(item['path'])==item['sha256'],item['path']
        assert not (out/'paused-controller-first.json').exists(),'Use explicit recovery for an attempted handoff'
        root_fd=pidfd(root['pid']);same_process(root)
        phase=admitted_child(plan)
        child=phase['identity'];child_fd=pidfd(child['pid'])
        same_process(child);assert not exited(root_fd) and not exited(child_fd)
        write_json(out/'bound-first.json',{'time':time.time(),'controller':root,'phase':phase})
        # Persist the recovery binding BEFORE pausing; a hard guardian failure
        # leaves enough identity to resume this exact controller explicitly.
        write_json(out/'pause-intent-first.json',{'time':time.time(),'controller':root,'phase':phase,'guardian_pid':os.getpid()})
        same_process(root);os.kill(root['pid'],signal.SIGSTOP);paused=True
        for _ in range(100):
            state=same_process(root)['state']
            if state=='T':break
            time.sleep(.01)
        assert state=='T','Controller did not stop'
        assert children(root['pid'])==[child['pid']]
        same_process(child);assert not exited(child_fd),'Child finished during pause race; defer'
        write_json(out/'paused-controller-first.json',{'time':time.time(),'controller':root,'phase':phase,'model_child_signals':0})
        with open(plan['candidate_lock'],'a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            # The candidate context may finish before the driver writes its
            # terminal receipt. pidfd waits for actual exit without a deadline.
            while not exited(child_fd):select.select([child_fd],[],[])
            dead=same_process(child,allow_zombie=True)
            assert dead['state']=='Z' and dead['parent']==root['pid']
            assert dead['raw_exit_status']==0,'Original model phase did not exit0; no handoff/replay'
            terminal=Path(phase['driver_terminal'])
            assert json.loads(terminal.read_text())['state']=='complete','Driver completion is uncertain'
            run=Path(plan['recipe']).parent/'runs/first-answer'
            paths=[run/'FULL-000000.pt',run/f'FULL-{phase["stop"]:06d}.pt',
                run/f'RAW-{phase["stop"]:06d}.pt',run/f'NATIVE-{phase["stop"]:06d}.pt',
                run/f'probes-{phase["stop"]:06d}.json',run/'session.json',
                run/'initialization.json',run/'updates.jsonl',terminal]
            pins=[{'path':str(p),'sha256':file_sha(p)} for p in paths]
            write_json(out/'child-complete-prefix-first.json',{'time':time.time(),'phase':phase,
                'actual_child_raw_exit_status':dead['raw_exit_status'],'pins':pins})
            same_process(root);assert same_process(root)['state']=='T'
            # Child is already a verified exited zombie; the controller's
            # finally can reap it without ever signaling a live model child.
            os.kill(root['pid'],signal.SIGTERM);terminated=True
            os.kill(root['pid'],signal.SIGCONT);paused=False
            while not exited(root_fd):select.select([root_fd],[],[])
            reaped=json.loads(Path(phase['controller_terminal']).read_text())
            assert reaped['reaped'] and reaped['returncode']==0,'Original child was not reaped0'
            for item in pins:assert file_sha(item['path'])==item['sha256'],item['path']
            value={'time':time.time(),'state':'old-controller-exited-child0-prefix-preserved',
                'controller':root,'phase':phase,'pins':pins,
                'child_naturally_exited0_and_original_controller_reaped0':True,
                'old_controller_pidfd_exited':True,'candidate_EX_held_at_boundary':True,
                'model_child_signals':0,'foreign_signals':0,'new_model_calls':0,
                'new_optimizer_updates':0,'original_exec_handle_still_requires_collection':True}
            write_json(result,value)
            return value
    except BaseException as error:
        write_json(out/'refused-first.json',{'time':time.time(),'error':repr(error),'controller_terminated':terminated,
            'model_child_signals':0,'new_model_calls':0})
        raise
    finally:
        if paused and root_fd is not None and not exited(root_fd):
            same_process(root);os.kill(root['pid'],signal.SIGCONT)
            write_json(out/'resumed-old-controller-first.json',{'time':time.time(),'controller':root,'handoff_committed':False})
        for fd in [child_fd,root_fd]:
            if fd is not None:os.close(fd)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
