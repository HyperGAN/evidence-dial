"""Finite behavior PG phases; exact FULL resumes and existing native64 guard."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--phase',choices=['initialize','train','evaluate'],required=True)
    ap.add_argument('--stop',type=int,required=True);ap.add_argument('--attempt',required=True)
    args=ap.parse_args();plan=json.loads(args.plan.read_text());folder=args.plan.resolve().parent;out=folder/'run'
    from evidence_dial.benchmark import file_sha
    from evidence_dial.io import read_jsonl,write_json
    for p in plan['inputs']+plan['sources']:assert file_sha(p['path'])==p['sha256'],p['path']
    assert plan['schema']=='behavior-original-PG128-batch8-v1' and plan['ema_decay'] is None
    assert args.stop in ([0] if args.phase=='initialize' else [32,64,96,128])
    assert not (out/'candidate-failed.json').exists() and not list(out.glob('candidate-confirmed-failed-*.json'))
    receipt=folder/'executions'/f'{args.phase}-{args.stop:06d}-{args.attempt}.json';assert not receipt.exists()
    parent=os.getppid();assert ctypes.CDLL(None).prctl(1,signal.SIGTERM)==0 and os.getppid()==parent
    def stop(sig,frame):raise KeyboardInterrupt('Owned behavior phase interrupted; preserve FULL and native firsts')
    signal.signal(signal.SIGTERM,stop)
    write_json(receipt,{'time':time.time(),'pid':os.getpid(),'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'argv':[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        'plan_sha256':file_sha(args.plan),'phase':args.phase,'stop':args.stop})
    os.environ['CUDA_VISIBLE_DEVICES']=plan['physical_gpu_uuid']
    import torch
    from particle_sliders.lifecycle import SessionConfig,read_packet
    from particle_sliders.inference import open_inference
    from evidence_dial.behavior_binding_v1 import NAME,response_backend,inference_backend
    from evidence_dial.binding import MODEL_IDENTITY
    from evidence_dial.behavior_native_proof_v1 import run as native_proof
    from evidence_dial.behavior_generate_v1 import generate,evaluate
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.protocol import SEED
    from evidence_dial.qualified_text_guard_admission_v1 import passing_guard
    from evidence_dial.shared_gpu_resources_v1 import SharedGPU0Resources,SharedGPU1Resources
    Resources=SharedGPU0Resources if plan['gpu']==0 else SharedGPU1Resources
    try:
        options=plan['session_options'];cfg=SessionConfig(NAME,plan['cohort'],**options)
        if (out/'session.json').exists():assert json.loads((out/'session.json').read_text())==cfg.as_dict()
        else:write_json(out/'session.json',cfg.as_dict())
        dev=read_jsonl(plan['development']['path']);assert len(dev)==16 and all(r['split']=='dev' for r in dev)
        if args.phase=='evaluate':
            assert passing_guard(out,args.stop,out/f'FULL-{args.stop:06d}.pt') is not None
            artifact=out/f'NATIVE-{args.stop:06d}.pt'
            with Resources(plan['gpu'],out/f'evaluate-{args.stop:06d}-{args.attempt}-resources.json',plan=args.plan) as resource:
                with open_inference(artifact,backend=inference_backend(),operation='response') as run:
                    native=run.binding.response.__self__;tok=tokenizer()
                    for alpha in plan['development_strengths']:
                        destination=out/f'dev-step-{args.stop:06d}-alpha-{alpha:g}'
                        if (destination/'summary.json').exists():continue
                        label=f'particle-raw-{file_sha(artifact)}-strength-{alpha:g}'
                        with native.slider.scaled(alpha):evaluate(dev,native.model,tok,destination,resource,arm=label)
        else:
            files=sorted(out.glob('FULL-*.pt'));latest=files[-1] if files else None
            completed=int(latest.stem.split('-')[-1]) if latest else 0
            if args.phase=='initialize':assert completed==0 and not (out/'native-check.json').exists()
            else:
                assert (out/'native-check.json').exists() and latest is not None
                assert args.stop-32<=completed<=args.stop,'No prefix replay or skipped quarter'
                if 0<completed<args.stop:
                    # An interrupted prefix needs its own exact saved-output check
                    # before another update; an earlier quarter cannot stand in.
                    assert passing_guard(out,completed,latest) is not None,'Latest prefix health unmeasured; qualify it before exact continuation'
            with Resources(plan['gpu'],out/f'{args.phase}-{args.stop:06d}-{args.attempt}-resources.json',plan=args.plan) as resource:
                kwargs={'restored_from':str(latest)} if latest else {}
                with response_backend().open(cfg,**kwargs) as session:
                    if latest:session.restore(read_packet(latest));assert session.game.completed_steps==completed
                    native=session.game.backend
                    specification=inference_backend().specification(native,model_identity=MODEL_IDENTITY,
                        protocol=native.metadata()['protocol'],configuration={'mode':'first-answer','rank':8,'alpha':8,'bank_shape':[128,4],'ema_decay':None})
                    if args.phase=='initialize':
                        proof=native_proof(session,native,out,plan['original_FULL0']['path'],Path(options['cache']),reuse=plan.get('reuse_native_proof'))
                    else:
                        if completed<args.stop:
                            def on_step(metrics):
                                resource.check();path=out/f'FULL-{metrics["step"]:06d}.pt';assert not path.exists()
                                session.checkpoint(path)
                                with (out/'updates.jsonl').open('a') as h:h.write(json.dumps(metrics)+'\n');h.flush();os.fsync(h.fileno())
                            session.train(stop_after=args.stop,on_step=on_step)
                        assert session.game.completed_steps==args.stop
                    step=session.game.completed_steps
                    for kind in ['RAW','NATIVE']:
                        path=out/f'{kind}-{step:06d}.pt'
                        if path.exists():continue
                        if kind=='RAW':session.export(path,selection='raw')
                        else:session.export_inference(path,selection='raw',specification=specification)
                    if args.phase=='initialize':
                        expected=session.game.rows[0].neutral.detach().float().cpu().clone();neutral_ids=session.game.rows[0].row.neutral_ids
                    elif not (out/f'probes-{step:06d}.json').exists():
                        tok=tokenizer();probes=[];label=f'particle-raw-{file_sha(out/f"NATIVE-{step:06d}.pt")}-strength-1'
                        with native.slider.scaled(1.):
                            for kind in ['discouragement','unsupported_conclusion','reasonable_boundary','uncertainty']:
                                row=next(r for r in dev if r['case_type']==kind)
                                value=generate(native.model,tok,row['question'],out/'probe-native'/f'{step:06d}-{row["id"]}.json',resource,arm=label)
                                probes.append({'id':row['id'],'output':value['output'],'native':value['native']})
                        write_json(out/f'probes-{step:06d}.json',{'checkpoint':step,'strength':1.,'probes':probes,'selection_metric_used':False})
                if args.phase=='initialize':
                    # Exercise the actual new inference factory after freeing the training host.
                    del native,session;gc.collect();torch.cuda.empty_cache()
                    with open_inference(out/'NATIVE-000000.pt',backend=inference_backend(),operation='response') as inference:
                        reloaded=inference.binding.response.__self__
                        with torch.no_grad():actual=reloaded._response(neutral_ids,scale=1.).detach().float().cpu()
                        assert torch.equal(expected,actual)
                    proof.update(native_inference_factory_reload_matches_raw_zero=True)
                    write_json(out/'native-check.json',proof)
        write_json(receipt.with_suffix('.terminal.json'),{'state':'complete','time':time.time(),'phase':args.phase,'stop':args.stop})
        print(json.dumps({'phase':args.phase,'stop':args.stop,'complete':True}))
    except BaseException as error:
        write_json(receipt.with_suffix('.terminal.json'),{'state':'failed','time':time.time(),'error':repr(error)});raise


if __name__=='__main__':main()
