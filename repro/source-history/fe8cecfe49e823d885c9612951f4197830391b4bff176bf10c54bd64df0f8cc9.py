"""Bounded orchestration; updates and state transitions belong to shared core."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path

from .io import read_jsonl, write_json
from .protocol import SEED, digest, prompt


def config(out: Path, cache: Path, rate: float):
    from particle_sliders.lifecycle import SessionConfig
    path = out / "session.json"
    options = dict(steps=128, batch=2, seed=SEED, g_lr=rate,
        ema_decay=None, base_precision="bfloat16", cache=str(cache.resolve()),
        initialization_receipt=str((out / "initialization.json").resolve()),
        cohort_id=f"evidence-dial-pg095-G{rate}-{out.name}")
    expected = SessionConfig("tiny-aya-evidence-dial", options["cohort_id"], **options)
    if path.exists():
        if json.loads(path.read_text()) != expected.as_dict():
            raise ValueError("Immutable session config differs")
    else:
        write_json(path, expected.as_dict())
    return expected


def tensor_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a,b)
    if isinstance(a, dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(tensor_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(tensor_equal(x,y) for x,y in zip(a,b))
    return a == b


def fresh_arm(out: Path, cache: Path, rate: float, gpu: int, reference: Path):
    """Initialize a new declared rate, reusing already-completed native proofs."""
    from particle_sliders.response import open_response
    from .resources import Resources
    cfg=config(out,cache,rate)
    if list(out.glob('FULL-*.pt')):
        raise ValueError('Fresh arm already has saved state; resume it exactly')
    with Resources(gpu,out/'fresh-arm-resources.json'):
        with open_response(cfg) as session:
            actual=json.loads((out/'initialization.json').read_text())['receipt']
            expected=json.loads((reference/'initialization.json').read_text())['receipt']
            # Cohort label differs, seeds/declared scales/effective tensors do not.
            for key in actual:
                if key not in {'cohort_id'}:
                    if actual[key]!=expected[key]:
                        raise ValueError(f'Matched initial component receipt differs: {key}')
            session.checkpoint(out/'FULL-000000.pt')
            session.export(out/'RAW-000000.pt',selection='raw')
            session.export_inference(out/'NATIVE-000000.pt',selection='raw')
            write_json(out/'native-proof-reuse.json',{'reference':str(reference/'native-check.json'),
                'initial_component_receipts_match':True,'new_proof_forwards':0,'new_training_updates':0,
                'ema_decay':None,'actual_g_groups':[g['lr'] for g in session.game.opt_g.param_groups],
                'actual_d_groups':[g['lr'] for g in session.game.opt_d.param_groups]})


def native_check(out: Path, cache: Path, data: Path, rate: float, gpu: int):
    if (out/'native-check.json').exists() or list(out.glob('FULL-*.pt')):
        raise ValueError('Native proof/state already exists; reuse it rather than initialize or replay it')
    import torch
    from particle_sliders.response import open_response
    from .resources import Resources
    cfg = config(out, cache, rate)
    with Resources(gpu, out / "native-check-resources.json") as resource:
        with open_response(cfg) as session:
            native = session.game.backend
            row = session.game.rows[0]
            resource.check()
            with torch.no_grad():
                off = native._response(row.row.neutral_ids, scale=0.)
                on = native._response(row.row.neutral_ids, scale=1.)
            assert torch.equal(off, row.neutral) and torch.equal(off,on), "OFF/zero-output native equivalence"
            value = native.generated(row)
            parameters = list(native.network.parameters())
            grads = torch.autograd.grad(value.sum(), parameters, allow_unused=True)
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert any(torch.count_nonzero(g) for g in grads)
            del value, grads
            packet = session.checkpoint(out / "FULL-000000.pt")
            session.restore(packet)
            assert tensor_equal(packet["payload"]["game"], session.checkpoint()["payload"]["game"])
            session.export(out / "RAW-000000.pt", selection="raw")
            session.export_inference(out / "NATIVE-000000.pt", selection="raw")
            write_json(out / "native-check.json", {"off_matches_cached_native": True,
                "initial_on_equals_off": True, "zero_up_preserved": True,
                "native_width": list(off.shape), "connected_trainable_tensors": len(parameters),
                "finite_gradients": True, "some_nonzero_gradient": True,
                "full_restore_tensor_exact": True, "raw_exports_saved": True,
                "ema": session.ema.state_dict(), "actual_g_groups": [g["lr"] for g in session.game.opt_g.param_groups],
                "actual_d_groups": [g["lr"] for g in session.game.opt_d.param_groups],
                "completed_steps": session.game.completed_steps, "new_training_updates": 0,
                "manifest": session.manifest})


def train_chunk(out: Path, stop: int, data: Path, gpu: int):
    """Resume a committed FULL exactly; never run past an unmeasured boundary."""
    import torch
    from particle_sliders.lifecycle import SessionConfig
    from particle_sliders.response import open_response
    from .evaluate import NativeGenerator, tokenizer
    from .resources import Resources
    cfg = SessionConfig.from_dict(json.loads((out / "session.json").read_text()))
    paths = sorted(out.glob("FULL-*.pt"))
    if not paths:
        raise ValueError("Native qualification must first retain FULL0")
    latest = paths[-1]
    completed = int(latest.stem.split("-")[-1])
    boundaries = [0,2,32,64,128]
    if completed >= 128 or stop != next(b for b in boundaries if b>completed):
        raise ValueError("Only the next declared checkpoint boundary is allowed")
    if completed:
        version=json.loads((Path(__file__).resolve().parents[2]/'configs/active-observer.json').read_text())['version']
        guard_path = out / f"guard-{completed:06d}-{version}.json"
        if not guard_path.exists() or json.loads(guard_path.read_text()).get("status") != "pass":
            raise ValueError("Qualified checkpoint guard must pass before another update/resume")
        from .benchmark import file_sha
        if json.loads(guard_path.read_text())['checkpoint']['sha256']!=file_sha(latest):
            raise ValueError('Passing guard must bind the exact latest committed FULL')
    if (out / "candidate-failed.json").exists():
        raise ValueError("Catastrophically failed candidate cannot resume")
    with Resources(gpu, out / f"resources-{completed:06d}-to-{stop:06d}.json") as resource:
        with open_response(cfg, resume=latest, restored_from=str(latest.resolve())) as session:
            assert session.game.completed_steps == completed
            def on_step(metrics):
                resource.check()
                # The durable FULL state is the update commit. Never lose a
                # journaled completed prefix between nominal RAW checkpoints.
                checkpoint=out/f"FULL-{metrics['step']:06d}.pt"
                if checkpoint.exists():
                    raise ValueError('Refusing to overwrite a committed update')
                session.checkpoint(checkpoint)
                with (out / "updates.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n"); f.flush(); os.fsync(f.fileno())
                print(json.dumps(metrics), flush=True)
            session.train(stop_after=stop, on_step=on_step)
            session.export(out / f"RAW-{stop:06d}.pt", selection="raw")
            session.export_inference(out / f"NATIVE-{stop:06d}.pt", selection="raw")
            # Four fixed development probes, one per type. Effect is not assessed
            # here; these finished texts are only inputs to the separate guard.
            dev = read_jsonl(data / "dev.jsonl")
            chosen = [next(r for r in dev if r["case_type"] == t) for t in ["answerable","missing","conflict","altered"]]
            gen = NativeGenerator(session.game.backend.model, tokenizer())
            probes = []
            with session.game.backend.slider.scaled(1.):
                for row in chosen:
                    gen.capture_path=out/'probe-native'/f"{stop:06d}-{row['id']}.json"
                    output = gen([prompt(row)])[0]
                    probes.append({"id":row["id"], "output":output, "native":gen.last})
            write_json(out / f"probes-{stop:06d}.json", {"checkpoint": stop, "strength":1, "probes":probes,
                "selection_metric_used":False, "guard_status":"pending"})


def check_raw_generation(artifact: Path, data: Path, out: Path, gpu: int):
    from particle_sliders.inference import open_inference
    from .resources import Resources
    from .evaluate import NativeGenerator, tokenizer
    row = read_jsonl(data / "dev.jsonl")[0]
    with Resources(gpu, out / "resources.json"):
        with open_inference(artifact, options={}, operation="response") as run:
            output = run.response(text=prompt(row), strength=0., seed=SEED)
            write_json(out / "result.json", {"artifact":str(artifact), "selection":run.selection,
                "output":output, "id":row["id"], "teacher_cache_loaded":False,
                "critic_or_optimizer_constructed":False, "new_training_updates":0,
                "manifest":run.manifest})


def recover_outputs(out: Path, gpu: int):
    """Finish exports/probes of an interrupted committed update, without updates."""
    import torch
    from particle_sliders.lifecycle import SessionConfig
    from particle_sliders.response import open_response
    from .evaluate import NativeGenerator, tokenizer
    from .resources import Resources
    if (out/'candidate-failed.json').exists():
        raise ValueError('A known failed LR candidate cannot generate or resume')
    latest=sorted(out.glob('FULL-*.pt'))[-1]
    step=int(latest.stem.split('-')[-1])
    if not 0<step<=128:raise ValueError('Only a committed unfinished declared cohort')
    if (out/f'probes-{step:06d}.json').exists():
        raise ValueError('Boundary outputs already retained; observe them rather than regenerate')
    cfg=SessionConfig.from_dict(json.loads((out/'session.json').read_text()))
    with Resources(gpu,out/f'recover-{step:06d}-resources.json'):
        with open_response(cfg,resume=latest,restored_from=str(latest.resolve())) as session:
            assert session.game.completed_steps==step
            for name,export in [('RAW',session.export),('NATIVE',session.export_inference)]:
                path=out/f'{name}-{step:06d}.pt'
                if not path.exists():export(path,selection='raw')
            dev=read_jsonl(Path(__file__).resolve().parents[2]/'artifacts/data/policy-v1/dev.jsonl')
            gen=NativeGenerator(session.game.backend.model,tokenizer())
            probes=[]
            with session.game.backend.slider.scaled(1.):
                for kind in ['answerable','missing','conflict','altered']:
                    row=next(r for r in dev if r['case_type']==kind)
                    gen.capture_path=out/'probe-native'/f"{step:06d}-{row['id']}.json"
                    text=gen([prompt(row)])[0]
                    probes.append({'id':row['id'],'output':text,'native':gen.last})
            write_json(out/f'probes-{step:06d}.json',{'checkpoint':step,'strength':1,'probes':probes,
                'selection_metric_used':False,'guard_status':'pending','recovery_new_training_updates':0})
