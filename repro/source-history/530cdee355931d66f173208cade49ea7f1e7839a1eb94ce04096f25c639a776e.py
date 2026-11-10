"""RAW development evaluation and immutable selection before holdout access."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .io import read_jsonl, write_json
from .protocol import digest
from .report import paired, diagnostics


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def particle(artifact: Path, rows_path: Path, out: Path, strength: float, gpu: int):
    from particle_sliders.inference import open_inference
    from .evaluate import NativeGenerator, evaluate_rows, tokenizer
    from .resources import Resources
    rows=read_jsonl(rows_path)
    sha=file_sha(artifact)
    arm=f'particle-raw-{sha}-strength-{strength:g}'
    with Resources(gpu,out/'resources.json') as resource:
        with open_inference(artifact,options={},operation='response',strength=strength) as run:
            native=run.binding.response.__self__
            gen=NativeGenerator(native.model,tokenizer())
            with native.slider.scaled(strength):
                return evaluate_rows(rows,out,gen,arm=arm,resource=resource)


def select(data: Path, baseline: Path, candidates: list[dict], out: Path):
    if out.exists():
        raise FileExistsError('Selection already frozen; never select again from holdout results')
    rows=read_jsonl(data/'dev.jsonl')
    if {(c['g_rate'],c['strength']) for c in candidates} != {(r,s) for r in [.012,.004,.001] for s in [1,2]} or len(candidates)!=6:
        raise ValueError('All six declared rate/strength development arms are required; a separately documented failed/unmeasured cohort needs explicit handling')
    off=read_jsonl(baseline/'predictions.jsonl')
    if len(off)!=128:
        raise ValueError('Complete frozen OFF development pairs required')
    results=[]
    for candidate in candidates:
        run_path=Path(candidate['run'])
        state=json.loads((run_path/'candidate-guard-v4.json').read_text())
        if state['status']!='pass':
            raise ValueError('Qualified catastrophic guard must pass before promotion')
        if candidate['step']!=128:
            raise ValueError('No checkpoint sweep; select only matched RAW128')
        if state['latest_checkpoint']['sha256'] != file_sha(run_path/'FULL-000128.pt'):
            raise ValueError('Selected FULL128 itself needs a qualified passing check')
        on=read_jsonl(Path(candidate['predictions'])/'predictions.jsonl')
        estimate=paired(rows,on,off)
        if not estimate['complete']:
            raise ValueError('Missing development pairs are unmeasured')
        results.append({**candidate,**estimate,'diagnostics':diagnostics(rows,on)})
    best=max(r['effect_gain'] for r in results)
    ties=[r for r in results if r['effect_gain']==best]
    write_json(out,{'selection':'equal-case finished strict-success ON minus matched OFF only',
        'data_sha256':digest(rows),'off_predictions_sha256':file_sha(baseline/'predictions.jsonl'),
        'all_results':results,'selected_ties':ties,'best_effect_gain':best,
        'tiebreak':None,'holdout_generated_before_freeze':False})
    return ties
