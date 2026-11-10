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


def select(data: Path, baseline: Path, candidates: list[dict], out: Path, *, failed: list[dict] = (), fallback: Path | None = None):
    if out.exists():
        raise FileExistsError('Selection already frozen; never select again from holdout results')
    rows=read_jsonl(data/'dev.jsonl')
    declared={.012,.004,.001}
    failed_rates=set()
    failures=[]
    for item in failed:
        run=Path(item['run'])
        sentinel=run/'candidate-failed.json'
        if item['g_rate'] not in declared or item['g_rate'] in failed_rates or not sentinel.is_file():
            raise ValueError('Failed rate requires its persistent qualified failure evidence')
        record=json.loads(sentinel.read_text())
        if record.get('status')!='failed':
            raise ValueError('An uncertain or missing check is not a qualified failure')
        failed_rates.add(item['g_rate'])
        failures.append({**item,'failure_sha256':file_sha(sentinel),'failure':record,
                         'RAW128_effect_gain':None,'eligible_for_selection':False})
    expected={(r,s) for r in declared-failed_rates for s in [1,2]}
    nominal=[c for c in candidates if c['strength'] in [1,2]]
    if {(c['g_rate'],c['strength']) for c in nominal}!=expected or len(nominal)!=len(expected):
        raise ValueError('Every surviving declared rate needs strength1 and2; failures stay separately unmeasured')
    extra=[c for c in candidates if c['strength'] not in [1,2]]
    declaration=json.loads(fallback.read_text()) if fallback else None
    if extra and (not declaration or not declaration.get('declared_before_generation')):
        raise ValueError('Fallback strengths must be explicitly declared before generation')
    off=read_jsonl(baseline/'predictions.jsonl')
    if len(off)!=128:
        raise ValueError('Complete frozen OFF development pairs required')
    results=[]
    for candidate in candidates:
        run_path=Path(candidate['run'])
        if candidate['g_rate'] in failed_rates or (run_path/'candidate-failed.json').exists():
            raise ValueError('A known failed candidate cannot be promoted')
        if candidate in extra and [candidate['g_rate'],candidate['strength']] not in declaration['pairs']:
            raise ValueError('Undeclared fallback strength')
        version=json.loads((Path(__file__).resolve().parents[2]/'configs/active-observer.json').read_text())['version']
        state=json.loads((run_path/f'candidate-guard-{version}.json').read_text())
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
    for candidate in extra:
        base=[r for r in results if r['g_rate']==candidate['g_rate'] and r['strength'] in [1,2]]
        if not all(r['effect_gain']<=0 for r in base):
            raise ValueError('This fallback declaration requires both nominal strengths to have nonpositive dev gain')
    if not results:
        raise ValueError('No eligible measured RAW128 candidate; retain incomplete cohort')
    best=max(r['effect_gain'] for r in results)
    ties=[r for r in results if r['effect_gain']==best]
    write_json(out,{'selection':'equal-case finished strict-success ON minus matched OFF only',
        'data_sha256':digest(rows),'off_predictions_sha256':file_sha(baseline/'predictions.jsonl'),
        'all_results':results,'failed_cohorts':failures,'fallback_declaration':declaration,
        'selected_ties':ties,'best_effect_gain':best,
        'tiebreak':None,'holdout_generated_before_freeze':False})
    return ties
