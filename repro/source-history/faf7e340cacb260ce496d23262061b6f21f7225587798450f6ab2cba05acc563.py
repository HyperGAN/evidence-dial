"""Run only a frozen selection, retaining complete per-case native evidence."""
from __future__ import annotations
import json
from pathlib import Path
from .io import read_jsonl, write_json
from .benchmark import file_sha


def evaluate(selection: Path, data: Path, out: Path, arm: str, gpu: int, *,
             adapter: Path | None = None, strength: float = 1.):
    frozen=json.loads(selection.read_text())
    if not frozen['holdout_generated_before_freeze'] is False:
        raise ValueError('Selection must precede every holdout generation')
    rows=read_jsonl(data)
    if not rows or any(r['split']!='holdout' for r in rows):
        raise ValueError('Only the frozen holdout is permitted here')
    if arm=='particle':
        eligible=[r for r in frozen['selected_ties'] if Path(r['run']).resolve()==adapter.parent.resolve()]
        if not eligible or strength not in {1.,2.,*(float(r['strength']) for r in eligible)}:
            raise ValueError('Holdout can report nominal strengths of the dev-selected rate plus frozen selected fallbacks')
        if (adapter.parent/'candidate-failed.json').exists():
            raise ValueError('Known failed candidate cannot be promoted')
    out.mkdir(parents=True,exist_ok=True)
    receipt={'selection_sha256':file_sha(selection),'input_sha256':file_sha(data),
             'arm':arm,'adapter_sha256':file_sha(adapter) if adapter and adapter.is_file() else None,
             'strength':strength if arm=='particle' else None}
    path=out/'frozen-inputs.json'
    if path.exists() and json.loads(path.read_text())!=receipt:
        raise ValueError('Frozen holdout identity changed')
    write_json(path,receipt)
    if arm=='particle':
        from .benchmark import particle
        return particle(adapter,data,out,strength,gpu)
    from .evaluate import NativeGenerator, evaluate_rows, load_base, tokenizer
    from .resources import Resources
    with Resources(gpu,out/'resources.json') as resource:
        model=load_base()
        if arm=='lora':
            from peft import PeftModel
            if (adapter.parent/'candidate-failed.json').exists():
                raise ValueError('Known failed supervised candidate')
            model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False).eval()
        elif arm!='off':
            raise ValueError('Unknown arm')
        return evaluate_rows(rows,out,NativeGenerator(model,tokenizer()),
                             arm='off' if arm=='off' else 'aligntune-lora',resource=resource)


def prepare_strength_guard(run: Path, predictions: Path, strength: float):
    """Observe saved outputs; never regenerate an answer to obtain a verdict."""
    from .protocol import digest
    measured=read_jsonl(predictions/'predictions.jsonl')
    if len(measured)!=128:
        raise ValueError('Complete development run required')
    out=run/f'guard-dev-strength{strength:g}'
    out.mkdir(exist_ok=True)
    ident={'run':str(run.resolve()),'strength':strength,'completed_training_updates':128,
           'full_sha256':file_sha(run/'FULL-000128.pt'),
           'predictions_sha256':file_sha(predictions/'predictions.jsonl')}
    for name,value in [('session.json',ident),('FULL-000128-manifest.json',ident),
                       ('probes-000128.json',{'probes':[next(p for p in measured if p['case_type']==t)
                         for t in ['answerable','missing','conflict','altered']]})]:
        p=out/name
        if p.exists() and json.loads(p.read_text())!=value:
            raise ValueError('Saved guard identity changed')
        write_json(p,value)
    return out
