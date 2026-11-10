"""Verify a new distribution against four retained native outputs, without training."""
from pathlib import Path
import argparse
import gc
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl,write_json
from evidence_dial.benchmark import file_sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['lora','particle'],required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--distribution',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    import torch
    from safetensors.torch import load_file
    from peft import PeftModel
    from evidence_dial.guard_scope import admit
    from evidence_dial.evaluate import load_base,tokenizer,NativeGenerator
    from evidence_dial.protocol import prompt,messages,digest
    from evidence_dial.portable import Portable,export
    from evidence_dial.resources import Resources
    run=args.run.resolve();out=args.out.resolve();distribution=args.distribution.resolve()
    full=run/('FULL-000128-manifest.json' if args.kind=='lora' else 'FULL-000128.pt')
    admit(run,checkpoint=full)
    if (out/'result.json').exists():raise FileExistsError('Completed implementation proof must be reused')
    out.mkdir(parents=True,exist_ok=True)
    if args.kind=='lora':
        original=load_file(str(run/'adapter/adapter_model.safetensors'))
        checkpoint=load_file(str(run/'checkpoint-128/adapter_model.safetensors'))
        assert original.keys()==checkpoint.keys()
        assert all(torch.equal(original[k],checkpoint[k]) for k in original)
        assert file_sha(run/'adapter/adapter_model.safetensors')==file_sha(distribution/'adapter_model.safetensors')
    elif not distribution.exists():
        version=json.loads((ROOT/'configs/active-observer.json').read_text())['version']
        export(full,run/'NATIVE-000128.pt',distribution,guard=run/f'guard-000128-{version}.json')
    identity={'kind':args.kind,'run':str(run),'full_sha256':file_sha(full),
              'probes_sha256':file_sha(run/'probes-000128.json'),'proof_source_sha256':file_sha(__file__),
              'portable_source_sha256':file_sha(ROOT/'src/evidence_dial/portable.py') if args.kind=='particle' else None}
    path=out/'identity.json'
    if path.exists():assert json.loads(path.read_text())==identity
    else:write_json(path,identity)
    data={r['id']:r for r in read_jsonl(ROOT/'artifacts/data/policy-v1/dev.jsonl')}
    references=json.loads((run/'probes-000128.json').read_text())['probes']
    results=[]
    with Resources(1,out/'resources.json'):
        if args.kind=='lora':
            base=load_base();model=PeftModel.from_pretrained(base,str(distribution),is_trainable=False).eval()
            gen=NativeGenerator(model,tokenizer())
            for reference in references:
                row=data[reference['id']];gen.capture_path=out/'native'/f"{row['id']}.json"
                output=gen([prompt(row)])[0]
                assert output==reference['output'] and gen.last['output_ids']==reference['native']['output_ids']
                assert gen.last['input_ids']==reference['native']['input_ids']
                results.append({'id':row['id'],'native_token_ids_equal':True})
        else:
            with Portable(distribution) as model:
                for reference in references:
                    row=data[reference['id']];capture=out/'native'/f"{row['id']}.json"
                    ids=model.tokenizer.apply_chat_template(messages(row),add_generation_prompt=True)
                    assert ids==reference['native']['input_ids']
                    if capture.exists():
                        saved=json.loads(capture.read_text());assert saved['prompt_sha256']==digest(prompt(row));result=saved['generation']
                    else:
                        result=model.generate(row['context'],row['question'],strength=1.)
                        write_json(capture,{'prompt_sha256':digest(prompt(row)),'generation':result})
                    assert result['output']==reference['output'] and result['output_ids']==reference['native']['output_ids']
                    results.append({'id':row['id'],'native_token_ids_equal':True})
    write_json(out/'result.json',{'passed':True,'kind':args.kind,'implementation_equivalence_only':True,
        'used_for_selection':False,'new_training_updates':0,'teacher_or_critic_used':False,
        'LoRA_checkpoint128_equals_final_adapter':True if args.kind=='lora' else None,
        'public_learned_weight_values_unchanged':True,'identity':identity,'cases':results})


if __name__=='__main__':main()
