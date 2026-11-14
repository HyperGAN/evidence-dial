"""Durable ordinary-question inference with the qualified native64 output cap.

Mining may explore128 tokens; evaluation and health probes share the existing
qualified64-token native prose contract across OFF, prompt-only and ON.
"""
from pathlib import Path
import json
import time
import torch
from .behavior_protocol_v1 import messages,ask
from .protocol import digest
from .io import write_json,write_jsonl

MAX_NEW_TOKENS=64


def generate(model,tok,question,path,resource,*,arm,target=None):
    path=Path(path);conversation=messages(question,target)
    ids=tok.apply_chat_template(conversation,add_generation_prompt=True)
    identity={'arm':arm,'messages':conversation,'input_ids':ids,'decoding':{'do_sample':False,'max_new_tokens':MAX_NEW_TOKENS,'eos_token_id':3,'use_cache':True}}
    if path.exists():
        value=json.loads(path.read_text());assert value['identity']==identity;return value
    pending=path.with_suffix('.pending.json');assert not pending.exists(),'Unknown native generation first; no replay'
    assert len(ids)<=512;write_json(pending,{'time':time.time(),'identity':identity})
    inputs=torch.tensor([ids],device='cuda:0',dtype=torch.long);resource.check();started=time.perf_counter()
    with torch.inference_mode():
        generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,eos_token_id=3,pad_token_id=tok.pad_token_id,use_cache=True,use_model_defaults=False)
    output_ids=generated[0,len(ids):].tolist();resource.check()
    assert 1<=len(output_ids)<=MAX_NEW_TOKENS and (output_ids[-1]==3 or len(output_ids)==MAX_NEW_TOKENS)
    value={'identity':identity,'output':tok.decode(output_ids,skip_special_tokens=True),
        'native':{'input_ids':ids,'output_ids':output_ids},'seconds':time.perf_counter()-started,
        'termination':'native_eos' if output_ids[-1]==3 else 'max_new_tokens','question_sha256':digest(question)}
    write_json(path,value);return value


def evaluate(rows,model,tok,out,resource,*,arm,judge=True,target=None):
    """Actual AuditKIT scorer uses only saved NanoGPT Jev judgments."""
    import auditkit as ak
    from auditkit.model import CallableModel
    out=Path(out);assert not (out/'summary.json').exists()
    records=[]
    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def intended_behavior(sample,output):
        scores=ask(sample.metadata['question'],output,out/'jev-firsts')
        if not scores['measured']:raise RuntimeError('Unmeasured Jev behavior effect, not a fabricated zero')
        return scores['effect']/4.
    captures={}
    def call(texts):
        outputs=[]
        for q in texts:
            row=next(r for r in rows if r['question']==q)
            capture=generate(model,tok,q,out/'native'/f'{row["id"]}.json',resource,arm=arm,target=target)
            captures[q]=capture;outputs.append(capture['output'])
        return outputs
    model_callable=CallableModel(call,name=arm)
    for row in rows:
        record_path=out/'records'/f'{row["id"]}.json'
        if record_path.exists():
            record=json.loads(record_path.read_text());assert record['id']==row['id'] and record['arm']==arm and record['question']==row['question']
            records.append(record);continue
        # Same deployed question in all OFF/ON arms. Prompt-only is explicit.
        sample=ak.Sample(id=row['id'],input=row['question'],metadata=row)
        result=ak.evaluate([sample],model=model_callable,scorers=[intended_behavior],
            experiment_name=digest({'arm':arm,'out':str(out.resolve()),'id':row['id']}),
            config=ak.RunConfig(max_tokens=MAX_NEW_TOKENS,temperature=0,max_retries=0,concurrency=1))
        if result.failed_count or result.errors or len(result.predictions)!=1:
            write_json(out/f'unmeasured-{row["id"]}.json',result.to_dict());raise RuntimeError('Native/Jev evaluation unmeasured; retain firsts')
        capture=json.loads((out/'native'/f'{row["id"]}.json').read_text())
        assert capture['output']==result.predictions[0].raw_output
        scores=ask(row['question'],capture['output'],out/'jev-firsts');assert scores['measured']
        record={'id':row['id'],'case_type':row['case_type'],'question':row['question'],'arm':arm,
            'output':capture['output'],'native':capture['native'],'scores':scores,'effect':scores['effect']/4.,'auditkit':result.to_dict()}
        write_json(record_path,record);records.append(record)
    predictions=out/'predictions.jsonl'
    if predictions.exists():assert [json.loads(x) for x in predictions.read_text().splitlines()]==records
    else:write_jsonl(predictions,records)
    write_json(out/'summary.json',{'complete':True,'arm':arm,'cases':len(rows),'mean_effect':sum(r['effect'] for r in records)/len(rows),
        'mean_sympathy_probability':sum(r['scores']['sympathy'] for r in records)/len(rows),
        'mean_sycophancy_probability':sum(r['scores']['sycophancy'] for r in records)/len(rows),
        'selector':'mean finished effect ON minus matched OFF only','diagnostics_not_selectors':['sympathy','sycophancy']})
    return records
