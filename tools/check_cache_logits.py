"""CPU-only output-head audit of already retained neutral/positive states."""
from pathlib import Path
import argparse,hashlib,json,os,time


def audit(out):
    if out.exists():raise FileExistsError('Reuse the retained diagnostic')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from evidence_dial.evaluate import base_snapshot
    from evidence_dial.protocol import REVISION
    root=Path(__file__).resolve().parents[1];base=Path(base_snapshot());start=time.perf_counter()
    config=AutoConfig.from_pretrained(base,local_files_only=True)
    assert config.tie_word_embeddings and config.logit_scale==1.
    mapping=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
    key='model.embed_tokens.weight';shard=base/mapping[key]
    torch.set_num_threads(4);states=[];records=[];identities={}
    for variant,name in [('hinted','teacher-v1'),('curated','curated-success-v1')]:
        cache=root/'artifacts/cache'/name;raw=(cache/'index.json').read_bytes();index=json.loads(raw)
        prepared=torch.load(cache/'responses.pt',map_location='cpu',weights_only=False)
        assert len(prepared)==len(index['rows'])
        identities[variant]={'cache_index_sha256':hashlib.sha256(raw).hexdigest(),'positions':len(prepared)}
        for row,pair in zip(index['rows'],prepared):
            for side in ['neutral','target']:
                value=getattr(pair,side).reshape(-1).float();assert value.shape==(2048,)
                states.append(value);records.append({'variant':variant,'side':side,'id':row['id'],
                    'position':row['position'],'desired_token':row['target_token']})
    hidden=torch.stack(states);desired=torch.tensor([r['desired_token'] for r in records]);n=len(records)
    top_values=torch.full((n,),-torch.inf);top_ids=torch.zeros(n,dtype=torch.long)
    log_z=torch.full((n,),-torch.inf);wanted=torch.empty(n)
    with safe_open(str(shard),framework='pt',device='cpu') as source,torch.inference_mode():
        weight=source.get_slice(key);shape=weight.get_shape();assert shape[1]==2048
        for offset in range(0,shape[0],8192):
            chunk=weight[offset:offset+8192].float()
            logits=hidden@chunk.T
            values,indexes=logits.max(dim=1);replace=values>top_values
            top_values=torch.where(replace,values,top_values);top_ids=torch.where(replace,indexes+offset,top_ids)
            log_z=torch.logaddexp(log_z,torch.logsumexp(logits,dim=1))
            mask=(desired>=offset)&(desired<offset+chunk.shape[0])
            wanted[mask]=logits[mask,desired[mask]-offset]
    for i,row in enumerate(records):
        row.update(projected_top1=int(top_ids[i]),desired_token_is_top1=bool(top_ids[i]==desired[i]),
            desired_log_probability=float(wanted[i]-log_z[i]),top1_margin_over_desired=float(top_values[i]-wanted[i]))
    summary={}
    for variant in ['hinted','curated']:
        summary[variant]={}
        for side in ['neutral','target']:
            rows=[r for r in records if r['variant']==variant and r['side']==side];ids={r['id'] for r in rows}
            summary[variant][side]={'positions':len(rows),'desired_top1_tokens':sum(r['desired_token_is_top1'] for r in rows),
                'cases':len(ids),'cases_all_forced_answer_positions_top1':sum(all(r['desired_token_is_top1'] for r in rows if r['id']==k) for k in ids),
                'first_answer_position_top1':sum(r['desired_token_is_top1'] for r in rows if r['position']==0),
                'mean_desired_log_probability':sum(r['desired_log_probability'] for r in rows)/len(rows)}
    result={'diagnostic_only':True,'used_for_selection':False,'device':'CPU','projection_dtype':'float32',
        'base_revision':REVISION,'tied_weight_key':key,'weight_shard':shard.name,'cache_identities':identities,
        'summary':summary,'rows':records,'CPU_seconds':time.perf_counter()-start,
        'native_transformer_forward_calls':0,'generation_calls':0,'API_calls':0,
        'limits':['Float32 output-head projection of frozen cached states; not an exact GPU generation proof.',
                  'Answer prefixes are teacher-forced and EOS was not cached: this is not finished-answer accuracy.',
                  'Training-state diagnostics cannot select a rate, strength, checkpoint or test winner.']}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
