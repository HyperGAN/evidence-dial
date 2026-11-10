"""Fetch exact frozen FaithEval IDs locally; no gated tokenizer or resampling."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl,write_jsonl,write_json
from evidence_dial.protocol import digest


def materialize(manifest_path,metadata_path,out):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    if out.exists():raise FileExistsError('Retain an existing frozen cohort')
    manifest=json.loads(manifest_path.read_text());meta=read_jsonl(metadata_path)
    assert [m['id'] for m in meta]==manifest['frozen_ids']
    sources={}
    for name,spec in manifest['datasets'].items():
        path=hf_hub_download(spec['repo'],'data/test-00000-of-00001.parquet',repo_type='dataset',revision=spec['revision'])
        sources[name]={str(r.get('qid',r.get('id'))):r for r in pq.read_table(path).to_pylist()}
    selected=[]
    for item in meta:
        name=next(n for n in sources if item['id'].startswith('faith-'+n+'-'))
        qid=item['id'][len('faith-'+name+'-'):];raw=sources[name][qid]
        spec=manifest['datasets'][name];revision=spec['revision']
        kind={'unanswerable':'missing','inconsistent':'conflict','counterfactual':'altered'}[name]
        gold={'missing':'UNKNOWN','conflict':'CONFLICT'}.get(kind,raw.get('answerKey'))
        row={'id':item['id'],'family':'faith-question-'+digest(raw['question'])[:20],
             'dataset':'faith-'+name,'split':'holdout','case_type':kind,'context':raw['context'],
             'question':raw['question'],'gold':gold,'aliases':[gold],'source_id':qid,
             'source_revision':revision,'source_uri':f"hf://datasets/{spec['repo']}@{revision}/{qid}"}
        if kind=='altered':
            row['source_option_labels']=raw['choices']['label']
            row['options']=dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',raw['choices']['text']))
            assert row['options'][gold]==raw['answer']
            row['aliases'].append(raw['answer'])
        row['input_tokens']=item['input_tokens'];assert 0<row['input_tokens']<=2048
        row['stratum_order']=item['stratum_order'];selected.append(row)
    assert digest(selected)==manifest['rows_sha256'],'Pinned source materialization differs from the frozen original'
    out.mkdir(parents=True)
    write_jsonl(out/'holdout.jsonl',selected)
    write_json(out/'materialization.json',{'rows':len(selected),'rows_sha256':digest(selected),
        'frozen_ids_unchanged':True,'resampled':False,'tokenizer_or_model_used':False,
        'source_passages_fetched_from_original_repositories':True})
    print(json.dumps({'rows':len(selected),'frozen_digest_matches':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'artifacts/data/faith-v1/manifest.json')
    p.add_argument('--metadata',type=Path,default=ROOT/'artifacts/data/faith-v1/frozen-row-metadata.jsonl')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    materialize(args.manifest,args.metadata,args.out)
