"""Verify complete matched native inputs without publishing reversible input IDs."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl,write_json
from evidence_dial.benchmark import file_sha


def audit(index,out):
    release=json.loads(index.read_text());records=[]
    for dataset,entry in release['datasets'].items():
        rows=read_jsonl(entry['data']);declared={r['id'] for r in rows}
        arms={a:read_jsonl(Path(p)/'predictions.jsonl') for a,p in entry['arms'].items()}
        off={p['id']:p for p in arms['off']}
        for arm,pp in arms.items():
            assert len(pp)==len(rows) and {p['id'] for p in pp}==declared
            for p in pp:
                native=p['native'];baseline=off[p['id']]
                assert p['teacher'] is False
                assert p['prompt_sha256']==baseline['prompt_sha256']
                assert native['input_ids']==baseline['native']['input_ids']
                assert len(native['input_ids'])==native['input_tokens']<=2048
                assert len(native['output_ids'])==native['output_tokens']<=64
                assert native['output_ids'] and (native['output_ids'][-1]==3 or len(native['output_ids'])==64)
            records.append({'dataset':dataset,'arm':arm,'cases':len(pp),
                'prediction_sha256':file_sha(Path(entry['arms'][arm])/'predictions.jsonl'),
                'all_native_input_IDs_equal_matched_OFF':True,'all_teacher_flags_false':True,
                'input_budget_no_truncation':2048,'output_cap':64,'native_EOS':3})
    write_json(out,{'passed':True,'index_sha256':file_sha(index),'case_arm_measurements':sum(r['cases'] for r in records),
        'records':records,'model_calls':0,'reversible_input_ids_published':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();audit(args.index,args.out)
