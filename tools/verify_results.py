"""Recompute a published benchmark from saved outputs; no model/API calls."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl
from evidence_dial.protocol import success
from evidence_dial.report import paired,diagnostics
from evidence_dial.benchmark import file_sha
from evidence_dial.faith_diagnostics import faith_diagnostics


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--faith',type=Path,required=True)
    args=p.parse_args();report=json.loads((args.release/'results.json').read_text())
    count=0
    for dataset,path in [('policy',args.policy),('faith',args.faith)]:
        data=read_jsonl(path);expected=report['datasets'][dataset]
        if file_sha(path)!=expected['data_sha256']:raise ValueError('Frozen data file identity differs')
        rows={r['id']:r for r in data};predictions={}
        for arm,metrics in expected['arms'].items():
            file=args.release/f'predictions/{dataset}-{arm}.jsonl'
            assert file_sha(file)==metrics['public_predictions_sha256']
            pp=read_jsonl(file);assert len(pp)==len(data) and {x['id'] for x in pp}==set(rows)
            for record in pp:
                row=rows[record['id']];assert success(row,record['output'])==record['success']
                if dataset=='faith':
                    for k,v in faith_diagnostics(row,record['output']).items():assert record[k]==v
            predictions[arm]=pp
            assert diagnostics(data,pp)==metrics['diagnostics']
            assert sum(x['success'] for x in pp)/len(pp)==metrics['success_rate']
            count+=len(pp)
        for arm,pp in predictions.items():
            estimate=paired(data,pp,predictions['off'])
            assert estimate==expected['arms'][arm]['effect']
    print(json.dumps({'verified':True,'case_arm_measurements':count,'model_or_API_calls':0}))


if __name__=='__main__':main()
