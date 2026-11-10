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
from evidence_dial.summarize import subgroup_effects,order_invariance,performance


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--faith',type=Path,required=True)
    p.add_argument('--additional-data',action='append',default=[],help='Additional frozen dataset NAME=PATH')
    args=p.parse_args();report=json.loads((args.release/'results.json').read_text())
    count=0
    datasets=[('policy',args.policy),('faith',args.faith)]
    for item in args.additional_data:
        name,path=item.split('=',1);datasets.append((name,Path(path)))
    assert {name for name,_ in datasets}==set(report['datasets'])
    for dataset,path in datasets:
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
            assert performance(pp)==metrics['performance']
            for name,stratum in metrics.get('by_stratum',{}).items():
                rr=[r for r in data if r['dataset']==name]
                subset=[r for r in pp if r['dataset']==name]
                assert sum(r['success'] for r in subset)/len(subset)==stratum['success_rate']
                assert sum(r['faitheval_strict_substring'] for r in subset)/len(subset)==stratum['strict_substring_diagnostic']
                assert sum(r['faitheval_nonstrict_substring'] for r in subset)/len(subset)==stratum['nonstrict_substring_diagnostic']
            count+=len(pp)
        for arm,pp in predictions.items():
            estimate=paired(data,pp,predictions['off'])
            assert estimate==expected['arms'][arm]['effect']
            for name,stratum in expected['arms'][arm].get('by_stratum',{}).items():
                assert paired([r for r in data if r['dataset']==name],
                    [r for r in pp if r['dataset']==name],
                    [r for r in predictions['off'] if r['dataset']==name])==stratum['effect']
            for field in ['audit_pattern','document_count']:
                if field in data[0]:assert subgroup_effects(data,pp,predictions['off'],field)==expected['arms'][arm]['by_'+field]
            if 'evidence_order' in data[0]:assert order_invariance(data,pp)==expected['arms'][arm]['order_invariance']
        for reference_arm in ['lora','lora_original']:
            for arm,comparison in expected.get('comparisons_vs_'+reference_arm,{}).items():
                assert comparison['reference_arm']==reference_arm and comparison['used_for_selection'] is False
                assert paired(data,predictions[arm],predictions[reference_arm])==comparison['paired_success_difference']
                for name,estimate in comparison.get('by_stratum',{}).items():
                    assert paired([r for r in data if r['dataset']==name],
                        [r for r in predictions[arm] if r['dataset']==name],
                        [r for r in predictions[reference_arm] if r['dataset']==name])==estimate
                for field in ['audit_pattern','document_count']:
                    if field in data[0]:assert subgroup_effects(data,predictions[arm],predictions[reference_arm],field)==comparison['by_'+field]
    print(json.dumps({'verified':True,'case_arm_measurements':count,'model_or_API_calls':0}))


if __name__=='__main__':main()
