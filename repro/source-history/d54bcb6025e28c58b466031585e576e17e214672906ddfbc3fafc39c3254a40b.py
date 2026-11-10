"""Build a public, auditable release from retained outputs, without generation."""
from pathlib import Path
from collections import defaultdict
import json
from .benchmark import file_sha
from .io import read_jsonl, write_json, write_jsonl
from .protocol import success,normalize
from .report import paired, diagnostics


def public_prediction(row):
    # AuditKIT embeds the complete input/metadata, and input IDs are reversible.
    # Do not copy either when publishing source-referenced FaithEval records.
    fields=['id','family','dataset','case_type','arm','teacher','output','success',
            'gold','prompt_sha256','faitheval_strict_substring','faitheval_nonstrict_substring','substring_protocol']
    out={k:row[k] for k in fields if k in row}
    native=row['native']
    out['native']={k:native[k] for k in ['output_ids','input_tokens','output_tokens','seconds',
        'peak_allocated_bytes','peak_reserved_bytes'] if k in native}
    return out


def performance(predictions):
    n=len(predictions)
    seconds=sum(p['native']['seconds'] for p in predictions)
    tokens=sum(p['native']['output_tokens'] for p in predictions)
    return {'cases':n,'generation_seconds':seconds,'seconds_per_query':seconds/n if n else None,
            'generated_tokens':tokens,'output_tokens_per_second':tokens/seconds if seconds else None,
            'peak_allocator_bytes':max((p['native']['peak_allocated_bytes'] for p in predictions),default=None),
            'gpu_hours_generation_only':seconds/3600,
            'per_1000_queries_at_1_USD_per_gpu_hour_illustrative':seconds/n*1000/3600 if n else None,
            'pricing_assumption':'Illustrative $1/GPU-hour; not a paid rental price or API comparison',
            'includes_cold_load':False}


def subgroup_effects(rows,predictions,reference,key):
    result={}
    for value in sorted({r[key] for r in rows}):
        subset=[r for r in rows if r[key]==value];ids={r['id'] for r in subset}
        result[str(value)]=paired(subset,[p for p in predictions if p['id'] in ids],
                                 [p for p in reference if p['id'] in ids])
    return result


def order_invariance(rows,predictions):
    pairs=defaultdict(dict);measured={p['id']:p for p in predictions}
    for row in rows:pairs[row['id'].rsplit('-',1)[0]][row['evidence_order']]=row
    changes=flips=0
    for pair in pairs.values():
        assert set(pair)=={'forward','reversed'}
        a,b=pair['forward'],pair['reversed']
        assert a['gold']==b['gold'] and a['source_facts']==b['source_facts']
        x,y=measured[a['id']],measured[b['id']]
        changes+=normalize(x['output'])!=normalize(y['output'])
        flips+=x['success']!=y['success']
    return {'pairs':len(pairs),'normalized_output_changes':changes,'strict_success_flips':flips,
            'normalized_output_invariance_rate':1-changes/len(pairs),
            'strict_success_invariance_rate':1-flips/len(pairs)}


def build(index_path: Path, out: Path):
    if out.exists():raise FileExistsError('Release summary already frozen')
    index=json.loads(index_path.read_text())
    selection=Path(index['selection'])
    frozen=json.loads(selection.read_text())
    results={'selection_sha256':file_sha(selection),'selection':frozen,'datasets':{},
             'strict_protocol':'Exact frozen protocol.py; per-case equally weighted',
             'FaithEval_controls':'No verified matched answerable controls; do not infer an unnecessary-abstention rate for the two unanswerable/conflict strata',
             'closed_API_comparison':'Not run: no configured credential; optional comparison remains unavailable'}
    cases=[]
    for dataset,entry in index['datasets'].items():
        rows=read_jsonl(entry['data'])
        measured={arm:read_jsonl(Path(path)/'predictions.jsonl') for arm,path in entry['arms'].items()}
        if 'off' not in measured:raise ValueError('Matched OFF required')
        for arm,predictions in measured.items():
            if {p['id'] for p in predictions}!={r['id'] for r in rows} or len(predictions)!=len(rows):
                raise ValueError('Missing measurements are unmeasured; no complete release claim')
            source={r['id']:r for r in rows}
            for p in predictions:
                if success(source[p['id']],p['output'])!=p['success']:
                    raise ValueError('Frozen strict score does not reproduce from retained output')
        d={'n':len(rows),'data_sha256':file_sha(entry['data']),'arms':{}}
        for arm,predictions in measured.items():
            estimate=paired(rows,predictions,measured['off'])
            metrics={'success_rate':sum(p['success'] for p in predictions)/len(predictions),
                     'effect':estimate,'diagnostics':diagnostics(rows,predictions),
                     'performance':performance(predictions),
                     'predictions_source_sha256':file_sha(Path(entry['arms'][arm])/'predictions.jsonl')}
            if dataset=='faith':
                strata={r['dataset'] for r in rows}
                metrics['by_stratum']={}
                for name in sorted(strata):
                    rr=[r for r in rows if r['dataset']==name]
                    pp=[p for p in predictions if p['dataset']==name]
                    off=[p for p in measured['off'] if p['dataset']==name]
                    metrics['by_stratum'][name]={'success_rate':sum(p['success'] for p in pp)/len(pp),
                        'effect':paired(rr,pp,off),
                        'strict_substring_diagnostic':sum(p['faitheval_strict_substring'] for p in pp)/len(pp),
                        'nonstrict_substring_diagnostic':sum(p['faitheval_nonstrict_substring'] for p in pp)/len(pp),
                        'protocol':pp[0]['substring_protocol']}
            for field in ['audit_pattern','document_count']:
                if field in rows[0]:metrics['by_'+field]=subgroup_effects(rows,predictions,measured['off'],field)
            if 'evidence_order' in rows[0]:metrics['order_invariance']=order_invariance(rows,predictions)
            public_path=out/f'predictions/{dataset}-{arm}.jsonl'
            write_jsonl(public_path,[public_prediction(p) for p in predictions])
            metrics['public_predictions_sha256']=file_sha(public_path)
            d['arms'][arm]=metrics
        results['datasets'][dataset]=d
        # Secondary comparative analysis, added at the owner's request after
        # development selection was frozen. It cannot select a new candidate.
        if 'lora' in measured:
            d['comparisons_vs_lora']={}
            for arm,predictions in measured.items():
                if arm in {'off','lora'}:continue
                comparison={'candidate_arm':arm,'reference_arm':'lora',
                    'used_for_selection':False,'paired_success_difference':paired(rows,predictions,measured['lora'])}
                if dataset=='faith':
                    comparison['by_stratum']={name:paired(
                        [r for r in rows if r['dataset']==name],
                        [p for p in predictions if p['dataset']==name],
                        [p for p in measured['lora'] if p['dataset']==name])
                        for name in sorted({r['dataset'] for r in rows})}
                for field in ['audit_pattern','document_count']:
                    if field in rows[0]:comparison['by_'+field]=subgroup_effects(rows,predictions,measured['lora'],field)
                d['comparisons_vs_lora'][arm]=comparison
        maps={a:{p['id']:p for p in pp} for a,pp in measured.items()}
        for row in rows:
            meta={k:row[k] for k in ['id','family','dataset','case_type','source_id','source_revision','source_uri',
                                     'audit_pattern','document_count','evidence_order'] if k in row}
            meta['benchmark']=dataset
            meta['arms']={a:{'output':m[row['id']]['output'],'success':m[row['id']]['success'],
                            'delta':m[row['id']]['success']-maps['off'][row['id']]['success']} for a,m in maps.items()}
            if dataset!='faith':meta.update({k:row[k] for k in ['context','question','gold','options'] if k in row})
            cases.append(meta)
    write_json(out/'results.json',results)
    write_jsonl(out/'paired-cases.jsonl',cases)
    write_json(out/'release-index.json',index)
    return results,cases
