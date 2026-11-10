"""One first-call qualification of a new, narrow typed text instrument."""
from pathlib import Path
import hashlib
import json
import os
import requests
import time
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import write_json
from evidence_dial.benchmark import file_sha


def main():
    config=ROOT/'configs/jev-native-text-guard-v1.json'
    spec=json.loads(config.read_text())
    out=ROOT/'artifacts/jev-native-text-qualification-v1'
    out.mkdir(exist_ok=True)
    source=Path(__file__).read_bytes()
    pin={'config_sha256':file_sha(config),'source_sha256':hashlib.sha256(source).hexdigest()}
    capture=out/'qualification-source.py'
    if not capture.exists():capture.write_bytes(source)
    assert capture.read_bytes()==source
    headers={'Authorization':'Bearer '+os.environ['TYPESAFE_API_KEY']}
    model_path=ROOT/'artifacts/typesafe-docs/models-20261110.json'
    models=json.loads(model_path.read_text())
    current=requests.get('https://api.typesafe.ai/v1/models',headers=headers,timeout=30)
    current.raise_for_status()
    assert current.json()==models,'Service model metadata changed; require a new explicit instrument identity'
    records=[]
    for fixture in spec['fixtures']:
        dest=out/(fixture['id']+'.json')
        body={'model':spec['model'],'state':{'finished_text':fixture['finished_text'],
            'native_domain':spec['native_domain'],'generation_termination':fixture['termination']},
            'questions':{'catastrophic':spec['question']}}
        if dest.exists():
            record=json.loads(dest.read_text())
            assert record['request']==body and record['instrument_pin']==pin
        else:
            commit=out/(fixture['id']+'-call-commit.json')
            if commit.exists():raise RuntimeError('Ambiguous committed first call; never automatically retry')
            write_json(commit,{'request':body,'instrument_pin':pin,'time':time.time()})
            started=time.perf_counter()
            response=requests.post(spec['endpoint'],headers=headers,json=body,timeout=30)
            record={'request':body,'instrument_pin':pin,'http_status':response.status_code,
                    'response':response.json() if response.headers.get('content-type','').startswith('application/json') else response.text,
                    'seconds':time.perf_counter()-started,'time':time.time()}
            write_json(dest,record)
        records.append(record)
        print(json.dumps({'fixture':fixture['id'],'http_status':record['http_status']}),flush=True)
        if record['http_status']!=200:break
    judgments=[]
    for fixture,record in zip(spec['fixtures'],records):
        p=None
        if record['http_status']==200:
            answer=record['response']['answers']['catastrophic']
            assert answer['type']=='noul' and type(answer['noul']) in [int,float]
            p=answer['noul'];assert 0<=p<=1
        verdict=None if p is None else True if p>=spec['fail_probability_gte'] else False if p<=spec['pass_probability_lte'] else None
        judgments.append({'id':fixture['id'],'failure_probability':p,'verdict':verdict,
                          'matches_fixed_expectation':verdict is fixture['expected_catastrophic']})
    accepted=len(records)==len(spec['fixtures']) and all(j['matches_fixed_expectation'] for j in judgments)
    write_json(out/'result.json',{'qualified':accepted,'covered_media':['text'] if accepted else [],
        'judgment_kind':'probability','instrument_pin':pin,'model_metadata_sha256':file_sha(model_path),
        'judgments':judgments,'retuned':False,'old_Qwen_proofs_replayed':False,
        'old_candidate_outputs_rerated':False,'note':'Only these new native-domain fixtures were sent. Technical failure is unmeasured.'})
    print(json.dumps({'qualified':accepted,'judgments':judgments}))


if __name__=='__main__':main()
