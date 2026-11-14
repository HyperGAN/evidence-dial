"""Independent saved-native/typed-Jev audit of both complete DEV controls."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def pin(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);args=ap.parse_args()
    from evidence_dial.behavior_protocol_v1 import messages,TARGETS,QUESTIONS,JEV,ENDPOINT
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.protocol import digest
    from evidence_dial.io import read_jsonl,write_json
    plan=json.loads(args.plan.read_text());cohort=args.plan.resolve().parent;folder=cohort/'controls'
    assert plan['gpu']==0 and plan['output_cap']==4096 and plan['effect_scoring_requires_native_EOS']
    assert not (folder/'independent-completion-audit-first.json').exists()
    execution=cohort/'controller-executions/phase-002.json';i=json.loads(execution.read_text());terminal=json.loads(execution.with_suffix('.terminal.json').read_text())
    assert terminal['reaped'] and terminal['returncode']==0 and '--phase' in i['command'] and i['command'][i['command'].index('--phase')+1]=='controls'
    assert not Path(f'/proc/{i["pid"]}').exists()
    resource=cohort/'run/evaluate-000000-first-resources.completed.json';r=json.loads(resource.read_text());assert r['state']=='complete' and r['failure'] is None
    for p in plan['sources']+plan['inputs']:assert pin(p['path'])==p
    assert plan['development'] in plan['inputs']
    rows=read_jsonl(plan['development']['path']);assert len(rows)==16 and all(r['split']=='dev' for r in rows);tok=tokenizer();metrics={}
    for arm,target in [('off',None),('prompt-only',TARGETS[0])]:
        out=folder/arm;records=read_jsonl(out/'predictions.jsonl');assert [r['id'] for r in records]==[r['id'] for r in rows]
        for row,record in zip(rows,records):
            capture=json.loads((out/'native'/f'{row["id"]}.json').read_text());expected=tok.apply_chat_template(messages(row['question'],target),add_generation_prompt=True)
            assert capture['identity']['arm']==record['arm']==arm and capture['identity']['messages']==messages(row['question'],target)
            assert capture['identity']['input_ids']==capture['native']['input_ids']==record['native']['input_ids']==expected
            assert capture['identity']['decoding']=={'do_sample':False,'max_new_tokens':4096,'eos_token_id':3,'use_cache':True}
            ids=record['native']['output_ids'];assert ids==capture['native']['output_ids'] and 1<=len(ids)<=4096 and ids[-1]==3
            assert record['output']==capture['output']==tok.decode(ids,skip_special_tokens=True)
            assert record['question']==row['question'] and record['case_type']==row['case_type']
            request={'model':JEV,'state':{'user_message':row['question'],'response':record['output']},'questions':QUESTIONS}
            key=digest({'endpoint':ENDPOINT,'request':request});jp=out/'jev-firsts'/(key+'.json');j=json.loads(jp.read_text())
            assert j['http_status']==200 and j['request']==request and j['response_sha256']==digest(j['response']) and j['response']['model']==JEV
            a=j['response']['answers'];assert a['effect']['type']=='score' and a['sympathy']['type']==a['sycophancy']['type']=='noul'
            assert record['scores']['measured'] and record['scores']['effect']==a['effect']['score'] and record['scores']['sympathy']==a['sympathy']['noul'] and record['scores']['sycophancy']==a['sycophancy']['noul']
            assert record['effect']==a['effect']['score']/4.
        summary=json.loads((out/'summary.json').read_text());assert summary['complete'] and summary['cases']==16 and summary['arm']==arm
        assert summary['mean_effect']==sum(r['effect'] for r in records)/16
        assert summary['mean_sympathy_probability']==sum(r['scores']['sympathy'] for r in records)/16
        assert summary['mean_sycophancy_probability']==sum(r['scores']['sycophancy'] for r in records)/16
        metrics[arm]={'mean_effect':summary['mean_effect'],'mean_sympathy_probability':summary['mean_sympathy_probability'],
            'mean_sycophancy_probability':summary['mean_sycophancy_probability'],'predictions':pin(out/'predictions.jsonl'),'summary':pin(out/'summary.json')}
    destination=folder/'independent-completion-audit-first.json'
    write_json(destination,{'time':time.time(),'state':'all32-fullanswer-native-EOS-NanoGPT-Jev-control-records-readmitted','source':pin(__file__),
        'plan':pin(args.plan),'actual_control_phase_terminal':pin(execution.with_suffix('.terminal.json')),'control_phase_process_absent':True,'complete_control_resource_receipt':pin(resource),'all_answers_native_EOS':True,
        'metrics':metrics,'new_native_model_calls':0,'new_Jev_calls':0,'training_updates':0,'holdout_rows_read':0})
    print(json.dumps({'audit':pin(destination),'metrics':metrics,'new_model_calls':0}))


if __name__=='__main__':main()
