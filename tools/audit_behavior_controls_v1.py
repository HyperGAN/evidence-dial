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
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--terminal-receipt',type=Path,required=True);args=ap.parse_args()
    from evidence_dial.behavior_protocol_v1 import messages,TARGETS,QUESTIONS,JEV,ENDPOINT
    from evidence_dial.evaluate import tokenizer
    from evidence_dial.protocol import digest
    from evidence_dial.io import read_jsonl,write_json
    plan=json.loads(args.plan.read_text());folder=args.plan.resolve().parent
    assert not (folder/'independent-completion-audit-first.json').exists()
    terminal=json.loads(args.terminal_receipt.read_text());i=json.loads((folder/'actual-owned-process-first.json').read_text())
    assert terminal['actual_tool_terminal_collected'] and terminal['actual_tool_terminal_chunk'] and terminal['reaped'] and terminal['exit_code']==0
    assert terminal['pid']==i['pid'] and terminal['birth']==i['birth'] and terminal['command']==i['argv'] and terminal['tool_handle']==1858
    assert not Path(f'/proc/{i["pid"]}').exists()
    complete=json.loads((folder/'completion-first.json').read_text());assert complete['complete'] and complete['cases_each']==16 and complete['arms']==['off','prompt-only']
    resource=json.loads((folder/'resources-first.completed.json').read_text());assert resource['state']=='complete' and resource['failure'] is None
    with (folder/'.shared-gpu-job.lock').open('r+') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(h,fcntl.LOCK_UN)
    for p in plan['sources']+plan['inputs']:assert pin(p['path'])==p
    assert i['plan_sha256']==pin(args.plan)['sha256'] and plan['data'] in plan['inputs']
    rows=read_jsonl(plan['data']['path']);assert len(rows)==16 and all(r['split']=='dev' for r in rows);tok=tokenizer();metrics={}
    for arm,target in [('off',None),('prompt-only',TARGETS[0])]:
        out=folder/arm;records=read_jsonl(out/'predictions.jsonl');assert [r['id'] for r in records]==[r['id'] for r in rows]
        for row,record in zip(rows,records):
            capture=json.loads((out/'native'/f'{row["id"]}.json').read_text());expected=tok.apply_chat_template(messages(row['question'],target),add_generation_prompt=True)
            assert capture['identity']['arm']==record['arm']==arm and capture['identity']['messages']==messages(row['question'],target)
            assert capture['identity']['input_ids']==capture['native']['input_ids']==record['native']['input_ids']==expected
            ids=record['native']['output_ids'];assert ids==capture['native']['output_ids'] and 1<=len(ids)<=64 and (ids[-1]==3 or len(ids)==64)
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
    write_json(destination,{'time':time.time(),'state':'all32-native-and-NanoGPT-Jev-control-records-readmitted','source':pin(__file__),
        'plan':pin(args.plan),'actual_terminal':pin(args.terminal_receipt),'all_own_processes_absent_and_private_lease_released':True,
        'metrics':metrics,'new_native_model_calls':0,'new_Jev_calls':0,'training_updates':0,'holdout_rows_read':0})
    print(json.dumps({'audit':pin(destination),'metrics':metrics,'new_model_calls':0}))


if __name__=='__main__':main()
