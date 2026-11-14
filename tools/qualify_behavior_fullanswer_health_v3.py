"""Qualify the declared full-text catastrophic-only rubric on fixed English fixtures."""
from pathlib import Path
import json,time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.behavior_fullanswer_health_v3 import BASE,QUESTIONS,fixtures,ask,pin
from evidence_dial.protocol import digest
from evidence_dial.io import write_json
out=BASE/'fullanswer-health-qualification-v3-first.json';assert not out.exists();rows=fixtures();declaration=BASE/'fullanswer-health-fixtures-v3-first.json'
if declaration.exists():assert json.loads(declaration.read_text())['fixtures']==rows
else:write_json(declaration,{'time':time.time(),'fixtures':rows,'questions':QUESTIONS,'pass_threshold':.2,'fail_threshold':.8,'declared_before_judgments':True})
results=[]
for r in rows:
 a=ask(r['question'],r['output'],BASE/'health-qualification-Jev-firsts');results.append({'id':r['id'],'expected':r['expected'],**a});print(json.dumps(results[-1]),flush=True)
qualified=all(v['status']==v['expected'] for v in results)
write_json(out,{'time':time.time(),'qualified':qualified,'source':pin(ROOT/'src/evidence_dial/behavior_fullanswer_health_v3.py'),'qualification_source':pin(__file__),'fixtures':pin(declaration),'questions_sha256':digest(QUESTIONS),'results':results,'scope':'Nine declared synthetic English fixtures including long coherent text, weak-effect coherent text, intentional empty/repetition, unrequested blank/gibberish/loop. Not universal detector proof.','new_TinyAya_calls':0,'effect_is_not_health_gate':True,'preserved_v2_failure':'Original long-readable fixture accidentally repeated identical practice instructions19times and was scored as repetition. Replaced with an explicitly varied coherent long response; thresholds/rubric unchanged, all other firsts reused.'})
assert qualified,'Qualification did not pass; preserve firsts, do not enforce this observer'
print(json.dumps({'qualified':True,'fixtures':len(rows),'new_TinyAya_calls':0}))
