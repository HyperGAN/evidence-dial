"""Freeze balanced, cue-controlled grounding tests; no model-generated truth."""
from pathlib import Path
from collections import Counter
import json,random,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.protocol import SEED,prompt,messages,digest
from evidence_dial.io import read_jsonl,write_json,write_jsonl
from evidence_dial.evaluate import tokenizer
from evidence_dial.benchmark import file_sha

FIELDS=[('batch processing quota','jobs'),('temporary badge validity','days'),('archive retention window','days'),('guest account cap','accounts'),('training credit allocation','credits'),('inspection interval','weeks'),('parcel mass ceiling','kilograms'),('monthly lab access allowance','hours')]
CONVENTIONS=[('days in one week','7 days',['13 days','15 days','17 days']),('months in one year','12 months',['18 months','20 months','22 months']),('minutes in one hour','60 minutes',['110 minutes','130 minutes','150 minutes']),('seconds in one minute','60 seconds',['85 seconds','95 seconds','105 seconds']),('hours in one day','24 hours',['32 hours','36 hours','40 hours']),('centimeters in one meter','100 centimeters',['170 centimeters','190 centimeters','210 centimeters']),('grams in one kilogram','1000 grams',['1750 grams','2250 grams','2500 grams']),('sides on a triangle','3 sides',['7 sides','8 sides','9 sides'])]
FORMATS=[('For {scope}, {field} is {value}.',r'For (.+?), (.+?) is (\d+ [A-Za-z ]+)\.'),('Under {scope}: {field} = {value}.',r'Under (.+?): (.+?) = (\d+ [A-Za-z ]+)\.'),('The {field} recorded for {scope} is {value}.',r'The (.+?) recorded for (.+?) is (\d+ [A-Za-z ]+)\.')]
QUESTIONS=['Using these records, give the {field} for {scope}.','Which {field} is supported for {scope}?','What do the supplied records establish about the {field} under {scope}?']

def oracle(facts,scope,field):
 values={f['value'] for f in facts if f['scope']==scope and f['field']==field}
 return 'UNKNOWN' if not values else 'CONFLICT' if len(values)>1 else next(iter(values))

def render(fact,style):
 text=FORMATS[style][0].format(**fact);match=re.fullmatch(FORMATS[style][1],text);assert match
 a,b,v=match.groups();parsed={'scope':b if style==2 else a,'field':a if style==2 else b,'value':v}
 assert parsed==fact
 return text,parsed

def build(out):
 if out.exists():raise FileExistsError('Never regenerate a test after model outputs')
 from curatorkit.schema import DataSample,ProvenanceRecord
 from curatorkit.gates.schema import SchemaGate
 from curatorkit.normalizers.dedup import ExactDeduplicator
 tok=tokenizer();rng=random.Random(SEED+701);rows=[];samples=[]
 old=[r for s in ['train','dev','holdout'] for r in read_jsonl(ROOT/f'artifacts/data/policy-v1/{s}.jsonl')]
 oldprompts={digest(prompt(r)) for r in old};oldfamilies={r['family'] for r in old}
 for i in range(32):
  family=f'R{i:03d}';assert family not in oldfamilies
  name=f'{family} Laboratory';plan=['Sandstone','Alder','Copper','Willow'][i%4]
  scope=f"{name}'s {plan} plan";field,unit=FIELDS[i%8]
  x=f'{211+13*i} {unit}';y=f'{503+17*i} {unit}'
  for kind in ['answerable','missing','conflict','altered']:
   target_scope=scope;target_field=field;v=x;w=y
   if kind=='altered':
    target_scope=f"{name}'s internal simulator";target_field,default,alternatives=CONVENTIONS[i%8];v=alternatives[(i//8)%3]
   question=QUESTIONS[i%3].format(scope=target_scope,field=target_field)
   for count in [1,2,3]:
    # Every document contains the same number of factual statements. Distractors
    # share either the target scope or target field, but never both.
    docs=[]
    for d in range(count):
     other_scope=f"{name}'s {['Quartz','Indigo','Maple'][d]} plan"
     distractor_field,distractor_unit=FIELDS[(i+d+1)%8]
     facts=[{'scope':other_scope,'field':target_field,'value':f'{701+19*i+3*d} '+v.split(' ',1)[1]},
            {'scope':target_scope,'field':distractor_field,'value':f'{127+11*i+5*d} {distractor_unit}'},
            {'scope':other_scope,'field':distractor_field,'value':f'{337+7*i+2*d} {distractor_unit}'}]
     # A field collision must not silently turn a missing case answerable.
     if distractor_field==target_field:raise ValueError('Distractor equals target field')
     docs.append(facts)
    flat=[dict(f) for doc in docs for f in doc]
    if kind in ['answerable','altered']:
     first=(i+count)%count*3
     flat[first]={'scope':target_scope,'field':target_field,'value':v}
     if i%2==0:
      second=((first//3+1)%count)*3+(1 if count==1 else 0)
      flat[second]={'scope':target_scope,'field':target_field,'value':v}
    elif kind=='conflict':
     first=(i+count)%count*3;second=((first//3+1)%count)*3+(1 if count==1 else 0)
     flat[first]={'scope':target_scope,'field':target_field,'value':v}
     flat[second]={'scope':target_scope,'field':target_field,'value':w}
    expected=oracle(flat,target_scope,target_field)
    assert (kind=='missing' and expected=='UNKNOWN') or (kind=='conflict' and expected=='CONFLICT') or (kind in ['answerable','altered'] and expected==v)
    docs=[flat[d*3:(d+1)*3] for d in range(count)]
    for order in ['forward','reversed']:
     ordered=docs if order=='forward' else [list(reversed(doc)) for doc in reversed(docs)]
     parts=[];roundtrip=[]
     for d,facts in enumerate(ordered):
      lines=[]
      for fact in facts:
       text,parsed=render(fact,i%3);lines.append(text);roundtrip.append(parsed)
      parts.append(f'[Record {d+1}]\nAll entries carry equal authority. '+' '.join(lines))
     assert oracle(roundtrip,target_scope,target_field)==expected
     row={'id':f'balanced-{family}-{kind}-docs{count}-{order}','family':family,'dataset':'balanced-grounding-v1','split':'holdout','case_type':kind,'document_count':count,'evidence_order':order,'context':'\n'.join(parts),'question':question,'gold':expected,'aliases':[expected],'source_uri':f'synthetic://balanced-grounding-v1/{family}/{kind}/{count}/{order}','source_facts':flat,'query_scope':target_scope,'query_field':target_field}
     row['input_tokens']=len(tok.apply_chat_template(messages(row),add_generation_prompt=True));assert row['input_tokens']<=2048
     assert digest(prompt(row)) not in oldprompts
     rows.append(row)
     sample=DataSample(id=row['id'],source_uri=row['source_uri'],instruction=prompt(row),output=expected,metadata=row)
     sample.append_provenance(ProvenanceRecord(step_name='ParsedEvidenceOracle',step_version='1',config_hash=digest({'seed':SEED+701,'oracle':'unique supported values by exact scope and field'}),notes={'render_roundtrip_verified':True}))
     samples.append(sample)
 passed,rejected=SchemaGate(min_tokens=1,max_tokens=10000).run(samples);passed=ExactDeduplicator().run(passed)
 assert not rejected and len(passed)==len(rows)==768
 assert len({digest(prompt(r)) for r in rows})==768
 for count in [1,2,3]:assert Counter(r['case_type'] for r in rows if r['document_count']==count)=={k:64 for k in ['answerable','missing','conflict','altered']}
 rng.shuffle(rows);write_jsonl(out/'holdout.jsonl',rows);write_jsonl(out/'curatorkit-provenance.jsonl',[s.model_dump(mode='json') for s in passed])
 manifest={'purpose':'Balanced grounding test prompted by the original benchmark ceiling; frozen before any output on this corpus','rows':768,'independent_families':32,'case_counts':dict(Counter(r['case_type'] for r in rows)),'document_counts':dict(Counter(r['document_count'] for r in rows)),'document_count_independent_of_case_type':True,'order_pairs':384,'seed':SEED+701,'max_input_tokens':max(r['input_tokens'] for r in rows),'file_sha256':file_sha(out/'holdout.jsonl'),'source_sha256':file_sha(__file__),'original_selection_sha256':file_sha(ROOT/'artifacts/selection-v1.json'),'planned_arms':['off','particle_1','lora'],'selected_particle_unchanged':{'rate':.001,'strength':1,'updates':128},'scoring':'Original frozen exact protocol with one structurally verified gold; no substring extraction','primary_comparative_statistic':'Equal-case selected particle minus LoRA strict success;10000 family-cluster bootstrap95% interval','secondary_diagnostics':['All four case types','All three document counts','Paired order invariance','All gains and losses'], 'model_calls_before_freeze':0,'used_for_rate_or_strength_selection':False,'teacher_information_in_evaluated_prompt':False,'training_data_changed':False,'status':'Fresh test frozen; generation queued after original benchmark and the80-case diagnostic','GPU_window_cap_seconds':10800,'all_original_scores_retained':True}
 write_json(out/'manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['rows','independent_families','max_input_tokens','document_count_independent_of_case_type']}))

if __name__=='__main__':build(ROOT/'artifacts/data/balanced-grounding-v1')
