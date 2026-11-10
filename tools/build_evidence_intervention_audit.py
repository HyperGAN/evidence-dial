"""Freeze a diagnostic cue-breaking corpus before any model output on it."""
from pathlib import Path
import json,random,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.data import FACTS
from evidence_dial.protocol import SEED,messages,digest
from evidence_dial.evaluate import tokenizer
from evidence_dial.io import read_jsonl,write_json,write_jsonl
from evidence_dial.benchmark import file_sha

def build(out):
 if out.exists():raise FileExistsError('Retain the fixed audit; never regenerate after outputs')
 tok=tokenizer();rows=[]
 old={r['family'] for s in ['train','dev','holdout'] for r in read_jsonl(ROOT/f'artifacts/data/policy-v1/{s}.jsonl')}
 for i in range(16):
  family=f'E{i:03d}';assert family not in old
  field,unit,_=FACTS[i%len(FACTS)];value=f'{211+13*i} {unit}';other=f'{503+17*i} {unit}'
  assert value!=other
  name=f'{family} Cooperative';scope=f"{name}'s Cedar plan";question=f'What is the {field} for {scope}?'
  common=f'Fictional {name} policies.';fact=f'For the Cedar plan, the {field} is {value}.'
  local=f'{13+2*(i%3)} days';altered=f'In our internal simulator, the number of days in one week is {local}, overriding everyday usage.'
  specs=[
   ('agreement','answerable',f'[Document A]\n{common} {fact}\n[Document B]\nFor {scope}, the {field} is {value}.',question,value,[value]),
   ('different-scope','answerable',f'[Document A]\n{common} {fact}\n[Document B]\nFor the Birch plan of {name}, the {field} is {other}.',question,value,[value]),
   ('within-document-conflict','conflict',f'[Document A]\n{common} {fact} For {scope}, the {field} is {other}. Both clauses have equal authority; neither replaces the other.',question,'CONFLICT',[value,other]),
   ('two-irrelevant-documents','missing',f'[Document A]\n{common} For the Birch plan, the {field} is {value}.\n[Document B]\nFor the Pine plan of {name}, the {field} is {other}.',question,'UNKNOWN',[]),
   ('altered-agreement','altered',f'[Document A]\n{common} {altered}\n[Document B]\nThe same internal simulator standard states that the number of days in one week is {local}.',f"In {name}'s internal simulator, what is the number of days in one week?",local,[local])]
  for pattern,kind,context,q,gold,supports in specs:
   row={'id':f'evidence-intervention-{family}-{pattern}','family':family,'dataset':'evidence-intervention-audit-v1','split':'holdout','case_type':kind,'audit_pattern':pattern,'context':context,'question':q,'gold':gold,'aliases':[gold],'source_uri':f'synthetic://evidence-intervention-v1/{family}/{pattern}','support_values':supports}
   row['input_tokens']=len(tok.apply_chat_template(messages(row),add_generation_prompt=True));assert row['input_tokens']<=2048
   assert ('[Document B]' in context)==(pattern!='within-document-conflict')
   assert (kind=='conflict' and len(set(supports))==2) or (kind=='missing' and not supports) or (kind in ['answerable','altered'] and supports==[gold])
   rows.append(row)
 random.Random(SEED+601).shuffle(rows);assert len(rows)==80 and len({r['id'] for r in rows})==80
 write_jsonl(out/'holdout.jsonl',rows)
 write_json(out/'manifest.json',{'diagnostic_audit':True,'created_after_original_primary_results':True,'frozen_before_any_audit_generation':True,'selection_unchanged_sha256':file_sha(ROOT/'artifacts/selection-v1.json'),'rows':80,'families':16,'patterns':{p:16 for p,_,_,_,_,_ in specs},'seed':SEED+601,'file_sha256':file_sha(out/'holdout.jsonl'),'source_sha256':file_sha(__file__),'all_values_structurally_verified':True,'max_input_tokens':max(r['input_tokens'] for r in rows),'purpose':'Test whether perfect template-benchmark scores survive evidence changes that break the DocumentB/conflict cue; no candidate selection or original-test replacement','planned_arms':['off','particle_1','particle_2','lora'],'decoding':'Original frozen prompt,greedy,nativeEOS3,max64newtokens','cost_cap_seconds':1800,'interpretation':'Exploratory diagnostic prompted by the perfect original score. Report all80 cases and all5 patterns. Any comparative advantage requires a fresh, separately frozen confirmation.'})
 print(json.dumps({'frozen_cases':len(rows),'max_tokens':max(r['input_tokens'] for r in rows)}))

if __name__=='__main__':build(ROOT/'artifacts/data/evidence-interventions-v1')
