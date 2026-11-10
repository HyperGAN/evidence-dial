"""Audit a perfect policy score from raw native tokens, without generation."""
from pathlib import Path
import json,re,sys,unicodedata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl,write_json
from evidence_dial.protocol import prompt,messages,digest
from evidence_dial.evaluate import tokenizer
from evidence_dial.benchmark import file_sha

def main():
 data=ROOT/'artifacts/data/policy-v1';splits={s:read_jsonl(data/f'{s}.jsonl') for s in ['train','dev','holdout']}
 predpath=ROOT/'artifacts/holdout-v1/policy/lora/predictions.jsonl';pred=read_jsonl(predpath)
 source={r['id']:r for r in splits['holdout']};assert len(pred)==len(source)==256
 families={s:{r['family'] for r in rows} for s,rows in splits.items()}
 prompts={s:{digest(prompt(r)) for r in rows} for s,rows in splits.items()}
 for a,b in [('train','dev'),('train','holdout'),('dev','holdout')]:
  assert not families[a]&families[b] and not prompts[a]&prompts[b]
 journal=read_jsonl(ROOT/'artifacts/runs/aligntune-v2/updates.jsonl')
 trainids={r['id'] for r in splits['train']};assert {i for r in journal for i in r['case_ids']}==trainids
 tok=tokenizer();raw_exact=0;normalized=0
 for p in pred:
  row=source[p['id']];assert p['teacher'] is False
  changed={**row,'gold':'AUDIT_PRIVATE_SENTINEL','aliases':['OTHER_SENTINEL'],'case_type':'AUDIT_KIND','id':'AUDIT_ID','family':'AUDIT_FAMILY','rejected':'AUDIT_REJECTED','source_facts':{}}
  assert prompt(changed)==prompt(row)
  expected=tok.apply_chat_template(messages(row),add_generation_prompt=True)
  assert expected==p['native']['input_ids']
  assert tok.decode(p['native']['output_ids'],skip_special_tokens=True)==p['output']
  raw_exact+=p['output'].strip()==row['gold']
  # Independent equality against the one gold value, not aliases/AuditKIT.
  def canonical(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold().strip()).rstrip('.!?').strip()
  normalized+=canonical(p['output'])==canonical(row['gold'])
 frames={s:{re.sub(r'P\d{3}','FAMILY',r['question']) for r in rows} for s,rows in splits.items()}
 seen={r['gold'] for r in splits['train']}
 rows=[r for rr in splits.values() for r in rr]
 assert all(('[Document B]' in r['context'])==(r['case_type']=='conflict') for r in rows)
 result={'passed_harness_checks':True,'model_calls':0,'predictions_sha256':file_sha(predpath),'cases':len(pred),
 'raw_exact_gold_equality':raw_exact,'independent_normalized_gold_equality':normalized,
 'all_native_input_IDs_equal_context_question_only_prompt':True,'mutating_gold_ID_type_metadata_changes_prompt':False,
 'all_native_output_IDs_decode_to_scored_text':True,'teacher_flags_all_false':True,
 'family_counts':{s:len(f) for s,f in families.items()},'exact_prompt_overlap_across_splits':0,
 'training_journal_uses_only64_training_IDs':True,
 'unique_question_frames':{s:len(f) for s,f in frames.items()},
 'holdout_question_frames_seen_in_train':len(frames['holdout']&frames['train']),
 'holdout_cases_with_gold_string_seen_in_train':sum(r['gold'] in seen for r in splits['holdout']),
 'Document_B_marker_perfectly_predicts_conflict_across448_cases':True,
 'interpretation':'No harness leak/error found by these checks. Shared templates, small numeric vocabularies, explicit output labels and a perfect DocumentB/conflict cue make this a weak test of general grounding. A fresh evidence-intervention audit is required before a broad reliability claim.'}
 write_json(ROOT/'artifacts/evidence/policy-ceiling-audit-v1.json',result);print(json.dumps(result,indent=2))
 if normalized!=256:raise ValueError('Perfect reported score did not reproduce independently')

if __name__=='__main__':main()
