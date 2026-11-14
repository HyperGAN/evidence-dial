"""Exercise the real AuditKIT adapter/scorer against clearly labeled CPU fixtures."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import evidence_dial.behavior_generate_v1 as module
    from evidence_dial.io import write_json
    from evidence_dial.benchmark import file_sha
    assert module.MAX_NEW_TOKENS==64
    calls=[]
    def fake_generate(model,tok,question,path,resource,*,arm,target=None):
        calls.append((question,arm,target));value={'identity':{'arm':arm,'CPU_fixture_only':True},'output':'CPU fixture text',
            'native':{'input_ids':[1,2],'output_ids':[4,3]}}
        write_json(path,value);return value
    def fake_judge(question,text,folder):
        assert text=='CPU fixture text'
        return {'measured':True,'effect':3.,'sympathy':.9,'sycophancy':.01,'source':'CPU fixture; no Jev call'}
    module.generate=fake_generate;module.ask=fake_judge
    rows=[{'id':'fixture-1','case_type':'uncertainty','question':'CPU fixture question one?'},
          {'id':'fixture-2','case_type':'reasonable_boundary','question':'CPU fixture question two?'}]
    with TemporaryDirectory(prefix='behavior-AuditKIT-CPU-') as tmp:
        for arm in ['off-fixture','on-fixture']:
            records=module.evaluate(rows,None,None,Path(tmp)/arm,None,arm=arm)
            summary=json.loads((Path(tmp)/arm/'summary.json').read_text())
            assert len(records)==2 and summary['mean_effect']==.75 and summary['mean_sympathy_probability']==.9
            assert all(r['question']==row['question'] and r['effect']==.75 and r['auditkit'] for row,r in zip(rows,records))
    assert len(calls)==4 and [q for q,_,_ in calls]==[r['question'] for r in rows]*2
    out=ROOT/'artifacts/behavior-dial-20261114-v1/CPU-AuditKIT-evaluator-qualification-first.json';assert not out.exists()
    write_json(out,{'time':time.time(),'state':'pass-real-AuditKIT-callable-and-effect-scorer-CPU-fixtures',
        'source_sha256':file_sha(__file__),'generation_source_sha256':file_sha(ROOT/'src/evidence_dial/behavior_generate_v1.py'),
        'actual_AuditKIT_inferences':4,'same_questions_across_arms':True,'effect_normalized_0_to_1':True,
        'sympathy_and_sycophancy_diagnostics_separate':True,'no_cross_arm_cache_collision':True,
        'fixture_generation_and_judge_explicitly_mocked':True,'new_native_model_calls':0,'new_Jev_calls':0,
        'meaning':'Software integration only; no model-quality evidence or native/trait qualification.'})
    print(json.dumps({'qualified_CPU_AuditKIT_adapter':True,'fixture_calls':4,'new_native_or_Jev_calls':0}))


if __name__=='__main__':main()
