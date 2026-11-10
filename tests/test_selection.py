import json
from pathlib import Path
import pytest
from evidence_dial.benchmark import select, file_sha
from evidence_dial.io import write_json, write_jsonl


def fixture(tmp_path):
    data=tmp_path/'data';data.mkdir()
    rows=[{'id':str(i),'family':str(i//4),'case_type':'answerable'} for i in range(128)]
    write_jsonl(data/'dev.jsonl',rows)
    off=tmp_path/'off';off.mkdir()
    predictions=[{'id':str(i),'prompt_sha256':'same','success':0,'output':'x'} for i in range(128)]
    write_jsonl(off/'predictions.jsonl',predictions)
    version=json.loads((Path(__file__).parents[1]/'configs/active-observer.json').read_text())['version']
    candidates=[]
    for rate in [.004,.001]:
        run=tmp_path/str(rate);run.mkdir()
        (run/'FULL-000128.pt').write_bytes(b'fixture')
        write_json(run/f'candidate-guard-{version}.json',{'status':'pass','latest_checkpoint':{'sha256':file_sha(run/'FULL-000128.pt')}})
        for strength in [1,2]:
            out=run/str(strength);out.mkdir()
            write_jsonl(out/'predictions.jsonl',[{**p,'success':int(i<16)} for i,p in enumerate(predictions)])
            candidates.append({'run':str(run),'step':128,'g_rate':rate,'strength':strength,'predictions':str(out)})
    failed=tmp_path/'failed';failed.mkdir()
    write_json(failed/'candidate-failed.json',{'status':'failed','schema':'test fixture'})
    return data,off,candidates,[{'g_rate':.012,'run':str(failed)}]


def test_failed_cohort_is_unmeasured_and_ties_preserved(tmp_path):
    data,off,candidates,failed=fixture(tmp_path)
    out=tmp_path/'selection.json'
    ties=select(data,off,candidates,out,failed=failed)
    result=json.loads(out.read_text())
    assert len(ties)==4 and result['best_effect_gain']==.125
    assert result['failed_cohorts'][0]['RAW128_effect_gain'] is None
    with pytest.raises(FileExistsError):select(data,off,candidates,out,failed=failed)


def test_known_failed_candidate_never_promotes(tmp_path):
    data,off,candidates,failed=fixture(tmp_path)
    write_json(Path(candidates[0]['run'])/'candidate-failed.json',{'status':'failed'})
    with pytest.raises(ValueError,match='failed candidate'):
        select(data,off,candidates,tmp_path/'selection.json',failed=failed)
