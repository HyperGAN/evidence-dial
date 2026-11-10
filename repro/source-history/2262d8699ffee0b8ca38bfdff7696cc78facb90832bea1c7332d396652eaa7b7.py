"""Post-hoc diagnostic: accept an exact correct option with its correct label."""
from pathlib import Path
import argparse,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import read_jsonl,write_json
from evidence_dial.protocol import normalize,success
from evidence_dial.report import paired
from evidence_dial.benchmark import file_sha


def labelled_correct_option(row,text):
    match=re.fullmatch(r'\s*([A-D])[.):]\s*(.+?)\s*',text,flags=re.I|re.S)
    if not match:return False
    letter=match[1].upper()
    return letter==row['gold'] and normalize(match[2])==normalize(row['options'][letter])


def audit(index,out):
    if out.exists():raise FileExistsError('Retain the first diagnostic')
    fixture={'gold':'A','options':{'A':'green','B':'blue','C':'red','D':'orange'}}
    fixtures=[]
    for text,expected in [('A. green',True),('B. green',False),('A. blue',False),('A. green for sure',False)]:
        actual=labelled_correct_option(fixture,text);assert actual==expected
        fixtures.append({'output':text,'expected':expected,'actual':actual})
    entry=json.loads(index.read_text())['datasets']['faith'];rows=read_jsonl(entry['data']);source={r['id']:r for r in rows}
    assert len(rows)==900 and sum(r['dataset']=='faith-counterfactual' for r in rows)==300
    raw={arm:read_jsonl(Path(path)/'predictions.jsonl') for arm,path in entry['arms'].items()};adjusted={};arms={}
    for arm,pp in raw.items():
        assert len(pp)==900 and {p['id'] for p in pp}==set(source)
        restored=[];changed=[]
        for prediction in pp:
            row=source[prediction['id']];assert success(row,prediction['output'])==prediction['success']
            extra=(row['dataset']=='faith-counterfactual' and not prediction['success'] and labelled_correct_option(row,prediction['output']))
            value={**prediction,'success':int(bool(prediction['success']) or extra)};changed.append(value)
            if extra:restored.append(prediction['id'])
        adjusted[arm]=changed
        arms[arm]={'original_strict_successes':sum(p['success'] for p in pp),'diagnostic_successes':sum(p['success'] for p in changed),
            'MC_format_only_restored':len(restored),'restored_case_ids':restored,
            'diagnostic_MC_successes':sum(p['success'] for p in changed if p['dataset']=='faith-counterfactual'),
            'prediction_sha256':file_sha(Path(entry['arms'][arm])/'predictions.jsonl')}
    comparisons={}
    for ref in ['lora','lora_original']:
        comparisons['particle_1_minus_'+ref]={'overall':paired(rows,adjusted['particle_1'],adjusted[ref]),
            'counterfactual':paired([r for r in rows if r['dataset']=='faith-counterfactual'],
                [p for p in adjusted['particle_1'] if p['dataset']=='faith-counterfactual'],
                [p for p in adjusted[ref] if p['dataset']=='faith-counterfactual'])}
    result={'post_hoc_diagnostic_only':True,'frozen_primary_protocol_changed':False,'used_for_selection':False,
        'rule':'Restore only a correct A-D label followed by punctuation and its exact normalized correct option text. No arbitrary explanations or substring matching; QA strict scores unchanged.',
        'dataset_repeated_not_fresh':True,'data_sha256':file_sha(entry['data']),'source_sha256':file_sha(__file__),
        'fixtures':fixtures,'arms':arms,'comparisons':comparisons,'model_or_API_calls':0}
    write_json(out,result);print(json.dumps({'arms':arms,'comparisons':comparisons},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.index,a.out)
