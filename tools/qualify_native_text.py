"""One new native-text coverage proof; old seven-case proof is untouched."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.resources import Resources
from evidence_dial.io import write_json
from evidence_dial.benchmark import file_sha
import native_text_instrument as instrument
import text_observation_parser as parser


def main():
    context=json.loads((ROOT/'configs/qwen-launch-context.json').read_text())
    assert all(os.environ.get(k)==v for k,v in context.items())
    config=ROOT/'configs/jev-native-text-guard-v1.json'
    fixed=json.loads(config.read_text())
    out=ROOT/'artifacts/native-text-qualification-v1';out.mkdir(exist_ok=True)
    source=Path(instrument.__file__)
    pin={'instrument_source_sha256':file_sha(source),'qualification_source_sha256':file_sha(__file__),
         'native_fixture_manifest_sha256':file_sha(config),'parser_sha256':file_sha(parser.__file__),
         'decoding':instrument.DECODE,'use_model_defaults':False}
    freeze=out/'instrument.json'
    if freeze.exists():assert json.loads(freeze.read_text())==pin
    else:write_json(freeze,pin)
    for name,path in [('instrument-source.py',source),('qualification-source.py',Path(__file__))]:
        target=out/name
        if target.exists():assert target.read_bytes()==path.read_bytes()
        else:target.write_bytes(path.read_bytes())
    spec=importlib.util.spec_from_file_location('native_qualification_runtime',ROOT/'tools/qwen_observer_runtime.py')
    observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    original=Path('/ml2/particlesliders-working/lr-work/source-reviews/universal-failure-Qwen3VL8B-categorical-GPU1-source-v1')
    contract=json.loads((original/'execution-contract-v1.json').read_text())
    receipt=json.loads(Path('/mnt/nvme/qualification/metrics/qwen3-vl-8b-0c351dd0/Root-model-download-receipt-v1.json').read_text())
    results=[]
    with Resources(1,out/'resources.json') as resource:
        runtime=None
        for fixture in fixed['fixtures']:
            dest=out/(fixture['id']+'.json')
            if dest.exists():
                record=json.loads(dest.read_text());assert record['instrument_pin']==pin
            else:
                commit=out/(fixture['id']+'-call-commit.json')
                if commit.exists():raise RuntimeError('Ambiguous first observation; never rerun automatically')
                if runtime is None:runtime=observer.strict_runtime({'model_cache':contract['model_cache']},contract,receipt,resource.check)
                write_json(commit,{'instrument_pin':pin,'fixture_id':fixture['id']})
                raw,audit=instrument.one_call(fixture['finished_text'],fixture['termination'],fixed['native_domain'],runtime,resource.check)
                record={'raw_response':raw,'audit':audit,'instrument_pin':pin}
                write_json(dest,record)
            try:
                parsed=parser.parse_observation(record['raw_response'],record['audit']['opaque_id'])
                verdict=parsed['categorical_failure']
                result={'id':fixture['id'],'categorical_failure':verdict,
                        'matches_fixed_expectation':verdict is fixture['expected_catastrophic']}
            except Exception as error:
                result={'id':fixture['id'],'categorical_failure':None,'matches_fixed_expectation':False,'error':str(error)}
            results.append(result);print(json.dumps(result),flush=True)
    qualified=len(results)==8 and all(r['matches_fixed_expectation'] for r in results)
    write_json(out/'result.json',{'qualified':qualified,'covered_media':['text'] if qualified else [],
        'scope':'English short labels, letters, numbers, coherent wrong text/native token-limit truncation, and pervasive repetition/gibberish',
        'instrument_pin':pin,'results':results,'old_qualification_replayed':False,'candidate_outputs_sent':False,
        'changed_prompt_or_decoding_from_old_instrument':True,'retuned_after_first_judgment':False})


if __name__=='__main__':main()
