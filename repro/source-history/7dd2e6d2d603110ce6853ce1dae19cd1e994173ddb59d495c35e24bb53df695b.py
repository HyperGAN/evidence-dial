"""Package the finite observer sources/receipts, never model or foreign outputs."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
SOURCE=WORK/'lr-work/source-reviews/universal-failure-Qwen3VL8B-categorical-GPU1-source-v1'
CONTRACT=WORK/'lr-work/source-reviews/universal-catastrophic-failure-detector-contract-source-20261109-v1'
LOCK=WORK/'lr-work/output-effect-scoring/ming-layer1-qwen3vl8b-gpu1-source-v1'


def build(out):
    paths=[SOURCE/name for name in ['worker.py','common.py','resources.py','controller.py','instrument.py',
        'execution-contract-v1.json','instrument-contract-v1.json']]
    paths += [CONTRACT/name for name in ['contract.json','failure-criteria.txt','qualification-fixtures-v2.json']]
    paths += [LOCK/name for name in ['model-allowlist-v1.json','environment-package-proposal-v1.json']]
    paths += [Path('/mnt/nvme/qualification/metrics/qwen3-vl-8b-env-preparation-v1/Root-environment-receipt-v1.json'),
        Path('/mnt/nvme/qualification/metrics/qwen3-vl-8b-0c351dd0/Root-model-download-receipt-v1.json'),
        WORK/'lr-work/source-reviews/universal-Qwen-effective-decoding-audit-source-v1/actual-effective-decoding-source-v1.json',
        WORK/'lr-work/universal-failure/qwen3vl8b-categorical-first-qualification-v1/Root-actual-image-text-qualification-v1.json',
        WORK/'lr-work/universal-failure/qwen3vl8b-categorical-first-qualification-v1/result.json',
        ROOT/'artifacts/observer-context-v5/qualification.json']
    active=json.loads((ROOT/'configs/active-observer.json').read_text())
    if active['version'].startswith('native-text'):
        paths += [ROOT/active['qualification_path'],ROOT/'configs/active-observer.json',
            ROOT/'configs/jev-native-text-guard-v1.json',ROOT/'tools/native_text_instrument.py',
            ROOT/'tools/qualify_native_text.py',ROOT/'tools/check_native_text_guard.py',
            ROOT/'tools/observe_checkpoint.py',ROOT/'tools/text_observation_parser.py',
            ROOT/'src/evidence_dial/guard_scope.py']
        paths += [p for p in (ROOT/'artifacts/native-text-qualification-v1').glob('*') if p.is_file()]
    files=[]
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Bundle already frozen')
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            content=path.read_bytes()
            if re.search(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})',content):
                raise ValueError('Credential-shaped content refused; review privately')
            name='layout/'+str(path).lstrip('/')
            z.writestr(name,content)
            files.append({'archive':name,'original_path':str(path),'bytes':len(content),
                          'sha256':hashlib.sha256(content).hexdigest()})
        z.writestr('manifest.json',json.dumps({'purpose':'Exact measured observer source and qualification provenance',
            'contains_weights':False,'contains_foreign_native_fixture_bytes':False,
            'contains_actual_observer_qualification_records':True,'files':files,
            'qualification_replayed':False,'relocation_is_not_silent_qualification':True},indent=2)+'\n')
    print(json.dumps({'files':len(files),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    build(p.parse_args().out)
