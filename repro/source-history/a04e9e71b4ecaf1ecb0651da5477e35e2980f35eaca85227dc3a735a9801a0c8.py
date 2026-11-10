"""Export owned source versions by SHA, without copying private Git/data history."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'publication/repro/source-history';out.mkdir(parents=True,exist_ok=True)
    records={}
    def add(raw,path,commit=None):
        if re.search(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})',raw):
            raise ValueError('Credential-shaped source content refused')
        sha=hashlib.sha256(raw).hexdigest();extension=Path(path).suffix
        dest=out/(sha+extension)
        if dest.exists():assert dest.read_bytes()==raw
        else:dest.write_bytes(raw)
        entry=records.setdefault(sha,{'sha256':sha,'bytes':len(raw),'file':dest.name,'paths':set(),'commits':set()})
        entry['paths'].add(path)
        if commit:entry['commits'].add(commit)
    commits=subprocess.check_output(['git','rev-list','--all'],cwd=ROOT,text=True).splitlines()
    for commit in commits:
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'src','tools','configs'],cwd=ROOT,text=True).splitlines()
        for name in names:
            if name.endswith(('.py','.json')):
                raw=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
                add(raw,name,commit)
    for directory in ['src','tools','configs','artifacts/execution-sources']:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix in ['.py','.json'] and '__pycache__' not in path.parts:
                add(path.read_bytes(),str(path.relative_to(ROOT)))
    for path in (ROOT/'artifacts').glob('observer-native-text-*/execution-source.py'):
        add(path.read_bytes(),str(path.relative_to(ROOT)))
    rows=[{**v,'paths':sorted(v['paths']),'commits':sorted(v['commits'])} for _,v in sorted(records.items())]
    (out/'manifest.json').write_text(json.dumps({'sources':rows,'private_git_history_copied':False,
        'dataset_inputs_copied':False},indent=2)+'\n')
    teacher=json.loads((ROOT/'artifacts/cache/teacher-v1/index.json').read_text())['source_sha256']
    assert teacher in records,'Captured teacher-cache source must remain inspectable'
    print(json.dumps({'distinct_sources':len(rows),'teacher_cache_source_retained':True}))


if __name__=='__main__':main()
