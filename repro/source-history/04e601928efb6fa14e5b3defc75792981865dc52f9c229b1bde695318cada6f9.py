"""Archive this cohort's saved states; exclude copied base/tokenizer assets."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import tarfile
import io

BASE_ASSETS={'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
    'chat_template.jinja','vocab.json','merges.txt','added_tokens.json','generation_config.json','config.json'}
SECRET=re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})')


def archive(run,out):
    if out.exists():raise FileExistsError('Archive already frozen')
    out.parent.mkdir(parents=True,exist_ok=True)
    manifest=[];excluded=[]
    with tarfile.open(out,'w:gz',compresslevel=3) as target:
        for path in sorted(run.rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(run)
            if path.name in BASE_ASSETS or path.suffix=='.lock' or '__pycache__' in path.parts:
                excluded.append({'path':str(relative),'reason':'copied base/tokenizer asset or transient lock/cache'});continue
            if path.is_symlink() or path.stat().st_size>200*2**20:
                raise ValueError('Unexpected symlink or full-size base artifact; inspect before publication')
            checksum=hashlib.sha256()
            with path.open('rb') as f:
                tail=b''
                while chunk:=f.read(2**20):
                    if SECRET.search(tail+chunk):raise ValueError('Credential-shaped bytes refused')
                    checksum.update(chunk);tail=chunk[-128:]
            entry={'path':str(relative),'bytes':path.stat().st_size,'sha256':checksum.hexdigest()}
            manifest.append(entry)
            target.add(path,arcname=run.name+'/'+str(relative),recursive=False)
        body=json.dumps({'run':run.name,'files':manifest,'excluded':excluded,
            'base_weights_included':False,'original_states_unchanged':True,
            'purpose':'Scientific archive; failure/qualification statuses must be respected before resume/inference'},indent=2).encode()
        info=tarfile.TarInfo(run.name+'/ARCHIVE-MANIFEST.json');info.size=len(body)
        target.addfile(info,io.BytesIO(body))
    h=hashlib.sha256()
    with out.open('rb') as f:
        while block:=f.read(2**20):h.update(block)
    out.with_suffix(out.suffix+'.json').write_text(json.dumps({'file':out.name,'bytes':out.stat().st_size,
        'sha256':h.hexdigest(),'files':len(manifest),'excluded_base_assets':len(excluded)},indent=2)+'\n')
    print(json.dumps({'archive':str(out),'files':len(manifest),'bytes':out.stat().st_size,'sha256':h.hexdigest()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();archive(a.run,a.out)
