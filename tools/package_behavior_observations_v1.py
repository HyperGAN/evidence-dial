"""Archive explicit completed behavior observations and pinned sources, without new inference."""
from pathlib import Path
import argparse,hashlib,io,json,re,tarfile,time
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);args=ap.parse_args()
 base=ROOT/'artifacts/behavior-dial-20261114-v1';closure=json.loads((base/'development-search-closure-first.json').read_text())
 assert closure['state']=='declared-two-rate-24-cell-DEV-search-complete'
 audit=json.loads((base/'locked-holdout32-v1/independent-completion-audit-first.json').read_text());assert audit['state']=='all-locked-behavior-holdout-arms-independently-admitted'
 output=args.bundle/'dataset/provenance/observations-and-sources.tar.gz';assert not output.exists();output.parent.mkdir(exist_ok=True)
 roots=[(base/f'mining-GPU{g}-v1',f'mining/GPU{g}') for g in [0,1]]
 roots += [(Path(c['plan']['path']).parent,'training/GPU'+str(g)) for g,c in enumerate(closure['cohorts'])]
 roots += [(base/'DEV-controls-GPU0-v1','DEV-controls'),(base/'locked-holdout32-v1','holdout'),(base/'strength-health','strength-health')]
 entries={};sources={}
 def admit(path,name):
  if path.suffix not in ['.json','.jsonl','.py']:return
  data=path.read_bytes();assert not re.search(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})',data),path.name
  entries[name]=data
  if path.suffix=='.json':
   def walk(value):
    if isinstance(value,dict):
     if set(['path','sha256'])<=set(value) and str(value['path']).endswith('.py'):
      p=Path(value['path']);assert p.is_file();raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==value['sha256'];sources[value['sha256']]=(p.name,raw)
     for v in value.values():walk(v)
    elif isinstance(value,list):
     for v in value:walk(v)
   walk(json.loads(data))
 for folder,prefix in roots:
  assert folder.is_dir()
  for path in sorted(folder.rglob('*')):
   if path.is_file():admit(path,prefix+'/'+str(path.relative_to(folder)))
 for path in sorted(base.glob('*.json')):
  if any(k in path.name for k in ['independent','qualification','development-search-closure','strength-preparation']):admit(path,'admissions/'+path.name)
 for sha,(name,data) in sources.items():entries['sources/'+sha+'/'+name]=data
 manifest=[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in sorted(entries.items())]
 with tarfile.open(output,'w:gz') as archive:
  for name,data in sorted(entries.items()):
   info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644;archive.addfile(info,io.BytesIO(data))
  raw=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(raw);info.mtime=0;archive.addfile(info,io.BytesIO(raw))
 receipt=base/'public-observations-archive-first.json';assert not receipt.exists();receipt.write_text(json.dumps({'time':time.time(),'archive':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size,'files':len(entries),'pinned_source_files':len(sources),'new_model_or_Jev_calls':0,'no_private_Git_or_Babel_history':True},indent=2)+'\n');print(receipt.read_text())
if __name__=='__main__':main()
