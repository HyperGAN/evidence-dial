"""Publish one explicit artifact tree and verify every remote Git/LFS content identity."""
import argparse,hashlib,json,re,time
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);ap.add_argument('--repo',required=True);ap.add_argument('--repo-type',choices=['model','dataset'],required=True);ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args()
 assert a.repo in ['ntc-ai/evidence-dial-behavior-particle','ntc-ai/evidence-dial-behavior']
 assert not a.receipt.exists();api=HfApi();files=[]
 for p in sorted(a.folder.rglob('*')):
  if not p.is_file():continue
  assert '.git' not in p.parts and '.babel-reports' not in p.parts
  raw=p.read_bytes()
  if p.suffix in ['.json','.jsonl','.txt','.md','.py']:
   assert not re.search(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})',raw),p.name
  files.append({'path':str(p.relative_to(a.folder)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()})
 assert files
 plan=a.receipt.with_name(a.receipt.stem+'-upload-plan.json');assert not plan.exists();plan.write_text(json.dumps({'time':time.time(),'repo':a.repo,'type':a.repo_type,'files':files,'public':True},indent=2)+'\n')
 try:
  old=api.repo_info(a.repo,repo_type=a.repo_type);existing=api.list_repo_files(a.repo,repo_type=a.repo_type);assert set(existing)<=set(['.gitattributes']),'Refuse to replace an existing populated repository'
 except RepositoryNotFoundError:
  api.create_repo(a.repo,repo_type=a.repo_type,private=False);old=api.repo_info(a.repo,repo_type=a.repo_type)
 assert not old.private
 commit=api.upload_folder(repo_id=a.repo,repo_type=a.repo_type,folder_path=a.folder,commit_message='Publish audited preference-alignment experiment and complete records',parent_commit=old.sha)
 info=api.repo_info(a.repo,repo_type=a.repo_type,revision=commit.oid);assert not info.private and info.sha==commit.oid
 remote={r.path:r for r in api.list_repo_tree(a.repo,repo_type=a.repo_type,revision=commit.oid,recursive=True) if hasattr(r,'blob_id')}
 for row in files:
  r=remote[row['path']];assert r.size==row['bytes']
  if r.lfs:assert r.lfs.sha256==row['sha256']
  else:assert r.blob_id==row['git_blob']
 a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps({'time':time.time(),'repo':a.repo,'repo_type':a.repo_type,'public':not info.private,'commit':commit.oid,'commit_url':commit.commit_url,'files_verified':len(files),'bytes_verified':sum(r['bytes'] for r in files),'verification':'Every remote Git blob identity or LFS SHA256 and byte length compared with frozen upload plan','upload_plan':str(plan),'upload_plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'no_new_model_or_Jev_calls':True},indent=2)+'\n');print(a.receipt.read_text())
if __name__=='__main__':main()
