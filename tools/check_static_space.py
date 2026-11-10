"""Verify public static assets and the exact platform HTML insertions."""
from pathlib import Path
import argparse,hashlib,json,re
import requests
from huggingface_hub import HfApi


def check(repo,folder,out):
    api=HfApi();info=api.space_info(repo);runtime=api.get_space_runtime(repo)
    assert info.sdk=='static' and not info.private and runtime.stage=='RUNNING'
    records=[]
    for name in ['index.html','benchmark.json','README.md']:
        local=(folder/name).read_bytes()
        raw=requests.get(f'https://huggingface.co/spaces/{repo}/resolve/{info.sha}/{name}',timeout=45)
        raw.raise_for_status();assert raw.content==local
        entry={'file':name,'bytes':len(local),'repository_sha256':hashlib.sha256(local).hexdigest(),
               'anonymous_repository_bytes_verified':True}
        if name!='README.md':
            response=requests.get(info.host+'/'+name,timeout=45);response.raise_for_status()
            entry['served_sha256']=hashlib.sha256(response.content).hexdigest()
            if name=='index.html':
                html=response.content.decode('utf-8')
                html,n=re.subn(r'<script>window\.huggingface=\{variables:.*?</script>','',html,count=1)
                assert n==1
                html,n=re.subn(r'<meta (?:property="og:(?:title|description|image)"|name="twitter:(?:card|image)") content="[^"]*">','',html)
                assert n==5 and html.encode()==local
                entry['served_HTML_matches_after_only_platform_head_insertions']=True
            else:
                assert response.content==local;entry['served_bytes_identical']=True
        records.append(entry)
    data=json.loads((folder/'benchmark.json').read_text())
    value={'repository':repo,'commit':info.sha,'url':info.host,'sdk':info.sdk,'private':info.private,
        'runtime_stage':runtime.stage,'hardware':runtime.hardware,'files':records,
        'cases':len(data['cases']),'aggregate_rows':len(data['aggregates']),
        'real_browser_visual_QA':False,'model_or_API_inference_calls':0}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['repository','commit','url','runtime_stage','cases','aggregate_rows']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();check(a.repo,a.folder,a.out)
