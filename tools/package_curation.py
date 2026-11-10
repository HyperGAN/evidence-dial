"""Package retained successful-sample evidence, flattening the shared caches."""
from pathlib import Path
import argparse,hashlib,io,json,tarfile
from archive_run import SECRET

ROOT=Path(__file__).resolve().parents[1]


def package(out):
    if out.exists():raise FileExistsError('Completed publication package must be reused')
    pool=ROOT/'artifacts/curation-success-v4';cache=ROOT/'artifacts/cache/curated-success-v1'
    selected=json.loads((pool/'selected.json').read_text())
    index=json.loads((cache/'index.json').read_text())
    assert selected['complete'] and selected['minimum_coverage_passed']
    assert selected['cases']==index['cases']==48 and len(index['rows'])==118
    assert hashlib.sha256((pool/'selected.json').read_bytes()).hexdigest()==index['identity']['selected_sha256']
    for row in selected['selection']:
        generation=pool/'generations'/(row['generation_key']+'.json')
        trace=generation.with_suffix('.safetensors')
        assert hashlib.sha256(trace.read_bytes()).hexdigest()==row['trajectory_sha256']
        judge=json.loads((pool/'jev-cache'/(row['jev_key']+'.json')).read_text())
        assert judge['http_status']==200 and judge['response']['model']=='typesafe/jev-1.14'
    files={}
    def add(path,arc):
        if arc in files:assert files[arc].resolve()==path.resolve()
        else:files[arc]=path
    for name in ['selected.json','identity.json','progress.json']:
        add(pool/name,'artifacts/curation-success-v4/'+name)
    for directory in ['qualification','decisions','generations','jev-cache']:
        # rglob starts from the resolved directory: intermediate symlinks must
        # not silently discard the physically retained shared caches.
        source=(pool/directory).resolve()
        assert ROOT in source.parents
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.safetensors'}:
                add(path,'artifacts/curation-success-v4/'+directory+'/'+str(path.relative_to(source)))
    for path in sorted(cache.rglob('*')):
        if path.is_file() and path.suffix in {'.json','.pt'}:
            add(path,'artifacts/cache/curated-success-v1/'+str(path.relative_to(cache)))
    for name in ['recipe-v2.json','historical-method-audit.json','parser-amendment-v2.json']:
        path=ROOT/'artifacts/method-repair-v1'/name
        add(path,str(path.relative_to(ROOT)))
    out.parent.mkdir(parents=True,exist_ok=True);manifest=[]
    with tarfile.open(out,'w:gz',compresslevel=3) as archive:
        for arc,path in sorted(files.items()):
            raw=path.read_bytes()
            if SECRET.search(raw):raise ValueError(f'Credential-shaped bytes refused in {arc}')
            info=tarfile.TarInfo(arc);info.size=len(raw);info.mode=0o644;info.mtime=int(path.stat().st_mtime)
            archive.addfile(info,io.BytesIO(raw))
            manifest.append({'path':arc,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        body=json.dumps({'files':manifest,'symlinks_flattened':True,'base_weights_included':False,
            'FaithEval_passages_included':False,'model_calls':0,'source_cases':64,'accepted_cases':48,
            'paired_positions':118,'raw_generations':455,'current_parser_decisions':444,
            'note':'Eleven earlier generations remain retained although the amended parser found an earlier accepted attempt for their case.'},indent=2).encode()
        info=tarfile.TarInfo('CURATION-MANIFEST.json');info.size=len(body);info.mode=0o644
        archive.addfile(info,io.BytesIO(body))
    with tarfile.open(out,'r:gz') as archive:
        actual={m.name:m for m in archive.getmembers() if m.name!='CURATION-MANIFEST.json'}
        assert len(actual)==len(manifest)
        for record in manifest:
            member=actual[record['path']];assert member.isfile() and not member.issym()
            raw=archive.extractfile(member).read()
            assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
    receipt={'file':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'files':len(manifest),'readback_all_file_hashes_verified':True,'symlinks_flattened':True,
        'model_or_API_calls':0}
    out.with_suffix(out.suffix+'.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    package(p.parse_args().out)
