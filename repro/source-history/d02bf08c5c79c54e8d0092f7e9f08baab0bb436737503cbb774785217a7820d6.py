"""Create a clean publication tree without private research history or inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]


def export(dest):
    dest=Path(dest).resolve()
    if dest==ROOT or ROOT in dest.parents:raise ValueError('Use a separate publication checkout')
    files=[]
    for directory in ['src','tests','tools','configs','docs','demo','artifacts/evidence','artifacts/data/policy-v1','artifacts/data/evidence-interventions-v1','artifacts/data/balanced-grounding-v2','artifacts/data/balanced-grounding-confirm-v1','artifacts/data/policy-curated-success-v1']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(ROOT/p for p in ['README.md','pyproject.toml','.gitignore','LICENSE'] if (ROOT/p).is_file())
    files.extend(ROOT/p for p in ['artifacts/selection-v1.json','artifacts/holdout-plan-v1.json',
        'artifacts/followup-tests-plan-v2.json','artifacts/release-index-v2.json'] if (ROOT/p).is_file())
    files.extend(ROOT/'artifacts/method-repair-v1'/name for name in [
        'historical-method-audit.json','recipe.json','recipe-v2.json','parser-amendment-v2.json',
        'confirmation-plan-v1.json','confirmation-plan-v2.json','confirmation-plan-v3.json',
        'selection-v1.json','evaluation-plan-v1.json','release-index-v1.json']
        if (ROOT/'artifacts/method-repair-v1'/name).is_file())
    files.extend(ROOT/'artifacts/curation-success-v4'/name for name in [
        'selected.json','qualification/fixtures.json','qualification/result.json',
        'qualification/conservative-filter-admission.json']
        if (ROOT/'artifacts/curation-success-v4'/name).is_file())
    files.extend(p for p in (ROOT/'artifacts/data/faith-v1').glob('*.json*') if p.name!='holdout.jsonl')
    files.extend(p for p in (ROOT/'publication/repro').rglob('*') if p.is_file())
    manifest=[]
    for source in files:
        rel=source.relative_to(ROOT)
        if rel.parts[:2]==('publication','repro'):rel=Path('repro')/source.relative_to(ROOT/'publication/repro')
        content=source.read_bytes()
        if re.search(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})',content):
            raise ValueError(f'Credential-shaped content refused in {rel}')
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
        manifest.append({'path':str(rel),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    (dest/'publication-manifest.json').write_text(json.dumps({'files':manifest,
        'research_history_included':False,'FaithEval_full_passages_included':False,
        'source_inputs_reproducible_from_pinned_upstream_revisions':True},indent=2)+'\n')
    print(json.dumps({'directory':str(dest),'files':len(files)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    export(p.parse_args().out)
