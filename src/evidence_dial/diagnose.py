"""Training-target response errors, never used by finished-output selection."""
from pathlib import Path
import json
from .io import write_json
from .benchmark import file_sha


def response_error(artifact: Path, cache: Path, out: Path, strengths, gpu=1):
    import os
    import subprocess
    # This helper also runs directly, outside the CLI's CUDA mapping.
    # Bind the actual physical device before importing/loading CUDA models.
    os.environ['CUDA_VISIBLE_DEVICES']=subprocess.check_output(
        ['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    import torch
    from particle_sliders.inference import open_inference
    from .resources import Resources
    if out.exists():raise FileExistsError('Completed diagnostics are retained, not replayed')
    from .guard_scope import admit
    admit(artifact.parent,checkpoint=artifact.parent/'FULL-000128.pt')
    rows=torch.load(cache/'responses.pt',map_location='cpu',weights_only=False)
    index=json.loads((cache/'index.json').read_text())['rows']
    assert len(rows)==len(index)
    records=[]
    with Resources(gpu,out.parent/(out.stem+'-resources.json')) as resource:
        with open_inference(artifact,options={},operation='response') as run:
            native=run.binding.response.__self__
            with torch.inference_mode():
                for strength in strengths:
                    for row,meta in zip(rows,index):
                        resource.check()
                        generated=native._response(row.row.neutral_ids,scale=float(strength)).float().cpu()
                        reference=row.target.float().cpu()
                        neutral=row.neutral.float().cpu()
                        # Full native2048 coordinates, no learned projection.
                        records.append({'id':meta['id'],'position':meta['position'],'strength':strength,
                            'squared_error_sum':float((generated-reference).square().sum()),
                            'neutral_squared_error_sum':float((neutral-reference).square().sum()),
                            'coordinates':reference.numel()})
    results={}
    for strength in strengths:
        subset=[r for r in records if r['strength']==strength]
        sse=sum(r['squared_error_sum'] for r in subset)
        off=sum(r['neutral_squared_error_sum'] for r in subset)
        results[str(strength)]={'native_mse':sse/sum(r['coordinates'] for r in subset),
                              'relative_teacher_sse':sse/off,'rows':len(subset)}
    write_json(out,{'diagnostic_only':True,'used_for_selection':False,'training_cases_only':len({r['id'] for r in index}),
                   'artifact_sha256':file_sha(artifact),'cache_sha256':file_sha(cache/'responses.pt'),
                   'results':results,'rows':records})
