"""Explicit, hash-bound audits of two v5 verdicts outside the frozen contract.

This does not clear a confirmed catastrophe. Original failed files remain
immutable. An audited scope error becomes unmeasured; only a separate qualified
measurement may subsequently admit that checkpoint. No automatic repair flag.
"""
from pathlib import Path
import json
from dataclasses import asdict
from .benchmark import file_sha
from .io import write_json
from .protocol import digest

V5_SOURCE='0ccbca004c55b9936002568b5d44bacb02cfabe75c507a710055fe2c14aaea08'
CUTOFF_REASON='The output ends abruptly with a partial sentence and an incomplete document reference, lacking a coherent conclusion or answer to the question.'


def facts(failure: Path, probes: Path):
    from particle_sliders.candidate_failure import FileRecord
    state=json.loads(failure.read_text());native=json.loads(probes.read_text())['probes']
    assert state['status']=='failed'
    evidence=[]
    for checkpoint in state['checkpoints'].values():
        for observation in checkpoint.get('observations',[]):
            if observation['status']!='fail':continue
            binding=observation['binding']
            if binding['actual_instrument_source']['execution_adapter_source']['sha256']!=V5_SOURCE:
                raise ValueError('Only the explicitly identified v5 scope defect is covered')
            source=FileRecord(**binding['output_source']);source.verify()
            text=Path(source.path).read_text()
            p=next(p for p in native if p['output']==text)
            ids=p['native']['output_ids']
            allowed=binding['native_contract']['allowed_output']
            if text in {'UNKNOWN','CONFLICT'} and ids[-1]==3 and 'abstention label' in allowed:
                rule='explicitly_permitted_single_native_abstention_label'
            elif (len(ids)==64 and ids[-1]!=3 and observation['reason']==CUTOFF_REASON
                  and 'normal token-budget truncation' in allowed):
                rule='stated_failure_reason_is_only_explicitly_permitted_native_token_limit_and_missing_answer'
            else:
                raise ValueError('This flag has no proved native-contract scope contradiction; retain failure')
            evidence.append({'observation_sha256':digest(observation),'output_source':asdict(source),
                'native_output_ids':ids,'old_reason':observation['reason'],'scope_rule':rule})
    if not evidence:raise ValueError('No disputed flag to audit')
    return evidence


def record(run: Path, probes: Path, qualification: Path, *, replacement_run: Path | None=None):
    """Called explicitly for a reviewed counterexample, never by a trainer."""
    from particle_sliders.candidate_failure import FileRecord
    path=run/'guard-scope-adjudication-v1.json'
    if path.exists():raise FileExistsError('Scope audit already frozen')
    q=json.loads(qualification.read_text())
    assert q['qualified'] and q['accepted'] and q['covered_media']==['text']
    old=run/'candidate-failed.json'
    evidence=facts(old,probes)
    write_json(path,{'status':'old_instrument_scope_error_unmeasured','original_failure':asdict(FileRecord.from_path(old)),
        'probes':asdict(FileRecord.from_path(probes)),'evidence':evidence,
        'replacement_qualification':asdict(FileRecord.from_path(qualification)),
        'replacement_instrument_sha256':q['actual_instrument_source']['instrument_source']['sha256'],
        'replacement_guard_run':str((replacement_run or run).resolve()),
        'effect_scores_used':False,'old_records_modified':False,'automatic_health_inference':False,
        'meaning':'The stated v5 reason contradicts its frozen allowed native domain. Original first verdict retained; no qualified catastrophe is established by that reason. A new qualified check is still mandatory.'})


def admit(run: Path, *, observe_saved=False, checkpoint: Path | None=None):
    from particle_sliders.candidate_failure import FileRecord
    if list(run.glob('candidate-confirmed-failed-*.json')):
        raise ValueError('A retained replacement-instrument catastrophe cannot be overridden by a scope audit')
    failed=run/'candidate-failed.json'
    if not failed.exists():return
    path=run/'guard-scope-adjudication-v1.json'
    if not path.exists():raise ValueError('Retained failed candidate cannot generate, resume or promote')
    audit=json.loads(path.read_text())
    assert audit['status']=='old_instrument_scope_error_unmeasured' and not audit['automatic_health_inference']
    original=FileRecord(**audit['original_failure']);original.verify()
    assert Path(original.path).resolve()==failed.resolve()
    probes=FileRecord(**audit['probes']);probes.verify()
    assert facts(failed,Path(probes.path))==audit['evidence']
    qualification=FileRecord(**audit['replacement_qualification']).read_json()
    assert qualification['qualified'] and qualification['accepted']
    assert qualification['actual_instrument_source']['instrument_source']['sha256']==audit['replacement_instrument_sha256']
    if observe_saved:return
    root=Path(__file__).resolve().parents[2]
    active=json.loads((root/'configs/active-observer.json').read_text())
    replacement=json.loads((root/active['qualification_path']).read_text())
    if replacement['actual_instrument_source']['instrument_source']['sha256']!=audit['replacement_instrument_sha256']:
        raise ValueError('A different observer cannot silently admit this audit')
    for target in {run.resolve(),Path(audit['replacement_guard_run']).resolve()}:
        if list(target.glob('candidate-confirmed-failed-*.json')):
            raise ValueError('A retained replacement-instrument catastrophe remains failed')
        current=json.loads((target/f'candidate-guard-{active["version"]}.json').read_text())
        if current['status']!='pass':raise ValueError('Scope error is unmeasured until the replacement checkpoint check passes')
        if current['binding']['actual_instrument_source']!=replacement['actual_instrument_source']:
            raise ValueError('Replacement passing state belongs to a different instrument')
        latest=FileRecord(**current['latest_checkpoint']);latest.verify()
        if target==run.resolve() and checkpoint and latest.sha256!=file_sha(checkpoint):
            raise ValueError('Replacement guard must bind this exact checkpoint')
