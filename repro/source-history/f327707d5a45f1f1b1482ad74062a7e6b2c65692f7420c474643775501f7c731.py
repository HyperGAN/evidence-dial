"""Qualified native-text checks; old instrument states are never overwritten."""
from pathlib import Path
from dataclasses import asdict
import argparse
import importlib.util
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'vendor/particle-sliders/packages/particle-sliders-core/src'))
from evidence_dial.io import write_json
from evidence_dial.resources import Resources
from evidence_dial.protocol import digest
from evidence_dial.guard_scope import admit
from particle_sliders.candidate_failure import FileRecord,FinishedOutput,QualifiedDetector,ObservationCache,CandidateGuard,CandidateFailure
import native_text_instrument as instrument
import text_observation_parser as parser


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--step',type=int,required=True);ap.add_argument('--checkpoint',type=Path)
    args=ap.parse_args();out=args.candidate.resolve()
    admit(out,observe_saved=True)
    context=json.loads((ROOT/'configs/qwen-launch-context.json').read_text())
    assert all(os.environ.get(k)==v for k,v in context.items())
    active=json.loads((ROOT/'configs/active-observer.json').read_text())
    qualification=ROOT/active['qualification_path']
    q=json.loads(qualification.read_text())
    detector=QualifiedDetector(FileRecord(**q['detector_contract']),q['actual_instrument_source'],
                               FileRecord.from_path(qualification),judgment_kind='categorical')
    assert detector.qualification_error('text') is None
    for pin in q['actual_instrument_source'].values():
        if isinstance(pin,dict) and set(pin)=={'path','bytes','sha256'}:FileRecord(**pin).verify()
    fixed=json.loads((ROOT/'configs/jev-native-text-guard-v1.json').read_text())
    domain=fixed['native_domain']
    original=Path('/ml2/particlesliders-working/lr-work/source-reviews/universal-failure-Qwen3VL8B-categorical-GPU1-source-v1')
    contract=json.loads((original/'execution-contract-v1.json').read_text())
    model_receipt=json.loads(Path('/mnt/nvme/qualification/metrics/qwen3-vl-8b-0c351dd0/Root-model-download-receipt-v1.json').read_text())
    spec=importlib.util.spec_from_file_location('native_guard_runtime',ROOT/'tools/qwen_observer_runtime.py')
    observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    runtime=None
    with Resources(1,out/f'guard-resources-{args.step:06d}-{active["version"]}.json') as resource:
        def observe(file,media,native_contract):
            nonlocal runtime
            assert media=='text' and native_contract['allowed_domain']==domain
            text=Path(file.path).read_text();termination=native_contract['generation_termination']
            if not text:
                assert termination=='native_eos'
                return {'status':'pass','categorical_failure':False,'failure_probability':None,
                    'reason':'Explicit legal native early EOS, verified by native token IDs',
                    'route':'native_domain_metadata_no_semantic_call','output_source':asdict(file),
                    'detector_contract':asdict(detector.detector_contract),
                    'actual_instrument_source':detector.actual_instrument_source}
            key=digest({'text':text,'termination':termination,'domain':domain,
                        'instrument':{k:q['actual_instrument_source'][k] for k in [
                            'instrument_source','actual_native_qualification','native_fixture_manifest','parser',
                            'runtime_source','environment','model_download','actual_effective_decoding','kind']}})
            first=ROOT/'artifacts/native-text-guard-firsts'/f'{key}.json'
            reused=first.exists()
            if first.exists():record=json.loads(first.read_text())
            else:
                record=None
                # An identical retained qualification input is already its
                # actual first observation. Reuse it, without its expected label.
                for fixture in fixed['fixtures']:
                    if fixture['finished_text']==text and fixture['termination']==termination:
                        path=ROOT/'artifacts/native-text-qualification-v1'/f"{fixture['id']}.json"
                        retained=json.loads(path.read_text())
                        record={'raw_response':retained['raw_response'],'audit':retained['audit'],
                                'origin':'retained identical qualification observation','source':asdict(FileRecord.from_path(path))}
                        reused=True
                        break
                if record is None:
                    if runtime is None:runtime=observer.strict_runtime({'model_cache':contract['model_cache']},contract,model_receipt,resource.check)
                    raw,audit=instrument.one_call(text,termination,domain,runtime,resource.check)
                    record={'raw_response':raw,'audit':audit,'origin':'first candidate observation under this new instrument'}
                write_json(first,record)
            assert record['audit']['native_text_sha256']==file.sha256
            parsed=parser.parse_observation(record['raw_response'],record['audit']['opaque_id'])
            return {**parser.verdict(parsed),'raw_response':record,'first_observation_reused':reused,
                    'output_source':asdict(file),'detector_contract':asdict(detector.detector_contract),
                    'actual_instrument_source':detector.actual_instrument_source}
        cache=ObservationCache(ROOT/'artifacts/native-text-guard-observations',detector,observe)
        guard_path=out/f'candidate-guard-{active["version"]}.json'
        guard=CandidateGuard(guard_path,candidate_id=str(out),scientific_protocol=json.loads((out/'session.json').read_text()),media='text',cache=cache)
        probes=json.loads((out/f'probes-{args.step:06d}.json').read_text())['probes']
        def finished():
            for i,p in enumerate(probes):
                ids=p['native']['output_ids']
                termination='native_eos' if ids and ids[-1]==3 else 'max_new_tokens' if len(ids)==64 else None
                if termination is None:raise ValueError('Native termination is unmeasured or outside frozen decoding')
                path=out/f'guard-text-{args.step:06d}-{i}.txt'
                if path.exists():assert path.read_text()==p['output']
                else:path.write_text(p['output'])
                native={'kind':'finished_text','representation_class':'short_answer_or_prose',
                        'intentional_degenerative_repetition':False,'allowed_domain':domain,
                        'generation_termination':termination}
                yield FinishedOutput(FileRecord.from_path(path),'text',native)
        try:
            checkpoint=args.checkpoint.resolve() if args.checkpoint else out/f'FULL-{args.step:06d}.pt'
            result=guard.check_checkpoint(FileRecord.from_path(checkpoint),finished)
            write_json(out/f'guard-{args.step:06d}-{active["version"]}.json',result)
        except Exception:
            state=json.loads(guard_path.read_text())
            write_json(out/f'guard-{args.step:06d}-{active["version"]}.json',state)
            if state['status']=='failed':
                permanent=out/'candidate-failed.json'
                if permanent.exists():permanent=out/f'candidate-confirmed-failed-{active["version"]}.json'
                write_json(permanent,state)
            raise


if __name__=='__main__':main()
