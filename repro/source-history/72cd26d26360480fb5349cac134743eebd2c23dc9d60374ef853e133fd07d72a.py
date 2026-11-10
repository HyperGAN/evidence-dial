"""Reuse the existing bounded text qualification; never rerate old firsts.

Run with the original Qwen observer's locked interpreter, after unloading Tiny
Aya. The private Evidence Dial GPU lease sequences this with every trainer.
"""
from dataclasses import asdict
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/particle-sliders/packages/particle-sliders-core/src"))
from evidence_dial.resources import Resources
from evidence_dial.io import write_json
from particle_sliders.candidate_failure import FileRecord, FinishedOutput, QualifiedDetector, ObservationCache, CandidateGuard
import text_observation_parser as observation_parser

ORIGINAL = Path('/ml2/particlesliders-working/lr-work/source-reviews/universal-failure-Qwen3VL8B-categorical-GPU1-source-v1')
QUALIFICATION = ROOT/'artifacts/observer-context-v4/qualification.json'
NATIVE = {"kind":"finished_text", "representation_class":"short_answer_or_prose",
    "intentional_degenerative_repetition":False,
    "allowed_output":"Any coherent short answer, number, option letter, abstention label or prose is allowed. Mistaken facts, failure to follow a grounding instruction, weak or absent effect, normal token-budget truncation and native early EOS are not catastrophic. Native early EOS may produce empty text. Only pervasive unrequested degenerative repetition or gibberish replacing meaningful text is catastrophic."}


def parse_first_raw(raw, opaque):
    return observation_parser.parse_observation(raw,opaque)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--candidate',type=Path,required=True)
    argument_parser.add_argument('--step',type=int,required=True)
    args = argument_parser.parse_args()
    out = args.candidate.resolve()
    launch_context = json.loads((ROOT/'configs/qwen-launch-context.json').read_text())
    assert all(os.environ.get(k)==v for k,v in launch_context.items()), 'Original qualified launch context required before framework imports'
    qualified = json.loads(QUALIFICATION.read_text())
    detector = QualifiedDetector(FileRecord(**qualified['detector_contract']), qualified['actual_instrument_source'],
        FileRecord.from_path(QUALIFICATION), judgment_kind='categorical')
    assert detector.qualification_error('text') is None
    FileRecord(**qualified['actual_instrument_source']['execution_adapter_source']).verify()
    profile = json.loads((ORIGINAL/'instrument-contract-v1.json').read_text())
    equivalence = json.loads((ROOT/'configs/observer-equivalence.json').read_text())
    original = (ORIGINAL/'worker.py').read_bytes()
    derivative = (ROOT/'tools/qwen_observer_runtime.py').read_bytes()
    assert hashlib.sha256(original).hexdigest() == equivalence['original_sha256']
    assert hashlib.sha256(derivative).hexdigest() == equivalence['derivative_sha256']
    for pin in [qualified['detector_contract'], qualified['actual_instrument_source']['worker_source'],
                profile['criteria'], qualified['actual_instrument_source']['actual_effective_decoding']]:
        FileRecord(**pin).verify()
    source = importlib.util.spec_from_file_location('evidence_dial_qwen_observer',ROOT/'tools/qwen_observer_runtime.py')
    observer = importlib.util.module_from_spec(source)
    source.loader.exec_module(observer)
    contract = json.loads((ORIGINAL/'execution-contract-v1.json').read_text())
    receipt_pin = qualified['actual_instrument_source']['model_download']
    model_receipt = FileRecord(**receipt_pin).read_json()
    criteria = Path(profile['criteria']['path']).read_text().strip()
    runtime = None
    with Resources(1, out/f'guard-resources-{args.step:06d}.json') as resource:
        def observe(file, media, native_contract):
            nonlocal runtime
            assert media == 'text' and native_contract == NATIVE
            if file.bytes == 0:
                # The existing qualification explicitly covers legal native-empty
                # metadata. This is not a perceptual or effect judgment.
                return {"status":"pass", "failure_probability":None, "categorical_failure":False,
                    "reason":"Native early EOS empty text is explicitly permitted by this contract",
                    "output_source":asdict(file), "detector_contract":asdict(detector.detector_contract),
                    "actual_instrument_source":detector.actual_instrument_source,
                    "route":"qualified_native_empty_contract_case_no_model_call"}
            opaque = 'text_' + file.sha256[:12]
            row = {"media_type":"text", "file":asdict(file), "opaque_id":opaque,
                "output_contract":json.dumps(NATIVE,sort_keys=True)}
            # ObservationCache commits before entering this callback. Persist raw
            # response before parsing so format errors retain their actual first.
            rawpath = ROOT/'artifacts/guard-firsts'/f'{file.sha256}.json'
            if rawpath.exists():
                # Explicit parser repair reuses the actual FIRST raw judgment.
                # It never asks the observer again to improve a verdict/format.
                retained=json.loads(rawpath.read_text())
                assert retained['output_source']['sha256']==file.sha256
                raw,audit=retained['raw_response'],retained['audit']
                reused=True
            else:
                if runtime is None:
                    runtime = observer.strict_runtime({'model_cache':contract['model_cache']}, contract, model_receipt, resource.check)
                raw,audit = observer.one_call(row,runtime,resource.check,criteria)
                write_json(rawpath, {"raw_response":raw,"audit":audit,"output_source":asdict(file),
                    "observer_equivalence":equivalence})
                reused=False
            parsed = parse_first_raw(raw,opaque)
            return {**observation_parser.verdict(parsed), "output_source":asdict(file),
                "detector_contract":asdict(detector.detector_contract),
                "actual_instrument_source":detector.actual_instrument_source,
                "raw_response":raw,"audit":audit,"retained_first_raw_reused":reused}
        cache = ObservationCache(ROOT/'artifacts/guard-observations',detector,observe)
        guard = CandidateGuard(out/'candidate-guard-v4.json',candidate_id=str(out),
            scientific_protocol=json.loads((out/'session.json').read_text()),media='text',cache=cache)
        probes = json.loads((out/f'probes-{args.step:06d}.json').read_text())['probes']
        def finished():
            for i,p in enumerate(probes):
                if not p['output'] and (not p['native']['output_ids'] or p['native']['output_ids'][-1] != 3):
                    raise ValueError('Empty text without native EOS is unmeasured')
                path = out/f'guard-text-{args.step:06d}-{i}.txt'
                if path.exists():
                    assert path.read_text() == p['output']
                else:
                    path.write_text(p['output'])
                yield FinishedOutput(FileRecord.from_path(path),'text',NATIVE)
        try:
            result = guard.check_checkpoint(FileRecord.from_path(out/f'FULL-{args.step:06d}.pt'), finished)
            write_json(out/f'guard-{args.step:06d}-v4.json',result)
        except Exception:
            state = json.loads((out/'candidate-guard-v4.json').read_text())
            write_json(out/f'guard-{args.step:06d}-v4.json',state)
            if state['status'] == 'failed':
                write_json(out/'candidate-failed.json',state)
            raise


if __name__ == '__main__':
    main()
