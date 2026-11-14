"""Admit exact checkpoints checked by one of the existing qualified observers.

CPU and both GPU instruments keep their original qualifications/firsts. This
only validates an actual passing receipt; it never infers or requests a verdict.
"""
from pathlib import Path
import json
from particle_sliders.candidate_failure import FileRecord, QualifiedDetector
from .benchmark import file_sha

ROOT = Path(__file__).resolve().parents[2]
QUALIFICATIONS = {
    'native-text-cpu-v2': 'artifacts/native-text-cpu-qualification-20261112-v1/adapter-envelope-qualification-v2.json',
    'native-text-gpu0-v1': 'artifacts/native-text-qualification-gpu0-v1/observer-qualification.json',
    'native-text-v1e': 'artifacts/observer-native-text-v1e/qualification.json',
}


def passing_guard(out, step, checkpoint):
    out, checkpoint = Path(out), Path(checkpoint)
    if (out / 'candidate-failed.json').exists() or list(out.glob('candidate-confirmed-failed-*.json')):
        raise RuntimeError('Retained qualified catastrophe stops candidate')
    sha = file_sha(checkpoint)
    for version, relative in QUALIFICATIONS.items():
        path = out / f'guard-{step:06d}-{version}.json'
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if value['status'] != 'pass':
            raise RuntimeError(f'Retained {version} check is not a pass; preserve unknown/failure')
        assert value['checkpoint']['sha256'] == sha, 'Guard binds a different checkpoint'
        FileRecord(**value['checkpoint']).verify()
        qualification = ROOT / relative
        q = json.loads(qualification.read_text())
        detector = QualifiedDetector(FileRecord(**q['detector_contract']), q['actual_instrument_source'],
            FileRecord.from_path(qualification), judgment_kind='categorical')
        assert q['qualified'] and q['accepted'] and detector.qualification_error('text') is None
        assert len(value['observations']) == 4, 'Require all four declared native output probes'
        probes = json.loads((out / f'probes-{step:06d}.json').read_text())['probes']
        assert len(probes) == 4
        for observation, probe in zip(value['observations'], probes):
            assert observation['status'] == 'pass'
            assert observation['binding']['actual_instrument_source'] == detector.actual_instrument_source
            actual = FileRecord(**observation['actual_qualification'])
            actual.verify()
            assert actual.sha256 == file_sha(qualification)
            output = FileRecord(**observation['binding']['output_source'])
            output.verify()
            assert Path(output.path).read_text() == probe['output']
            ids = probe['native']['output_ids']
            termination = 'native_eos' if ids and ids[-1] == 3 else 'max_new_tokens' if len(ids) == 64 else None
            assert termination is not None
            assert observation['binding']['native_contract']['generation_termination'] == termination
        for source in q['actual_instrument_source'].values():
            if isinstance(source, dict) and set(source) == {'path', 'bytes', 'sha256'}:
                FileRecord(**source).verify()
        return path
    return None
