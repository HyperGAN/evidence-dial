"""Prepare additive expanded-study HF artifacts after independent final admission.

Never uploads or changes an existing repository. Original mining and the earlier
37-pair study stay in the existing public dataset. Only explicit completed study
files and pinned source bodies are exported; private owner/Babel history is not.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
PREFIX = 'expanded138-h1024-v2'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--closure', type=Path, required=True)
    ap.add_argument('--holdout-audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    from evidence_dial.behavior_fullanswer_confirmation_v2 import BASE, pin, verified, read, validate_plan
    from evidence_dial.io import read_jsonl, write_json, write_jsonl
    closure = read(args.closure)
    assert closure['state'] == 'declared-two-rate-24-cell-DEV-search-complete'
    assert closure['training_pairs'] == 138 and closure['all_answers_native_EOS']
    audit = read(args.holdout_audit)
    assert audit['state'] == 'all-locked-behavior-holdout-arms-independently-admitted'
    assert audit['cases_each'] == 32 and not audit['holdout_based_reselection']
    plan = validate_plan(read(verified(audit['plan'])))
    assert plan['development_closure'] == pin(args.closure)
    dataset = args.out / 'dataset' / PREFIX
    model = args.out / 'model'
    manifest = []
    secret = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})')
    def safe(data):
        assert not secret.search(data), 'Credential-shaped data refused'
    def copy(source, destination):
        source = Path(source)
        data = source.read_bytes()
        if source.suffix in ['.json', '.jsonl', '.txt', '.md', '.py', '.log']:
            safe(data)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        destination.write_bytes(data)
        manifest.append({'path': str(destination.relative_to(args.out)), 'sha256': pin(destination)['sha256'],
            'bytes': len(data), 'source': str(source.resolve())})
    def document(destination, value):
        assert not destination.exists()
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, value)
    original = ROOT / 'artifacts/behavior-dial-20261114-v1'
    original_data = read(original / 'data/manifest-first.json')
    for split in ['train', 'dev']:
        copy(verified(original_data['files'][split]), dataset / 'questions' / f'{split}.jsonl')
    copy(verified(plan['data']), dataset / 'questions/confirmation32.jsonl')
    copy(BASE / 'fresh-confirmation-data/manifest-first.json', dataset / 'questions/confirmation-manifest.json')
    richer = ROOT / 'artifacts/behavior-all-passing-20261114-v1'
    copy(richer / 'cache/index.json', dataset / 'training-pairs/index.json')
    copy(richer / 'cache/responses.pt', dataset / 'training-pairs/all-passing-138.pt')
    copy(richer / 'independent-cache-audit-first.json', dataset / 'audits/expanded-cache-admission.json')
    copy(args.closure, dataset / 'audits/development-search-closure.json')
    copy(args.holdout_audit, dataset / 'audits/fresh-confirmation-comparison.json')
    copy(verified(audit['plan']), dataset / 'evaluation/confirmation/plan.json')
    copy(BASE / 'fullanswer-health-qualification-v3-first.json', dataset / 'audits/health-qualification-v3.json')
    copy(BASE / 'fullanswer-health-qualification-first.json', dataset / 'audits/prior-v2-fixture-setup-failure.json')
    weights = {}
    for chosen in closure['all_maximum_ties']:
        sha = chosen['artifact']['sha256']
        if sha in weights:
            continue
        target = model / PREFIX / 'checkpoints' / sha[:16]
        copy(verified(chosen['artifact']), target / 'adapter-native.pt')
        quarter = read(verified(chosen['quarter_admission']))
        copy(verified(quarter['RAW']), target / 'adapter-raw.pt')
        weights[sha] = {'native_file': str((target / 'adapter-native.pt').relative_to(model)),
            'raw_file': str((target / 'adapter-raw.pt').relative_to(model)), 'native_sha256': sha}
    representative = weights[plan['display_checkpoint']['sha256']]
    copy(model / representative['native_file'], model / 'adapter-native.pt')
    copy(model / representative['raw_file'], model / 'adapter-raw.pt')
    selection = {'selection': closure['selector'], 'all_maximum_ties': closure['all_maximum_ties'],
        'weight_artifacts': weights, 'display_native_sha256': plan['display_checkpoint']['sha256'],
        'training_pairs': 138, 'unique_TRAIN_questions': 37, 'two_cohorts_steps_each': 1024,
        'base_model': 'CohereLabs/tiny-aya-global', 'base_revision': 'af89d219b53ed9b13b8a4645f9c8028973510324',
        'weight_EMA': False, 'generation_safetycap': 4096, 'require_native_EOS': True,
        'holdout_based_reselection': False, 'old_model_in_previous_commit_and_checkpoint_folder': True}
    document(model / 'selection.json', selection)
    observations = {}
    source_bodies = {}
    def archive_file(path, name):
        path = Path(path)
        assert path.suffix in ['.json', '.jsonl', '.py', '.log']
        data = path.read_bytes()
        safe(data)
        assert name not in observations
        observations[name] = data
        if path.suffix == '.json':
            def find_sources(value):
                if isinstance(value, dict):
                    if {'path', 'sha256'} <= value.keys() and str(value['path']).endswith('.py'):
                        source = verified({'path': value['path'], 'sha256': value['sha256']})
                        body = source.read_bytes(); safe(body)
                        source_bodies[value['sha256']] = (source.name, body)
                    for child in value.values():
                        find_sources(child)
                elif isinstance(value, list):
                    for child in value:
                        find_sources(child)
            find_sources(json.loads(data))
    for gpu, cohort in enumerate(closure['cohorts']):
        folder = verified(cohort['plan']).parent
        train_plan = read(folder / 'plan.json')
        assert train_plan['gpu'] == gpu and train_plan['training_rows'] == 138
        target = dataset / 'training' / train_plan['cohort']
        for name in ['plan.json', 'completion-first.json', 'controller-first.json', 'controller-failed.json',
                'actual-tool-terminal-receipt-first.json', 'recovery-controller-first.json',
                'recovery-actual-tool-terminal-receipt-first.json']:
            copy(folder / name, target / name)
        for name in ['updates.jsonl', 'initialization.json', 'session.json', 'native-check.json']:
            copy(folder / 'run' / name, target / name)
        for step in [0, 256, 512, 768, 1024]:
            for kind in ['FULL', 'RAW', 'NATIVE']:
                copy(folder / 'run' / f'{kind}-{step:06d}.pt', target / f'{kind}-{step:06d}.pt')
        for admission in cohort['independent_quarters']:
            quarter = read(verified(admission))
            copy(verified(admission), dataset / 'audits' / f'GPU{gpu}-quarter{quarter["step"]}.json')
            for arm in quarter['arms']:
                dest = dataset / 'evaluation/dev' / f'GPU{gpu}-step{arm["step"]}-alpha{arm["strength"]:g}'
                copy(verified(arm['predictions']), dest / 'predictions.jsonl')
                copy(verified(arm['summary']), dest / 'summary.json')
        # Only this completed cohort is walked; no parent/private task state.
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ['.json', '.jsonl', '.log']:
                archive_file(path, f'training/GPU{gpu}/' + str(path.relative_to(folder)))
    controls_path = Path(closure['all24_arms'][0]['quarter_admission']['path'])
    controls = read(verified(read(controls_path)['control_admission']))
    for name in ['off', 'prompt-only']:
        for field, filename in [('predictions', 'predictions.jsonl'), ('summary', 'summary.json')]:
            copy(verified(controls['metrics'][name][field]), dataset / 'evaluation/dev' / name / filename)
    confirmation_folder = Path(audit['plan']['path']).parent
    combined = []
    for arm in audit['arms']:
        for field, filename in [('predictions', 'predictions.jsonl'), ('summary', 'summary.json')]:
            copy(verified(arm[field]), dataset / 'evaluation/confirmation' / arm['name'] / filename)
        combined += [{**row, 'condition': arm['name'], 'selected_on_DEV': arm['selected_on_DEV']}
            for row in read_jsonl(verified(arm['predictions']))]
    write_jsonl(dataset / 'evaluation/confirmation/combined.jsonl', combined)
    for path in sorted(confirmation_folder.rglob('*')):
        if path.is_file() and path.suffix in ['.json', '.jsonl', '.log']:
            archive_file(path, 'confirmation/' + str(path.relative_to(confirmation_folder)))
    controls_folder = Path(controls['metrics']['off']['predictions']['path']).parent.parent
    for path in sorted(controls_folder.rglob('*')):
        if path.is_file() and path.suffix in ['.json', '.jsonl', '.log']:
            archive_file(path, 'DEV-controls/' + str(path.relative_to(controls_folder)))
    for name in ['fullanswer-health-qualification-first.json', 'fullanswer-health-fixtures-first.json',
            'fullanswer-health-qualification-v3-first.json', 'fullanswer-health-fixtures-v3-first.json']:
        archive_file(BASE / name, 'admissions/' + name)
    for path in sorted((BASE / 'health-qualification-Jev-firsts').glob('*.json')):
        archive_file(path, 'health-qualification-firsts/' + path.name)
    archive_file(args.closure, 'admissions/development-search-closure.json')
    archive_file(args.holdout_audit, 'admissions/fresh-confirmation-comparison.json')
    for sha, (name, body) in source_bodies.items():
        observations[f'sources/{sha}/{name}'] = body
    archive = dataset / 'provenance/observations-and-sources.tar.gz'
    archive.parent.mkdir(parents=True)
    archive_manifest = [{'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
        for name, body in sorted(observations.items())]
    with tarfile.open(archive, 'w:gz') as handle:
        for name, body in sorted(observations.items()):
            info = tarfile.TarInfo(name); info.size = len(body); info.mode = 0o644; info.mtime = 0
            handle.addfile(info, io.BytesIO(body))
        body = (json.dumps(archive_manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('manifest.json'); info.size = len(body); info.mtime = 0
        handle.addfile(info, io.BytesIO(body))
    document(dataset / 'study-summary.json', {'training_pairs': 138, 'unique_training_questions': 37,
        'cohort_updates': [1024, 1024], 'all24_DEV_arms': closure['all24_arms'],
        'all_DEV_maximum_ties': closure['all_maximum_ties'], 'fresh_confirmation_arms': audit['arms'],
        'fresh_confirmation_comparisons': audit['comparisons'], 'generation_safetycap': 4096,
        'all_answers_native_EOS': True, 'weight_EMA': False, 'new_confirmation_questions': 32,
        'question_freeze_scope': 'After oldstudy/newcontrols; before reading any trained expanded-run DEVscore',
        'mining': 'Reuse all138 accepted original context pairs/37questions; original sampled curation cap128',
        'old_raw_firsts_and_operational_errors_preserved': True, 'holdout_based_reselection': False,
        'archive_files': len(observations), 'source_files': len(source_bodies)})
    files = [{'path': str(path.relative_to(args.out)), 'bytes': path.stat().st_size, 'sha256': pin(path)['sha256']}
        for path in sorted(args.out.rglob('*')) if path.is_file()]
    write_json(args.out / 'upload-manifest-first.json', {'time': time.time(), 'source': pin(__file__),
        'files': files, 'copy_sources': manifest, 'development_closure': pin(args.closure),
        'fresh_confirmation_audit': pin(args.holdout_audit), 'existing_public_repositories_must_be_preserved': True,
        'public_upload_performed': False, 'cards_still_required': True, 'new_model_or_Jev_calls': 0})
    print(json.dumps({'manifest': pin(args.out / 'upload-manifest-first.json'), 'files': len(files),
        'bytes': sum(row['bytes'] for row in files), 'archive_files': len(observations), 'cards_still_required': True,
        'new_model_or_Jev_calls': 0}))


if __name__ == '__main__':
    main()
