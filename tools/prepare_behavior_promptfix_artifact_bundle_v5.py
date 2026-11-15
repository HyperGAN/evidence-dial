"""Prepare an additive publication bundle from fully admitted study results.

This never uploads. It exports explicit scientific files, not the private
worktree, coordination history, credentials or unrelated experiments.
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
sys.path.insert(0, str(ROOT / 'tools'))
from confirm_behavior_promptfix_v4 import BASE, DIAG, OUT, pin, read, validate
PREFIX = 'promptfix37-h1024-v4'


def main():
    from evidence_dial.io import read_jsonl, write_json, write_jsonl
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    closure_path = BASE / 'development-search-closure-first.json'
    audit_path = OUT / 'independent-completion-audit-first.json'
    closure, audit = read(closure_path), read(audit_path)
    assert closure['state'] == 'promptfix-two-rate-24-cell-DEV-search-complete'
    assert closure['training_pairs'] == 37 and len(closure['all24_arms']) == 24
    assert audit['state'] == 'all-locked-fresh64-full-EOS-answers-and-first-judgments-independently-audited'
    assert audit['all_answers_native_EOS'] and not audit['new_test_reselection']
    plan = validate(read(audit['plan']['path']))
    assert pin(audit['plan']['path']) == audit['plan']
    assert pin(closure_path) in plan['inputs']
    qualification_path = BASE / 'pending-confirmation-cache-namespace-v5-qualification-first.json'
    qualification = read(qualification_path)
    assert qualification['state'] == 'pending-confirmation-cache-namespace-v5-offline-qualified'
    assert qualification['qualified_wrapper'] in plan['sources']
    assert pin(qualification['qualified_wrapper']['path']) == qualification['qualified_wrapper']
    assert read(qualification['actual_terminal']['path'])['exit_code'] == 0
    dataset, model = args.out / 'dataset' / PREFIX, args.out / 'model'
    copied, observations, source_bodies = [], {}, {}
    secret = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})')

    def verified(item):
        assert pin(item['path']) == item, item['path']
        return Path(item['path'])

    def safe(body):
        assert not secret.search(body), 'Credential-shaped content refused'

    def copy(source, destination):
        source = Path(source)
        body = source.read_bytes()
        if source.suffix in ['.json', '.jsonl', '.md', '.py', '.log', '.txt']:
            safe(body)
        assert not destination.exists()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        copied.append({'path': str(destination.relative_to(args.out)), 'bytes': len(body),
            'sha256': hashlib.sha256(body).hexdigest(), 'source': str(source.resolve())})
        if source.suffix == '.json':
            sources(json.loads(body))

    def sources(value):
        if isinstance(value, dict):
            if {'path', 'sha256'} <= value.keys() and str(value['path']).endswith('.py'):
                source = verified({'path': value['path'], 'sha256': value['sha256']})
                body = source.read_bytes()
                safe(body)
                source_bodies[value['sha256']] = (source.name, body)
            for child in value.values():
                sources(child)
        elif isinstance(value, list):
            for child in value:
                sources(child)

    def archive(source, name):
        source = Path(source)
        assert source.suffix in ['.json', '.jsonl', '.log', '.py'] and name not in observations
        assert '.babel-reports' not in source.parts and '.git' not in source.parts
        body = source.read_bytes()
        safe(body)
        observations[name] = body
        if source.suffix == '.json':
            sources(json.loads(body))

    # Publish the qualified cache-namespace fix alongside its offline evidence.
    # These fixtures replay DEV captures and are never confirmation answers.
    qualification_root = BASE / 'confirmation-offline-qualification-v6-first'
    copy(qualification_path, dataset / 'audits/confirmation-cache-namespace-v5-qualification.json')
    copy(BASE / 'pending-confirmation-cache-namespace-derivative-v5-first.json',
         dataset / 'audits/confirmation-cache-namespace-v5-source-derivative.json')
    for name in ['checks-passed-first.json', 'declaration-first.json',
                 'actual-tool-terminal-receipt-first.json', 'resources-first.completed.json']:
        copy(qualification_root / name, dataset / 'audits/offline-confirmation-qualification' / name)
    for attempt in ['confirmation-offline-qualification-first', 'confirmation-offline-qualification-v5-first']:
        root = BASE / attempt
        for name in ['declaration-first.json', 'actual-tool-terminal-receipt-first.json',
                     'resources-first.completed.json']:
            archive(root / name, 'offline-qualification-attempts/' + attempt + '/' + name)
    for name in ['confirmation-offline-qualification-console-first.log',
                 'confirmation-offline-qualification-v5-console-first.log',
                 'confirmation-offline-qualification-v6-console-first.log']:
        archive(BASE / name, 'offline-qualification-attempts/' + name)

    original = ROOT / 'artifacts/behavior-dial-20261114-v1'
    for split in ['train', 'dev']:
        copy(original / 'data' / (split + '.jsonl'), dataset / 'questions' / (split + '.jsonl'))
    copy(verified(plan['data']), dataset / 'questions/confirmation64.jsonl')
    copy(DIAG / 'new-confirmation-data/manifest-first.json', dataset / 'questions/confirmation-manifest.json')
    copy(BASE / 'cache/index.json', dataset / 'training-pairs/index.json')
    copy(BASE / 'cache/responses.pt', dataset / 'training-pairs/approved37-first-states.pt')
    copy(closure_path, dataset / 'audits/development-search-closure.json')
    copy(audit_path, dataset / 'audits/fresh-confirmation-comparison.json')
    copy(verified(audit['plan']), dataset / 'evaluation/confirmation/plan.json')
    copy(BASE / 'independent-cache-audit-first.json', dataset / 'audits/training-cache-admission.json')
    copy(DIAG / 'independent-audit-first.json', dataset / 'audits/prompt-setup-ablation.json')
    previous = ROOT / 'artifacts/behavior-expanded-h1024-20261114-v2'
    copy(previous / 'fullanswer-health-qualification-v3-first.json', dataset / 'audits/reused-health-qualification-v3.json')
    for gpu in [0, 1]:
        shard = BASE / f'GPU{gpu}'
        for path in sorted(shard.rglob('*')):
            if path.is_file() and path.suffix in ['.json', '.jsonl', '.log']:
                archive(path, f'mining/GPU{gpu}/' + str(path.relative_to(shard)))
            elif path.is_file() and path.name == 'prefill-first.pt':
                copy(path, dataset / f'mining-states/GPU{gpu}' / path.relative_to(shard))
    weights = {}
    for candidate in closure['all_maximum_ties']:
        sha = candidate['artifact']['sha256']
        if sha in weights:
            continue
        destination = model / PREFIX / 'checkpoints' / sha[:16]
        copy(verified(candidate['artifact']), destination / 'adapter-native.pt')
        quarter = read(verified(candidate['quarter_admission']))
        copy(verified(quarter['RAW']), destination / 'adapter-raw.pt')
        weights[sha] = {'native_file': str((destination / 'adapter-native.pt').relative_to(model)),
            'raw_file': str((destination / 'adapter-raw.pt').relative_to(model)), 'native_sha256': sha}
    representative = weights[plan['display_checkpoint']['sha256']]
    copy(model / representative['native_file'], model / 'adapter-native.pt')
    copy(model / representative['raw_file'], model / 'adapter-raw.pt')
    write_json(model / 'selection.json', {'selector': closure['selector'],
        'all_maximum_ties': closure['all_maximum_ties'], 'weight_artifacts': weights,
        'display_native_sha256': plan['display_checkpoint']['sha256'], 'display_strength': plan['display_strength'],
        'display_rule': closure['display_rule'], 'training_pairs': 37, 'unique_TRAIN_questions': 37,
        'two_cohorts_steps_each': 1024, 'base_model': 'CohereLabs/tiny-aya-global',
        'base_revision': 'af89d219b53ed9b13b8a4645f9c8028973510324', 'weight_EMA': False,
        'generation_safetycap': 4096, 'require_native_EOS': True, 'test_based_reselection': False,
        'all_previous_artifacts_preserved': True})
    for gpu, cohort in enumerate(closure['cohorts']):
        folder = verified(cohort['plan']).parent
        training_plan = read(folder / 'plan.json')
        assert training_plan['gpu'] == gpu and training_plan['training_rows'] == 37
        target = dataset / 'training' / training_plan['cohort']
        for name in ['plan.json', 'completion-first.json', 'controller-first.json', 'actual-tool-terminal-receipt-first.json']:
            copy(folder / name, target / name)
        for name in ['updates.jsonl', 'initialization.json', 'session.json', 'native-check.json']:
            copy(folder / 'run' / name, target / name)
        for step in [0, 256, 512, 768, 1024]:
            for kind in ['FULL', 'RAW', 'NATIVE']:
                copy(folder / 'run' / f'{kind}-{step:06d}.pt', target / f'{kind}-{step:06d}.pt')
        if gpu == 0:
            copy(verified(training_plan['original_FULL0']), dataset / 'reference/original-initial-FULL0.pt')
        for admission in cohort['quarters']:
            quarter = read(verified(admission))
            copy(verified(admission), dataset / 'audits' / f'GPU{gpu}-quarter{quarter["step"]}.json')
            for arm in quarter['arms']:
                destination = dataset / 'evaluation/dev' / f'GPU{gpu}-step{arm["step"]}-alpha{arm["strength"]:g}'
                copy(verified(arm['predictions']), destination / 'predictions.jsonl')
                copy(verified(arm['summary']), destination / 'summary.json')
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ['.json', '.jsonl', '.log']:
                archive(path, f'training/GPU{gpu}/' + str(path.relative_to(folder)))
    diagnostic = read(DIAG / 'independent-audit-first.json')
    for control, name in [('off', 'off'), ('prompt-only', 'target0-unrestricted')]:
        summary = next(a for a in diagnostic['arms'] if a['name'] == name)
        folder = verified(summary['predictions']).parent
        copy(folder / 'predictions.jsonl', dataset / 'evaluation/dev' / control / 'predictions.jsonl')
        copy(folder / 'summary.json', dataset / 'evaluation/dev' / control / 'summary.json')
    combined = []
    for arm in audit['arms']:
        for field, name in [('predictions', 'predictions.jsonl'), ('summary', 'summary.json')]:
            copy(verified(arm[field]), dataset / 'evaluation/confirmation' / arm['name'] / name)
        combined += [{**r, 'condition': arm['name'], 'selected_on_DEV': arm['primary_selected']}
            for r in read_jsonl(verified(arm['predictions']))]
    write_jsonl(dataset / 'evaluation/confirmation/combined.jsonl', combined)
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.suffix in ['.json', '.jsonl', '.log']:
            archive(path, 'confirmation/' + str(path.relative_to(OUT)))
    for name in ['declaration-first.json', 'training-declaration-first.json', 'independent-cache-audit-first.json',
                 'runtime-source-derivatives-first.json', 'quarter-audit-source-derivative-first.json',
                 'TRAIN-curation-statistics-first.json']:
        archive(BASE / name, 'admissions/' + name)
    archive(closure_path, 'admissions/development-search-closure.json')
    archive(audit_path, 'admissions/fresh-confirmation-comparison.json')
    for path in sorted(DIAG.glob('*first.json')):
        archive(path, 'prompt-diagnosis/' + path.name)
    for sha, (name, body) in source_bodies.items():
        observations[f'sources/{sha}/{name}'] = body
    destination = dataset / 'provenance/observations-and-sources.tar.gz'
    destination.parent.mkdir(parents=True)
    archive_manifest = [{'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
        for name, body in sorted(observations.items())]
    with tarfile.open(destination, 'w:gz') as handle:
        for name, body in sorted(observations.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(body), 0o644, 0
            handle.addfile(info, io.BytesIO(body))
        body = (json.dumps(archive_manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('manifest.json')
        info.size, info.mtime = len(body), 0
        handle.addfile(info, io.BytesIO(body))
    write_json(dataset / 'study-summary.json', {'training_pairs': 37, 'unique_training_questions': 37,
        'cohort_updates': [1024, 1024], 'all24_DEV_arms': closure['all24_arms'],
        'all_DEV_maximum_ties': closure['all_maximum_ties'], 'fresh_confirmation_arms': audit['arms'],
        'fresh_confirmation_comparisons': audit['comparisons'], 'generation_safetycap': 4096,
        'all_answers_native_EOS': True, 'weight_EMA': False, 'new_confirmation_questions': 64,
        'question_freeze_scope': 'Before new PG training; after prompt development had begun; broad families overlap',
        'mining': '256 fullEOS answers: OFF+3 unrestricted instructions per64 TRAIN questions; one best qualifying context per question',
        'archive_files': len(observations), 'source_files': len(source_bodies),
        'per_update_FULL_checkpoints': 'All preserved locally; public bundle contains0/256/512/768/1024 milestones and complete update journals',
        'auxiliary111_contexts_used_in_these_models': False, 'test_based_reselection': False})
    files = [{'path': str(p.relative_to(args.out)), 'bytes': p.stat().st_size, 'sha256': pin(p)['sha256']}
        for p in sorted(args.out.rglob('*')) if p.is_file()]
    write_json(args.out / 'upload-manifest-first.json', {'time': time.time(), 'source': pin(__file__),
        'files': files, 'copy_sources': copied, 'development_closure': pin(closure_path),
        'fresh_confirmation_audit': pin(audit_path), 'public_upload_performed': False,
        'existing_public_repositories_must_be_preserved': True, 'cards_still_required': True,
        'new_model_or_Jev_calls': 0})
    print(json.dumps({'manifest': pin(args.out / 'upload-manifest-first.json'), 'files': len(files),
        'bytes': sum(f['bytes'] for f in files), 'archive_files': len(observations), 'cards_still_required': True}))


if __name__ == '__main__':
    main()
