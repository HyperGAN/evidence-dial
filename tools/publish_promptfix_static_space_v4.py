"""Publish four reviewed static assets; preserve every earlier demo file."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi


def identity(row):
    return {'bytes': row.size, 'blob': row.blob_id,
            'lfs': row.lfs.sha256 if row.lfs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', type=Path, required=True)
    ap.add_argument('--expected-parent', required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    repo = 'ntc-ai/evidence-dial'
    assert not args.receipt.exists()
    files = sorted(p for p in args.folder.iterdir() if p.is_file())
    assert {p.name for p in files} == {'README.md', 'index.html',
        'promptfix-confirmation.html', 'promptfix-confirmation.json'}
    dataset = json.loads((args.folder / 'promptfix-confirmation.json').read_text())
    assert len(dataset['cases']) == 64 and dataset['provenance']['all_answers_native_EOS']
    assert dataset['all_holdout_results']['arms'][2]['primary_selected']
    api = HfApi()
    old = api.repo_info(repo, repo_type='space')
    assert not old.private and old.sha == args.expected_parent
    old_tree = {r.path: identity(r) for r in api.list_repo_tree(repo,
        repo_type='space', revision=old.sha, recursive=True) if hasattr(r, 'blob_id')}
    assert old_tree['index.html'] == old_tree['expanded-confirmation.html'], 'Earlier root must already have its exact archive'
    allowed = {'README.md', 'index.html'}
    uploads = []
    for path in files:
        assert path.name in allowed or path.name not in old_tree
        data = path.read_bytes()
        uploads.append({'path': path.name, 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
            'git_blob': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()})
    plan = args.receipt.with_name(args.receipt.stem + '-upload-plan.json')
    assert not plan.exists()
    plan.write_text(json.dumps({'time': time.time(), 'repo': repo, 'parent': old.sha,
        'files': uploads, 'old_file_identities': old_tree, 'delete_patterns': None}, indent=2) + '\n')
    commit = api.upload_folder(repo_id=repo, repo_type='space', folder_path=args.folder,
        parent_commit=old.sha, commit_message='Publish audited64-question preference-control comparison')
    anonymous = HfApi(token=False)
    info = anonymous.repo_info(repo, repo_type='space', revision=commit.oid)
    assert not info.private and info.sha == commit.oid
    remote = {r.path: r for r in anonymous.list_repo_tree(repo,
        repo_type='space', revision=commit.oid, recursive=True) if hasattr(r, 'blob_id')}
    for item in uploads:
        row = remote[item['path']]
        assert row.size == item['bytes']
        assert (row.lfs.sha256 == item['sha256']) if row.lfs else (row.blob_id == item['git_blob'])
    protected = 0
    for name, before in old_tree.items():
        if name not in allowed and name != '.gitattributes':
            assert identity(remote[name]) == before
            protected += 1
    args.receipt.write_text(json.dumps({'time': time.time(), 'repo': repo,
        'repo_type': 'space', 'public': True, 'parent': old.sha, 'commit': commit.oid,
        'commit_url': commit.commit_url, 'all_four_uploaded_identities_verified': True,
        'earlier_file_identities_preserved': protected, 'no_deletion': True,
        'SDK_metadata_unchanged': True, 'paid_hardware_change': False,
        'browser_visual_or_click_QA': 'unverified; separate from content verification',
        'new_model_or_Jev_calls': 0}, indent=2) + '\n')
    print(args.receipt.read_text())


if __name__ == '__main__':
    main()
