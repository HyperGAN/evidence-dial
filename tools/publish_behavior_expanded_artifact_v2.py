"""Publish a reviewed additive expanded-study tree with an exact parent guard."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from huggingface_hub import HfApi

PREFIX = 'expanded138-h1024-v2/'


def identity(row):
    return {'bytes': row.size, 'blob': row.blob_id,
        'lfs': row.lfs.sha256 if row.lfs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', type=Path, required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--repo-type', choices=['model', 'dataset'], required=True)
    ap.add_argument('--expected-parent', required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    assert (args.repo, args.repo_type) in [
        ('ntc-ai/evidence-dial-behavior-particle', 'model'), ('ntc-ai/evidence-dial-behavior', 'dataset')]
    assert not args.receipt.exists()
    api = HfApi()
    old = api.repo_info(args.repo, repo_type=args.repo_type)
    assert old.sha == args.expected_parent and not old.private
    old_tree = {row.path: identity(row) for row in api.list_repo_tree(args.repo,
        repo_type=args.repo_type, revision=old.sha, recursive=True) if hasattr(row, 'blob_id')}
    allowed_replace = {'README.md'}
    if args.repo_type == 'model':
        allowed_replace |= {'adapter-native.pt', 'adapter-raw.pt', 'selection.json'}
    files = []
    for path in sorted(args.folder.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(args.folder))
        assert '.git' not in path.parts and '.babel-reports' not in path.parts
        assert relative.startswith(PREFIX) or relative in allowed_replace
        if relative not in allowed_replace:
            assert relative not in old_tree, 'Refuse replacing historical experiment data'
        data = path.read_bytes()
        if path.suffix in ['.json', '.jsonl', '.txt', '.md', '.py']:
            assert not re.search(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,})', data)
        files.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'git_blob': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()})
    assert files and any(row['path'] == 'README.md' for row in files), 'Reviewed updated card required'
    upload_plan = args.receipt.with_name(args.receipt.stem + '-upload-plan.json')
    assert not upload_plan.exists()
    upload_plan.parent.mkdir(parents=True, exist_ok=True)
    upload_plan.write_text(json.dumps({'time': time.time(), 'repo': args.repo, 'repo_type': args.repo_type,
        'parent': old.sha, 'files': files, 'old_file_identities': old_tree,
        'allowed_root_replacements': sorted(allowed_replace), 'delete_patterns': None}, indent=2) + '\n')
    commit = api.upload_folder(repo_id=args.repo, repo_type=args.repo_type, folder_path=args.folder,
        parent_commit=old.sha, commit_message='Publish audited expanded138-context full-answer1024-update study')
    anonymous = HfApi(token=False)
    info = anonymous.repo_info(args.repo, repo_type=args.repo_type, revision=commit.oid)
    assert info.sha == commit.oid and not info.private
    new_tree = {row.path: row for row in anonymous.list_repo_tree(args.repo,
        repo_type=args.repo_type, revision=commit.oid, recursive=True) if hasattr(row, 'blob_id')}
    for row in files:
        remote = new_tree[row['path']]
        assert remote.size == row['bytes']
        if remote.lfs:
            assert remote.lfs.sha256 == row['sha256']
        else:
            assert remote.blob_id == row['git_blob']
    protected = 0
    for name, old_identity in old_tree.items():
        if name not in allowed_replace and name != '.gitattributes':
            assert identity(new_tree[name]) == old_identity
            protected += 1
    args.receipt.write_text(json.dumps({'time': time.time(), 'repo': args.repo, 'repo_type': args.repo_type,
        'public': True, 'commit': commit.oid, 'parent': old.sha, 'commit_url': commit.commit_url,
        'every_uploaded_Git_or_LFS_content_identity_verified': True, 'files_verified': len(files),
        'old_file_identities_preserved': protected, 'no_deletion': True,
        'upload_plan': str(upload_plan.resolve()), 'upload_plan_sha256': hashlib.sha256(upload_plan.read_bytes()).hexdigest(),
        'new_model_or_Jev_calls': 0}, indent=2) + '\n')
    print(args.receipt.read_text())


if __name__ == '__main__':
    main()
