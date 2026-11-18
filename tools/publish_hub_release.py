"""Promote a reviewed artifact tree from an immutable Hub revision to main."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re

from huggingface_hub import CommitOperationCopy, HfApi
from huggingface_hub.hf_api import RepoFile


REPO = "ntc-ai/tiny-aya-particle-sliders"
SOURCE = "HyperGAN/tiny-aya-particle-sliders"
MUTABLE_FILES = {
    "dataset": {"README.md"},
    "model": {"README.md", "ARTICLE.md", "METHOD.md", "adapter-native.pt", "adapter-raw.pt",
              "selection.json", "winner.json"},
}
REVISION = re.compile(r"[0-9a-f]{40}")


def revision(value):
    if not isinstance(value, str) or not REVISION.fullmatch(value):
        raise ValueError("Hub revisions must be full lowercase commit SHAs")
    return value


def file_identity(entry):
    if entry.lfs is not None:
        return {"size": entry.size, "sha256": entry.lfs.sha256}
    return {"size": entry.size, "git_blob": entry.blob_id}


def tree(api, kind, sha):
    return {
        item.path: file_identity(item)
        for item in api.list_repo_tree(REPO, repo_type=kind, revision=sha, recursive=True)
        if isinstance(item, RepoFile)
    }


def reviewed_delta(kind, base, staged):
    removed = base.keys() - staged.keys()
    if removed:
        raise ValueError(f"Staged release removes existing paths: {sorted(removed)[:8]}")
    changes = {}
    for path, identity in staged.items():
        if base.get(path) == identity:
            continue
        normalized = PurePosixPath(path)
        if (
            normalized.is_absolute() or normalized.as_posix() != path or ".." in normalized.parts
            or any(part.startswith(".") for part in normalized.parts)
            or "\\" in path or any(ord(char) < 32 for char in path)
        ):
            raise ValueError(f"Unexpected staged path: {path!r}")
        if path in base and path not in MUTABLE_FILES[kind]:
            raise ValueError(f"Staged release overwrites a historical artifact: {path}")
        changes[path] = identity
    return changes


def main_revision(api, kind):
    info = api.repo_info(REPO, repo_type=kind, revision="main")
    if info.id != REPO or info.private:
        raise RuntimeError("The release target must be the canonical public repository")
    return revision(info.sha)


def make_manifest(api, kind, base, staged):
    base = revision(base)
    staged = revision(staged)
    old_tree = tree(api, kind, base)
    new_tree = old_tree if base == staged else tree(api, kind, staged)
    if not old_tree or not new_tree:
        raise ValueError("Both pinned revisions must contain artifacts")
    return {
        "base_revision": base,
        "staged_revision": staged,
        "changes": reviewed_delta(kind, old_tree, new_tree),
    }, new_tree


def release(api, kind, manifest, *, writer=None, source_sha=None):
    if set(manifest) != {"base_revision", "staged_revision", "changes"}:
        raise ValueError("Unexpected release manifest fields")
    reviewed, expected_tree = make_manifest(api, kind, manifest["base_revision"], manifest["staged_revision"])
    if reviewed != manifest:
        raise ValueError("The pinned delta does not match the reviewed release manifest")

    head = main_revision(api, kind)
    if head == manifest["staged_revision"] or (
        head != manifest["base_revision"] and tree(api, kind, head) == expected_tree
    ):
        return head, "already-current"
    if head != manifest["base_revision"]:
        raise ValueError(f"Stale Hub base: main is {head}; review a fresh release manifest")
    if not manifest["changes"]:
        return head, "already-current"
    if writer is None:
        return head, "ready"

    commit = writer.create_commit(
        REPO,
        repo_type=kind,
        revision="main",
        parent_commit=head,
        commit_message=f"Publish Tiny Aya {kind} from {source_sha}",
        commit_description=f"Reviewed Hub staging revision: {manifest['staged_revision']}",
        operations=[
            CommitOperationCopy(path, path, src_revision=manifest["staged_revision"])
            for path in sorted(manifest["changes"])
        ],
    )
    if tree(api, kind, commit.oid) != expected_tree or main_revision(api, kind) != commit.oid:
        raise RuntimeError(f"Published revision {commit.oid} did not pass full-tree and branch readback")
    return commit.oid, "published"


def github_writer(kind):
    if (
        os.getenv("GITHUB_ACTIONS") != "true" or os.getenv("GITHUB_REPOSITORY") != SOURCE
        or os.getenv("GITHUB_EVENT_NAME") != "push" or os.getenv("GITHUB_REF") != "refs/heads/main"
    ):
        raise ValueError("Publication requires a main push in the canonical GitHub repository")
    revision(os.environ["GITHUB_SHA"])
    token = os.environ["HF_TOKEN"]
    if not token.startswith("hf_jwt_"):
        raise ValueError("Publication requires a short-lived Hub Trusted Publisher token")
    writer = HfApi(endpoint="https://huggingface.co", token=token)
    writer.auth_check(REPO, repo_type=kind, write=True)
    print(f"Verified short-lived write authorization for {kind}/{REPO}", flush=True)
    return writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=MUTABLE_FILES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stage", metavar="HUB_SHA", help="Print the exact delta from current main to this revision")
    mode.add_argument("--apply", action="store_true", help="Publish the reviewed delta from the main workflow")
    args = parser.parse_args()
    api = HfApi(endpoint="https://huggingface.co", token=False)
    try:
        if args.stage:
            manifest, _ = make_manifest(api, args.kind, main_revision(api, args.kind), args.stage)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return
        path = Path(__file__).resolve().parents[1] / "releases" / f"huggingface-{args.kind}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        writer = github_writer(args.kind) if args.apply else None
        sha, status = release(api, args.kind, manifest, writer=writer, source_sha=os.getenv("GITHUB_SHA"))
        print(f"{args.kind}/{REPO} {status} {sha}; reviewed paths: {len(manifest['changes'])}")
    except (KeyError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
