"""Promote a reviewed artifact tree from an immutable Hub revision to main."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re

from huggingface_hub import CommitOperationCopy, HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import get_session, hf_raise_for_status


REPO = "ntc-ai/tiny-aya-particle-sliders"
SOURCE = "HyperGAN/tiny-aya-particle-sliders"
MUTABLE_FILES = {
    "dataset": {"README.md"},
    "model": {"README.md", "ARTICLE.md", "METHOD.md", "adapter-native.pt", "adapter-raw.pt",
              "selection.json", "winner.json"},
}
REPOSITORIES = {
    "model": (
        REPO, "ntc-ai/evidence-dial-aligntune", "ntc-ai/evidence-dial-aligntune-curated",
        "ntc-ai/evidence-dial-particle", "ntc-ai/evidence-dial-particle-hinted",
    ),
    "dataset": (
        REPO, "ntc-ai/evidence-dial-benchmark", "ntc-ai/evidence-dial-curated-successes",
        "ntc-ai/evidence-dial-grounding-tests", "ntc-ai/evidence-dial-training-archive",
    ),
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


def tree(api, kind, sha, repo=REPO):
    return {
        item.path: file_identity(item)
        for item in api.list_repo_tree(repo, repo_type=kind, revision=sha, recursive=True)
        if isinstance(item, RepoFile)
    }


def reviewed_delta(kind, base, staged, repo=REPO):
    removed = base.keys() - staged.keys()
    if removed:
        raise ValueError(f"Staged release removes existing paths: {sorted(removed)[:8]}")
    mutable = MUTABLE_FILES[kind] if repo == REPO else {"README.md", ".gitattributes"}
    changes = {}
    for path, identity in staged.items():
        if base.get(path) == identity:
            continue
        normalized = PurePosixPath(path)
        if (
            normalized.is_absolute() or normalized.as_posix() != path or ".." in normalized.parts
            or (path not in mutable and any(part.startswith(".") for part in normalized.parts))
            or "\\" in path or any(ord(char) < 32 for char in path)
        ):
            raise ValueError(f"Unexpected staged path: {path!r}")
        if path in base and path not in mutable:
            raise ValueError(f"Staged release overwrites a historical artifact: {path}")
        changes[path] = identity
    return changes


def main_revision(api, kind, repo=REPO):
    info = api.repo_info(repo, repo_type=kind, revision="main")
    if info.id != repo or info.private:
        raise RuntimeError("The release target must be the canonical public repository")
    return revision(info.sha)


def make_manifest(api, kind, base, staged, repo=REPO):
    base = revision(base)
    staged = revision(staged)
    old_tree = tree(api, kind, base, repo)
    new_tree = old_tree if base == staged else tree(api, kind, staged, repo)
    if not old_tree or not new_tree:
        raise ValueError("Both pinned revisions must contain artifacts")
    return {
        "base_revision": base,
        "staged_revision": staged,
        "changes": reviewed_delta(kind, old_tree, new_tree, repo),
    }, new_tree


def release(api, kind, manifest, *, writer=None, source_sha=None, repo=REPO):
    if repo not in REPOSITORIES[kind]:
        raise ValueError(f"The {kind} repository is outside the release allowlist")
    if set(manifest) != {"base_revision", "staged_revision", "changes"}:
        raise ValueError("Unexpected release manifest fields")
    reviewed, expected_tree = make_manifest(api, kind, manifest["base_revision"], manifest["staged_revision"], repo)
    if reviewed != manifest:
        raise ValueError("The pinned delta does not match the reviewed release manifest")

    head = main_revision(api, kind, repo)
    if head == manifest["staged_revision"] or (
        head != manifest["base_revision"] and tree(api, kind, head, repo) == expected_tree
    ):
        return head, "already-current"
    if head != manifest["base_revision"]:
        raise ValueError(f"Stale Hub base: main is {head}; review a fresh release manifest")
    if not manifest["changes"]:
        return head, "already-current"
    if writer is None:
        return head, "ready"

    commit = writer.create_commit(
        repo,
        repo_type=kind,
        revision="main",
        parent_commit=head,
        commit_message=f"Publish {kind} from GitHub {revision(source_sha)}",
        commit_description=f"Reviewed Hub staging revision: {manifest['staged_revision']}",
        operations=[
            CommitOperationCopy(path, path, src_revision=manifest["staged_revision"])
            for path in sorted(manifest["changes"])
        ],
    )
    if tree(api, kind, commit.oid, repo) != expected_tree or main_revision(api, kind, repo) != commit.oid:
        raise RuntimeError(f"Published revision {commit.oid} did not pass full-tree and branch readback")
    return commit.oid, "published"


def github_writer(kind, repo=REPO):
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
    # The Hub upload endpoint accepts repo publisher tokens and an empty file list.
    response = get_session().post(
        f"{writer.endpoint}/api/{kind}s/{repo}/preupload/main",
        headers={"Authorization": f"Bearer {token}"},
        json={"files": []},
        timeout=20,
    )
    hf_raise_for_status(response)
    preflight_head = revision(response.json()["commitOid"])
    print(f"Verified the upload preflight for {kind}/{repo} at {preflight_head}", flush=True)
    return writer


def targets():
    for kind, repos in REPOSITORIES.items():
        for repo in repos:
            yield {"kind": kind, "repo": repo, "resource": repo if kind == "model" else f"datasets/{repo}"}


def check_release(api, kind, repo, *, apply=False):
    if repo not in REPOSITORIES[kind]:
        raise ValueError(f"The {kind} repository is outside the release allowlist")
    suffix = kind if repo == REPO else repo.split("/", 1)[1]
    path = Path(__file__).resolve().parents[1] / "releases" / f"huggingface-{suffix}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    writer = github_writer(kind, repo) if apply else None
    sha, status = release(api, kind, manifest, writer=writer, source_sha=os.getenv("GITHUB_SHA"), repo=repo)
    print(f"{kind}/{repo} {status} {sha}; reviewed paths: {len(manifest['changes'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=MUTABLE_FILES, nargs="?")
    parser.add_argument("--repo", default=REPO, help="Canonical Tiny Aya or Evidence Dial Hub repository")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stage", metavar="HUB_SHA", help="Print the exact delta from current main to this revision")
    mode.add_argument("--apply", action="store_true", help="Publish the reviewed delta from the main workflow")
    mode.add_argument("--matrix", action="store_true", help="Print all publish targets for CI")
    mode.add_argument("--check-all", action="store_true", help="Validate each release manifest without publishing")
    args = parser.parse_args()
    if args.matrix:
        print(json.dumps({"include": list(targets())}, separators=(",", ":")))
        return
    if not args.check_all and (args.kind is None or args.repo not in REPOSITORIES[args.kind]):
        parser.error("Select a supported repository and its exact Hub kind")
    api = HfApi(endpoint="https://huggingface.co", token=False)
    try:
        if args.check_all:
            for target in targets():
                check_release(api, target["kind"], target["repo"])
        elif args.stage:
            head = main_revision(api, args.kind, args.repo)
            manifest, _ = make_manifest(api, args.kind, head, args.stage, args.repo)
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            check_release(api, args.kind, args.repo, apply=args.apply)
    except (KeyError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
