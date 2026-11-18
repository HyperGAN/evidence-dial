#!/usr/bin/env python3
"""Publish the reviewed Tiny Aya demo files to the existing static Space."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.hf_api import RepoFile
import yaml

SPACE = "ntc-ai/tiny-aya-particle-sliders"
HOST = "https://ntc-ai-tiny-aya-particle-sliders.static.hf.space"
DEMO = Path(__file__).resolve().parents[1] / "demo"
MANIFEST = Path(__file__).with_name("space-files.txt")
PROBES = ("index.html", "explorer.css", "promptfix-confirmation.json")
THUMBNAIL = rb"https://cdn-thumbnails\.huggingface\.co/social-thumbnails/spaces/ntc-ai/tiny-aya-particle-sliders\.png"
SPACE_SCRIPT = re.compile(
    rb'(?<=<head>)<script>window\.huggingface=\{variables:'
    rb'\{"SPACE_CREATOR_USER_ID":"[0-9a-f]+"\}\};</script>'
)
SPACE_METADATA = re.compile(
    rb'<meta property="og:title" content="[^"]*">'
    rb'<meta property="og:description" content="[^"]*">'
    rb'<meta property="og:image" content="' + THUMBNAIL + rb'">'
    rb'<meta name="twitter:card" content="summary_large_image">'
    rb'<meta name="twitter:image" content="' + THUMBNAIL + rb'">(?=</head>)'
)


def payload():
    files = {}
    for name in MANIFEST.read_text(encoding="utf-8").splitlines():
        path = PurePosixPath(name)
        source = DEMO / path
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or name in files
            or source.resolve() != source
            or not source.is_file()
        ):
            raise ValueError(f"Invalid Space file: {name!r}")
        files[name] = source.read_bytes()
    if not {"README.md", *PROBES} <= files.keys():
        raise ValueError("Space manifest is missing a required file")
    preamble, opening, body = files["README.md"].decode("utf-8").partition("---\n")
    header, closing, _ = body.partition("\n---\n")
    if preamble or not opening or not closing:
        raise ValueError("Space README is missing its YAML frontmatter")
    fields = yaml.safe_load(header)
    if (
        not isinstance(fields, dict)
        or fields.get("sdk") != "static"
        or fields.get("app_file") != "index.html"
        or "app_build_command" in fields
    ):
        raise ValueError("The Space must remain a raw static HTML app")
    return files


def blob_id(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def remote_files(api, revision):
    entries = api.list_repo_tree(SPACE, repo_type="space", revision=revision, recursive=True)
    return {item.path: item.blob_id for item in entries if isinstance(item, RepoFile)}


def verify_live(api, revision, files):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if api.space_info(SPACE).sha != revision:
            raise RuntimeError("Space head changed while verifying the release")
        if api.get_space_runtime(SPACE).stage == "RUNNING":
            try:
                for name in PROBES:
                    url = f"{HOST}/{quote(name)}?release={revision}"
                    with urlopen(url, timeout=12) as response:
                        received = response.read()
                    if name == "index.html":
                        # The static host injects platform metadata into the page head.
                        received = SPACE_METADATA.sub(b"", SPACE_SCRIPT.sub(b"", received, count=1), count=1)
                    if received != files[name]:
                        break
                else:
                    print(f"Verified live Space revision {revision}")
                    return
            except (URLError, TimeoutError):
                pass
        time.sleep(5)
    raise RuntimeError("The live Space did not serve the reviewed files within 180 seconds")


def deploy(files, *, apply):
    public = HfApi(endpoint="https://huggingface.co", token=False)
    info = public.space_info(SPACE)
    if info.host != HOST or info.sdk != "static" or info.private:
        raise RuntimeError("Unexpected Space configuration")
    old = remote_files(public, info.sha)
    expected = {name: blob_id(data) for name, data in files.items()}
    changed = [name for name, oid in expected.items() if old.get(name) != oid]
    print(f"Space {info.sha}: {len(changed)} changed files out of {len(files)}")
    if changed:
        print("\n".join(changed))
    revision = info.sha
    if apply and changed:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        latest = subprocess.check_output(
            ["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"], text=True
        ).split()[0]
        if source != os.environ["GITHUB_SHA"] or source != latest:
            raise RuntimeError("The tested revision is no longer the GitHub main head")
        token = os.environ.get("HF_SPACE_WRITE_TOKEN")
        if not token:
            raise RuntimeError("Set the Space write token in huggingface-space-production")
        writer = HfApi(endpoint="https://huggingface.co", token=token)
        written = writer.create_commit(
            SPACE,
            repo_type="space",
            revision="main",
            parent_commit=revision,
            commit_message=f"Publish GitHub revision {source}",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=files[name])
                for name in changed
            ],
        )
        revision = written.oid
        installed = remote_files(public, revision)
        if any(installed.get(name) != oid for name, oid in (old | expected).items()):
            raise RuntimeError("Published Space files differ from the expected tree")
    if apply or not changed:
        verify_live(public, revision, files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Commit the tested payload")
    options = parser.parse_args()
    deploy(payload(), apply=options.apply)
