"""Publish GitHub assets from an exact, reviewed draft."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

REPOSITORY = "HyperGAN/tiny-aya-particle-sliders"
RELEASES = f"repos/{REPOSITORY}/releases"
TAG = re.compile(r"lexsi-video-v[1-9][0-9]*")
ASSET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def request_from(path):
    request = json.loads(path.read_bytes())
    fields = {"tag", "release_id", "source_commit", "prerelease", "files"} | {"media"}
    if not isinstance(request, dict) or request.keys() != fields:
        raise ValueError("Unexpected release request fields")
    if not isinstance(request["tag"], str) or TAG.fullmatch(request["tag"]) is None:
        raise ValueError("Expected an established release tag")
    if not isinstance(request["source_commit"], str) or COMMIT.fullmatch(request["source_commit"]) is None:
        raise ValueError("Expected the full source commit")
    if type(request["release_id"]) is not int or request["release_id"] <= 0 or type(request["prerelease"]) is not bool:
        raise ValueError("Expected a numeric release ID and a prerelease boolean")
    files = request["files"]
    if not isinstance(files, dict):
        raise TypeError("Expected reviewed file records")
    for name, record in files.items():
        if (ASSET_NAME.fullmatch(name) is None or not isinstance(record, dict)
                or record.keys() != {"bytes", "sha256"} or type(record["bytes"]) is not int
                or record["bytes"] <= 0 or not isinstance(record["sha256"], str)
                or SHA256.fullmatch(record["sha256"]) is None):
            raise ValueError(f"Invalid file record for {name}")
    validate_manifest(request)
    return request


def output(*args, input=None):
    return subprocess.check_output(args, input=input, timeout=60)


def gh_json(*args, payload=None):
    if payload is None:
        return json.loads(output("gh", "api", *args))
    return json.loads(output("gh", "api", *args, "--input", "-", input=json.dumps(payload).encode()))


def verify_source(source):
    if output("git", "cat-file", "-t", source).strip() != b"commit":
        raise ValueError("Release source must be a commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"], check=True)


def verify_tag(request, required=False):
    ref = f"refs/tags/{request['tag']}"
    result = output("git", "ls-remote", f"https://github.com/{REPOSITORY}.git", ref, f"{ref}^{{}}")
    refs = {}
    for line in result.decode().splitlines():
        sha, name = line.split("\t")
        refs[name] = sha
    current = refs.get(f"{ref}^{{}}", refs.get(ref))
    if (current is not None or required) and current != request["source_commit"]:
        raise ValueError("Release tag does not point to the reviewed source commit")
    return current


def verified_record(request, destination=None):
    release_id = request["release_id"]
    record = gh_json(f"{RELEASES}/{release_id}")
    if (record["id"] != release_id or record["tag_name"] != request["tag"]
            or record["prerelease"] != request["prerelease"]
            or (record["draft"] and record["target_commitish"] != request["source_commit"])):
        raise ValueError("GitHub release differs from the reviewed tag, target or release type")
    pages = output(
        "gh", "api", "--paginate", f"{RELEASES}/{release_id}/assets?per_page=100",
        "--jq", ".[] | @json",
    )
    assets = {}
    for line in pages.splitlines():
        asset = json.loads(line)
        name = asset["name"]
        spec = request["files"].get(name)
        if (name in assets or spec is None or asset["state"] != "uploaded"
                or asset["size"] != spec["bytes"] or asset["digest"] != f"sha256:{spec['sha256']}"):
            raise ValueError(f"GitHub asset differs from the reviewed file {name}")
        assets[name] = asset["id"]
    if assets.keys() != request["files"].keys():
        raise ValueError("GitHub assets do not cover the reviewed files")
    if destination is not None:
        for name, asset_id in assets.items():
            download_asset(asset_id, destination / name, request["files"][name])
        verify_payload(request, destination)
    return record


def download_asset(asset_id, path, spec):
    with path.open("wb") as file:
        subprocess.run(
            ["gh", "api", "-H", "Accept: application/octet-stream", f"{RELEASES}/assets/{asset_id}"],
            stdout=file, check=True, timeout=240,
        )
    with path.open("rb") as file:
        size, digest = path.stat().st_size, hashlib.file_digest(file, "sha256").hexdigest()
    if size != spec["bytes"] or digest != spec["sha256"]:
        raise ValueError(f"Downloaded bytes differ from the reviewed file {path.name}")


def validate_manifest(request):
    files = request["files"]
    if len(files) != 1 or not next(iter(files)).endswith(".mp4"):
        raise ValueError("Expected one reviewed MP4 file")
    media = request["media"]
    if (not isinstance(media, dict) or media.keys() != {"width", "height", "duration_ms"}
            or any(type(value) is not int or value <= 0 for value in media.values())):
        raise ValueError("Expected positive reviewed dimensions and duration")


def verify_payload(request, directory):
    path = directory / next(iter(request["files"]))
    env = {name: value for name, value in os.environ.items() if name not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    options = ["-v", "error", "-protocol_whitelist", "file,pipe"]
    data = json.loads(subprocess.check_output([
        "ffprobe", *options, "-show_entries",
        "format=format_name,duration:stream=codec_type,width,height", "-of", "json", str(path),
    ], env=env, timeout=60))
    streams = data["streams"]
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    if len(video) != 1 or not any(stream["codec_type"] == "audio" for stream in streams):
        raise ValueError("Expected one video track and a narration track")
    container = data["format"]
    media = {
        "width": video[0]["width"],
        "height": video[0]["height"],
        "duration_ms": round(float(container["duration"]) * 1000),
    }
    if "mp4" not in container["format_name"].split(",") or media != request["media"]:
        raise ValueError("The MP4 differs from the reviewed dimensions or duration")
    subprocess.run([
        "ffmpeg", *options, "-xerror", "-nostdin", "-i", str(path),
        "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
    ], env=env, check=True, timeout=180)


def inspected_release(request):
    with tempfile.TemporaryDirectory(prefix="github-release-") as directory:
        record = verified_record(request, Path(directory))
    current = verify_tag(request, required=not record["draft"])
    return record, current


def publish(request):
    record, current = inspected_release(request)
    if not record["draft"]:
        return record["html_url"]
    if current is None:
        gh_json(
            "--method", "POST", f"repos/{REPOSITORY}/git/refs",
            payload={"ref": f"refs/tags/{request['tag']}", "sha": request["source_commit"]},
        )
    verify_tag(request, required=True)
    gh_json(
        "--method", "PATCH", f"{RELEASES}/{request['release_id']}",
        payload={"draft": False, "prerelease": request["prerelease"],
                 "make_latest": "false" if request["prerelease"] else "legacy"},
    )
    published = verified_record(request)
    verify_tag(request, required=True)
    if published["draft"]:
        raise ValueError("The reviewed release is still a draft")
    return published["html_url"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validate", "audit", "publish"))
    parser.add_argument("--request", type=Path, default=Path("release.json"))
    args = parser.parse_args()
    request = request_from(args.request)
    verify_source(request["source_commit"])
    if args.operation == "audit":
        record, _ = inspected_release(request)
        print(f"Verified {record['tag_name']} and {len(request['files'])} attached files")
    if args.operation == "publish":
        if (os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
                or os.environ.get("GITHUB_EVENT_NAME") != "push"
                or os.environ.get("GITHUB_REF") != "refs/heads/main"
                or output("git", "rev-parse", "HEAD").decode().strip() != os.environ.get("GITHUB_SHA")):
            raise ValueError("Publication requires a push to the release repository's main branch")
        print(publish(request))


if __name__ == "__main__":
    main()
