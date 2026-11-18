import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

script = Path(__file__).resolve().parents[1] / "tools/publish_github_release.py"
spec = importlib.util.spec_from_file_location("approved_release", script)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def file_record(path):
    with path.open("rb") as file:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(file, "sha256").hexdigest()}


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.source = "a" * 40
        self.request = {"tag": "lexsi-video-v8", "release_id": 42, "source_commit": self.source, "prerelease": False}
        self.draft = {"draft": True, "html_url": "https://example.com/release"}
        (self.path / "entry.mp4").write_bytes(b"video fixture")
        self.request["media"] = {"width": 1920, "height": 1080, "duration_ms": 101100}
        self.request["files"] = {file.name: file_record(file) for file in self.path.iterdir()}

    def test_request_matches_reviewed_shape(self):
        request = self.path / "release.json"
        request.write_text(json.dumps(self.request))
        self.assertEqual(release.request_from(request), self.request)

    def test_publish_creates_checked_tag_and_rechecks_release(self):
        with (
            patch.object(release, "inspected_release", return_value=(self.draft, None)),
            patch.object(release, "verified_record", return_value=dict(self.draft, draft=False)),
            patch.object(release, "verify_tag") as ref,
            patch.object(release, "gh_json") as api,
        ):
            self.assertEqual(release.publish(self.request), self.draft["html_url"])
        create_tag, publish = api.call_args_list
        self.assertEqual(create_tag.args[1], "POST")
        self.assertEqual(create_tag.kwargs["payload"]["sha"], self.source)
        self.assertEqual(publish.args[1], "PATCH")
        self.assertFalse(publish.kwargs["payload"]["draft"])
        self.assertEqual(ref.call_count, 2)

    def test_existing_release_is_idempotent(self):
        current = dict(self.draft, draft=False)
        with (
            patch.object(release, "inspected_release", return_value=(current, self.source)),
            patch.object(release, "gh_json") as api,
        ):
            self.assertEqual(release.publish(self.request), current["html_url"])
        api.assert_not_called()

    def test_invalid_payload_never_reaches_a_write(self):
        with (
            patch.object(release, "inspected_release", side_effect=ValueError("Invalid payload")),
            patch.object(release, "gh_json") as api,
            self.assertRaisesRegex(ValueError, "Invalid payload"),
        ):
            release.publish(self.request)
        api.assert_not_called()

    def test_media_probe_and_decode(self):
        report = {"format": {"format_name": "mov,mp4", "duration": "101.1"}, "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080}, {"codec_type": "audio"},
        ]}
        with (
            patch.object(release.subprocess, "check_output", return_value=json.dumps(report).encode()) as probe,
            patch.object(release.subprocess, "run") as decode,
            patch.dict(release.os.environ, {"GH_TOKEN": "fixture"}),
        ):
            release.verify_payload(self.request, self.path)
        self.assertEqual(probe.call_args.args[0][0], "ffprobe")
        self.assertEqual(decode.call_args.args[0][0], "ffmpeg")
        self.assertNotIn("GH_TOKEN", decode.call_args.kwargs["env"])

    def test_media_mismatch_stops_before_decode(self):
        report = {"format": {"format_name": "mp4", "duration": "5"}, "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080}, {"codec_type": "audio"},
        ]}
        with (
            patch.object(release.subprocess, "check_output", return_value=json.dumps(report).encode()),
            patch.object(release.subprocess, "run") as decode,
            self.assertRaisesRegex(ValueError, "reviewed dimensions or duration"),
        ):
            release.verify_payload(self.request, self.path)
        decode.assert_not_called()
