import io
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
from urllib.parse import unquote, urlsplit


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import publish_space as space


class VerifyLiveTests(unittest.TestCase):
    def test_waits_for_a_stale_secondary_asset(self):
        api = Mock()
        api.space_info.return_value.sha = "test-revision"
        api.get_space_runtime.return_value = SimpleNamespace(stage="RUNNING")
        files = {
            "README.md": b"Space metadata",
            "index.html": b"<head></head>",
            "explorer.css": b"body {}",
            "promptfix-confirmation.json": b"[]",
            "grounding.html": b"<head></head><p>Reviewed page</p>",
        }
        served = {**files, "grounding.html": b"<head></head><p>Older page</p>"}
        script = (
            b'<script>window.huggingface={variables:'
            b'{"SPACE_CREATOR_USER_ID":"abcdef012345"}};</script>'
        )

        def fetch(url, *, timeout):
            name = unquote(urlsplit(url).path.removeprefix("/"))
            self.assertNotEqual(name, "README.md")
            return io.BytesIO(served[name])

        def publish_reviewed_page(seconds):
            served["grounding.html"] = files["grounding.html"].replace(b"<head>", b"<head>" + script)

        with (patch.object(space, "urlopen", side_effect=fetch) as requests,
              patch.object(space.time, "sleep", side_effect=publish_reviewed_page) as wait):
            space.verify_live(api, "test-revision", files)

        wait.assert_called_once()
        checked = [urlsplit(call.args[0]).path for call in requests.call_args_list]
        self.assertEqual(checked.count("/grounding.html"), 2)
        self.assertEqual(set(checked), {"/" + name for name in files if name != "README.md"})


if __name__ == "__main__":
    unittest.main()
