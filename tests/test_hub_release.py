import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import publish_hub_release as hub


BASE, STAGED, PUBLISHED = "a" * 40, "b" * 40, "c" * 40
README = {"size": 100, "git_blob": "d" * 40}
WEIGHT = {"size": 250, "sha256": "e" * 64}


class HubReleaseTests(unittest.TestCase):
    def setUp(self):
        self.base = {"README.md": README, "study-v1/weights.pt": WEIGHT}
        self.changes = {"study-v2/weights.pt": WEIGHT}
        self.staged = self.base | self.changes
        self.manifest = {"base_revision": BASE, "staged_revision": STAGED, "changes": self.changes}
        self.api, self.writer = Mock(), Mock()
        self.api.repo_info.return_value = SimpleNamespace(id=hub.REPO, sha=BASE, private=False)

    def use_trees(self, **extra):
        trees = {BASE: self.base, STAGED: self.staged, **extra}
        return patch.object(hub, "tree", side_effect=lambda api, kind, sha: trees[sha])

    def test_read_only_validation(self):
        with self.use_trees():
            self.assertEqual(hub.release(self.api, "model", self.manifest), (BASE, "ready"))

    def test_reject_deletion_historical_replacement_and_hidden_path(self):
        with self.assertRaisesRegex(ValueError, "removes"):
            hub.reviewed_delta("model", self.base, {"README.md": README})
        with self.assertRaisesRegex(ValueError, "historical"):
            hub.reviewed_delta("model", self.base, self.base | {"study-v1/weights.pt": README})
        with self.assertRaisesRegex(ValueError, "Unexpected staged path"):
            hub.reviewed_delta("model", self.base, self.base | {".gitattributes": README})
        self.assertEqual(hub.reviewed_delta("dataset", self.base, self.base | {"README.md": WEIGHT}),
                         {"README.md": WEIGHT})

    def test_reject_unreviewed_delta_without_write(self):
        with self.use_trees(), self.assertRaisesRegex(ValueError, "reviewed release manifest"):
            hub.release(self.api, "model", self.manifest | {"changes": {}}, writer=self.writer)
        self.writer.create_commit.assert_not_called()

    def test_reject_stale_base_without_write(self):
        self.api.repo_info.return_value.sha = PUBLISHED
        with self.use_trees(**{PUBLISHED: self.base | {"external-change": WEIGHT}}), \
                self.assertRaisesRegex(ValueError, "Stale Hub base"):
            hub.release(self.api, "model", self.manifest, writer=self.writer)
        self.writer.create_commit.assert_not_called()

    def test_publish_with_parent_guard_and_full_readback(self):
        self.writer.create_commit.return_value = SimpleNamespace(oid=PUBLISHED)
        self.api.repo_info.side_effect = [
            self.api.repo_info.return_value,
            SimpleNamespace(id=hub.REPO, sha=PUBLISHED, private=False),
        ]
        with self.use_trees(**{PUBLISHED: self.staged}):
            self.assertEqual(hub.release(self.api, "dataset", self.manifest, writer=self.writer,
                                         source_sha=BASE), (PUBLISHED, "published"))
        call = self.writer.create_commit.call_args
        self.assertEqual(call.kwargs["revision"], "main")
        self.assertEqual(call.kwargs["parent_commit"], BASE)
        self.assertEqual(call.kwargs["repo_type"], "dataset")
        [operation] = call.kwargs["operations"]
        self.assertEqual(operation.src_revision, STAGED)
        self.assertEqual(operation.path_in_repo, "study-v2/weights.pt")

    def test_retry_checks_the_actual_published_tree(self):
        self.api.repo_info.return_value.sha = PUBLISHED
        with self.use_trees(**{PUBLISHED: self.staged}):
            self.assertEqual(hub.release(self.api, "model", self.manifest, writer=self.writer),
                             (PUBLISHED, "already-current"))
        self.writer.create_commit.assert_not_called()

    def test_noop_baseline_does_not_create_a_commit(self):
        manifest = {"base_revision": BASE, "staged_revision": BASE, "changes": {}}
        with self.use_trees():
            self.assertEqual(hub.release(self.api, "model", manifest, writer=self.writer),
                             (BASE, "already-current"))
        self.writer.create_commit.assert_not_called()
