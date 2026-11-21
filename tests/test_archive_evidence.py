from hashlib import sha256
from io import BytesIO

import pytest

from evidence_dial import archive_evidence


@pytest.fixture
def archived_file(monkeypatch):
    relative = "evidence/selection.json"
    data = b'{"measurement":"original"}\n'
    monkeypatch.setattr(archive_evidence, "ARCHIVED_FILES", {relative: (len(data), sha256(data).hexdigest())})
    monkeypatch.setattr(archive_evidence, "_git_blob", lambda *_: None)
    return relative, data


def test_restore_offline_to_original_path_and_refuse_conflicting_bytes(tmp_path, archived_file):
    relative, data = archived_file
    source = tmp_path / "archive" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    target_root = tmp_path / "checkout"

    restored = archive_evidence.restore_file(target_root, relative, archive_dir=source.parents[1], allow_network=False)
    assert restored == target_root / relative
    assert restored.read_bytes() == data
    restored.write_bytes(b"conflicting local file")

    with pytest.raises(ValueError, match="recorded SHA-256"):
        archive_evidence.restore_file(target_root, relative, archive_dir=source.parents[1], allow_network=False)
    assert restored.read_bytes() == b"conflicting local file"


def test_restore_without_git_checks_the_pinned_url_and_bytes(tmp_path, monkeypatch, archived_file):
    relative, data = archived_file
    called = []

    def download(url, *, timeout):
        called.append(url)
        return BytesIO(data)

    monkeypatch.setattr(archive_evidence, "urlopen", download)
    with pytest.raises(FileNotFoundError):
        archive_evidence.restore_file(tmp_path, relative, allow_network=False)
    assert not called
    assert archive_evidence.restore_file(tmp_path, relative).read_bytes() == data
    assert called == [
        f"https://raw.githubusercontent.com/HyperGAN/{archive_evidence.ARCHIVE_REPOSITORY}/"
        f"{archive_evidence.ARCHIVE_COMMIT}/{relative}"
    ]

    monkeypatch.setattr(archive_evidence, "urlopen", lambda *args, **kwargs: BytesIO(b"tampered"))
    other = tmp_path / "fresh"
    with pytest.raises(ValueError, match="recorded SHA-256"):
        archive_evidence.restore_file(other, relative)
    assert not (other / relative).exists()
