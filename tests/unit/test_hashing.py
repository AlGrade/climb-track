from pathlib import Path

from climbtrack.hashing import fingerprint_file, hash_json


def test_structured_hash_is_order_independent() -> None:
    assert hash_json({"b": 2, "a": 1}) == hash_json({"a": 1, "b": 2})


def test_file_fingerprint_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "video.bin"
    path.write_bytes(b"first")
    first = fingerprint_file(path)
    path.write_bytes(b"second")
    second = fingerprint_file(path)

    assert first["sha256"] != second["sha256"]
    assert second["path"] == str(path.resolve())
