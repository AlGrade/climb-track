from pathlib import Path

import pytest

from climbtrack.cache import StageCache
from climbtrack.errors import CacheIntegrityError


def _materialize(cache: StageCache, key: str, builder, *, force: bool = False):
    return cache.materialize(
        cache_key=key,
        stage_version="1",
        effective_config={"quality": "max"},
        input_fingerprint={"sha256": "input"},
        tools={"tool": {"version": "1"}},
        runtime={"python": "test"},
        git={"commit": None, "dirty": None},
        verify_checksums=True,
        force=force,
        builder=builder,
    )


def test_cache_reuses_complete_entry(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, "00_ingest")
    key = "a" * 64
    calls = 0

    def builder(output: Path) -> None:
        nonlocal calls
        calls += 1
        (output / "artifact.txt").write_text("stable", encoding="utf-8")

    first = _materialize(cache, key, builder)
    second = _materialize(cache, key, builder)

    assert not first.cache_hit
    assert second.cache_hit
    assert calls == 1


def test_cache_detects_corruption(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, "00_ingest")
    key = "b" * 64
    result = _materialize(
        cache,
        key,
        lambda output: (output / "artifact.txt").write_text("stable", encoding="utf-8"),
    )
    (result.path / "artifact.txt").write_text("tampered", encoding="utf-8")

    with pytest.raises(CacheIntegrityError, match="changed"):
        cache.load_complete(key, verify_checksums=True)


def test_failed_build_is_not_published(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, "00_ingest")
    key = "c" * 64

    def fail(output: Path) -> None:
        (output / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _materialize(cache, key, fail)

    assert not cache.entry_path(key).exists()
    assert list((cache.stage_root / ".failed").glob("*/partial.txt"))


def test_failed_force_rebuild_keeps_complete_entry(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, "00_ingest")
    key = "e" * 64
    first = _materialize(
        cache,
        key,
        lambda output: (output / "artifact.txt").write_text("stable", encoding="utf-8"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        _materialize(
            cache, key, lambda output: (_ for _ in ()).throw(RuntimeError("boom")), force=True
        )

    assert (first.path / "artifact.txt").read_text(encoding="utf-8") == "stable"
    assert cache.load_complete(key, verify_checksums=True) is not None


def test_force_keeps_replaced_entry(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, "00_ingest")
    key = "d" * 64
    _materialize(
        cache,
        key,
        lambda output: (output / "artifact.txt").write_text("first", encoding="utf-8"),
    )
    _materialize(
        cache,
        key,
        lambda output: (output / "artifact.txt").write_text("second", encoding="utf-8"),
        force=True,
    )

    assert (cache.entry_path(key) / "artifact.txt").read_text(encoding="utf-8") == "second"
    backups = list(cache.stage_root.glob(f".replaced-{key}-*"))
    assert len(backups) == 1
    assert (backups[0] / "artifact.txt").read_text(encoding="utf-8") == "first"
