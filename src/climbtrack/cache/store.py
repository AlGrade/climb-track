"""Atomic, content-addressed stage cache implementation."""

import fcntl
import json
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from climbtrack.cache.manifest import CacheManifest, OutputArtifact
from climbtrack.errors import CacheIntegrityError
from climbtrack.hashing import hash_file, hash_json


@dataclass(frozen=True)
class CacheResult:
    """Resolved cache entry and whether existing work was reused."""

    path: Path
    manifest: CacheManifest
    cache_hit: bool


class StageCache:
    """Cache for one deterministic pipeline stage."""

    def __init__(self, root: Path, stage: str) -> None:
        self.stage = stage
        self.stage_root = root / stage
        self.stage_root.mkdir(parents=True, exist_ok=True)
        (self.stage_root / ".locks").mkdir(exist_ok=True)

    @staticmethod
    def make_key(
        *,
        stage: str,
        stage_version: str,
        effective_config: dict[str, Any],
        input_fingerprint: dict[str, Any],
        tools: dict[str, Any],
    ) -> str:
        """Create the cache key from every input that can change output bytes."""
        return hash_json(
            {
                "stage": stage,
                "stage_version": stage_version,
                "effective_config": effective_config,
                "input_fingerprint": input_fingerprint,
                "tools": tools,
            }
        )

    def entry_path(self, cache_key: str) -> Path:
        """Return the final path for a cache key."""
        return self.stage_root / cache_key

    @contextmanager
    def _lock(self, cache_key: str) -> Iterator[None]:
        lock_path = self.stage_root / ".locks" / f"{cache_key}.lock"
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def load_complete(self, cache_key: str, *, verify_checksums: bool) -> CacheManifest | None:
        """Load a complete entry, rejecting missing or corrupted artifacts."""
        entry = self.entry_path(cache_key)
        manifest_path = entry / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = CacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise CacheIntegrityError(f"Invalid cache manifest: {manifest_path}") from exc
        if manifest.status != "complete":
            return None
        if manifest.cache_key != cache_key or manifest.stage != self.stage:
            raise CacheIntegrityError(f"Cache manifest identity mismatch: {manifest_path}")
        for output in manifest.outputs:
            artifact = entry / output.path
            if not artifact.is_file():
                raise CacheIntegrityError(f"Cached artifact is missing: {artifact}")
            if artifact.stat().st_size != output.size_bytes:
                raise CacheIntegrityError(f"Cached artifact size changed: {artifact}")
            if verify_checksums and hash_file(artifact) != output.sha256:
                raise CacheIntegrityError(f"Cached artifact checksum changed: {artifact}")
        return manifest

    def materialize(
        self,
        *,
        cache_key: str,
        stage_version: str,
        effective_config: dict[str, Any],
        input_fingerprint: dict[str, Any],
        tools: dict[str, Any],
        runtime: dict[str, Any],
        git: dict[str, Any],
        verify_checksums: bool,
        force: bool,
        resume_incomplete: bool = False,
        builder: Callable[[Path], None],
    ) -> CacheResult:
        """Build an entry once and publish it with an atomic directory rename."""
        with self._lock(cache_key):
            if not force:
                existing = self.load_complete(cache_key, verify_checksums=verify_checksums)
                if existing is not None:
                    return CacheResult(self.entry_path(cache_key), existing, cache_hit=True)

            final_path = self.entry_path(cache_key)
            if resume_incomplete:
                temporary = self.stage_root / f".work-{cache_key}"
                if force and temporary.exists():
                    _archive_incomplete(self.stage_root, temporary, "abandoned")
                if temporary.exists():
                    base = _load_incomplete(
                        temporary,
                        stage=self.stage,
                        stage_version=stage_version,
                        cache_key=cache_key,
                        config_hash=hash_json(effective_config),
                    )
                else:
                    base = _new_incomplete_manifest(
                        stage=self.stage,
                        stage_version=stage_version,
                        cache_key=cache_key,
                        effective_config=effective_config,
                        input_fingerprint=input_fingerprint,
                        tools=tools,
                        runtime=runtime,
                        git=git,
                    )
                    temporary.mkdir()
                    _write_manifest(temporary / "manifest.json", base)
            else:
                temporary = self.stage_root / f".incomplete-{cache_key}-{uuid4().hex}"
                temporary.mkdir()
                base = _new_incomplete_manifest(
                    stage=self.stage,
                    stage_version=stage_version,
                    cache_key=cache_key,
                    effective_config=effective_config,
                    input_fingerprint=input_fingerprint,
                    tools=tools,
                    runtime=runtime,
                    git=git,
                )
                _write_manifest(temporary / "manifest.json", base)

            try:
                builder(temporary)
                outputs = tuple(_inventory(temporary))
                complete = base.model_copy(
                    update={
                        "status": "complete",
                        "completed_at": datetime.now(UTC),
                        "outputs": outputs,
                    }
                )
                _write_manifest(temporary / "manifest.json", complete)
                if final_path.exists():
                    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
                    backup = self.stage_root / f".replaced-{cache_key}-{timestamp}"
                    os.replace(final_path, backup)
                os.replace(temporary, final_path)
            except BaseException:
                if not resume_incomplete and temporary.exists():
                    _archive_incomplete(self.stage_root, temporary, "failed")
                raise

            return CacheResult(final_path, complete, cache_hit=False)


def _new_incomplete_manifest(
    *,
    stage: str,
    stage_version: str,
    cache_key: str,
    effective_config: dict[str, Any],
    input_fingerprint: dict[str, Any],
    tools: dict[str, Any],
    runtime: dict[str, Any],
    git: dict[str, Any],
) -> CacheManifest:
    return CacheManifest(
        status="incomplete",
        stage=stage,
        stage_version=stage_version,
        cache_key=cache_key,
        created_at=datetime.now(UTC),
        config_hash=hash_json(effective_config),
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
        runtime=runtime,
        git=git,
    )


def _load_incomplete(
    path: Path,
    *,
    stage: str,
    stage_version: str,
    cache_key: str,
    config_hash: str,
) -> CacheManifest:
    manifest_path = path / "manifest.json"
    try:
        manifest = CacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise CacheIntegrityError(f"Invalid resumable cache manifest: {manifest_path}") from exc
    identity = (
        manifest.status == "incomplete"
        and manifest.stage == stage
        and manifest.stage_version == stage_version
        and manifest.cache_key == cache_key
        and manifest.config_hash == config_hash
    )
    if not identity:
        raise CacheIntegrityError(f"Resumable cache identity mismatch: {manifest_path}")
    return manifest


def _archive_incomplete(stage_root: Path, path: Path, category: str) -> None:
    archive_root = stage_root / f".{category}"
    archive_root.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    shutil.move(str(path), str(archive_root / f"{path.name}-{timestamp}"))


def _inventory(root: Path) -> Iterator[OutputArtifact]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            yield OutputArtifact(
                path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=hash_file(path),
            )


def _write_manifest(path: Path, manifest: CacheManifest) -> None:
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
