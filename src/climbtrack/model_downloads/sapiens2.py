"""Verified Sapiens2 snapshot acquisition and canonical metadata."""

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from climbtrack.config import AppConfig, resolve_project_path
from climbtrack.errors import ClimbTrackError
from climbtrack.hashing import fingerprint_file
from climbtrack.model_downloads.http import download_huggingface_file
from climbtrack.schema.keypoints import registry_from_sapiens_source, write_registry


def ensure_sapiens2_checkpoint(config: AppConfig, config_path: Path) -> tuple[Path, bool]:
    """Explicitly download a pinned Transformers snapshot and verify its identity."""
    model_config = config.models.sapiens2
    destination = resolve_project_path(model_config.model_dir, config_path)
    expected = (
        destination / "config.json",
        destination / "preprocessor_config.json",
        destination / model_config.checkpoint_filename,
        destination / "keypoints.json",
        destination / "download.json",
    )
    if destination.is_dir() and all(path.is_file() for path in expected):
        verify_sapiens2_checkpoint(
            destination / model_config.checkpoint_filename,
            model_config.checkpoint_sha256,
        )
        for path in expected:
            if path.name != model_config.checkpoint_filename:
                fingerprint_file(path)
        return destination, False
    if destination.exists():
        raise ClimbTrackError(
            f"Incomplete Sapiens2 model directory: {destination}. Move it aside and retry."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.download"
    temporary.mkdir(exist_ok=True)
    try:
        for filename in (
            "config.json",
            "preprocessor_config.json",
            model_config.checkpoint_filename,
        ):
            download_huggingface_file(
                model_config.model_id,
                model_config.revision,
                filename,
                temporary / filename,
                expected_size=(
                    model_config.checkpoint_size_bytes
                    if filename == model_config.checkpoint_filename
                    else None
                ),
                connections=model_config.download_connections,
                segment_size=model_config.download_segment_mb * 1024 * 1024,
            )
        checkpoint = fingerprint_file(temporary / model_config.checkpoint_filename)
        _require_expected_checkpoint(checkpoint, model_config.checkpoint_sha256)
        source = _download_text(model_config.keypoint_source_url)
        source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        registry = registry_from_sapiens_source(
            source,
            source_url=model_config.keypoint_source_url,
            source_sha256=source_sha256,
        )
        write_registry(registry, temporary / "keypoints.json")
        hub_cache = temporary / ".cache"
        if hub_cache.exists():
            shutil.rmtree(hub_cache)
        fingerprints = {
            "config.json": fingerprint_file(temporary / "config.json"),
            "preprocessor_config.json": fingerprint_file(temporary / "preprocessor_config.json"),
            model_config.checkpoint_filename: checkpoint,
            "keypoints.json": fingerprint_file(temporary / "keypoints.json"),
        }
        payload = {
            "schema_version": "1.0.0",
            "model_id": model_config.model_id,
            "revision": model_config.revision,
            "files": fingerprints,
        }
        (temporary / "download.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception as exc:
        if isinstance(exc, ClimbTrackError):
            raise
        raise ClimbTrackError(
            f"Could not download Sapiens2-1B: {exc}. Partial data was retained at {temporary}."
        ) from exc
    return destination, True


def _require_expected_checkpoint(fingerprint: dict[str, int | str], expected_sha256: str) -> None:
    actual = str(fingerprint["sha256"])
    if actual != expected_sha256:
        raise ClimbTrackError(
            f"Sapiens2 checkpoint SHA-256 mismatch: expected {expected_sha256}, found {actual}"
        )


def verify_sapiens2_checkpoint(path: Path, expected_sha256: str) -> dict[str, int | str]:
    """Fingerprint a checkpoint and reject any identity other than the pinned model."""
    fingerprint = fingerprint_file(path)
    _require_expected_checkpoint(fingerprint, expected_sha256)
    return fingerprint


def _download_text(url: str) -> str:
    if not url.startswith("https://"):
        raise ClimbTrackError("Sapiens2 metadata source must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "climbtrack/0.1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except (OSError, UnicodeError, urllib.error.URLError) as exc:
        raise ClimbTrackError(f"Could not download Sapiens2 keypoint metadata: {exc}") from exc
