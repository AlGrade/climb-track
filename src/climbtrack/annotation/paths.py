"""Stable paths shared by editable annotation workflows."""

import re
from pathlib import Path


def annotation_session_dir(annotation_root: Path, source_video: Path, ingest_key: str) -> Path:
    """Return one deterministic directory shared by a video's annotations."""
    return annotation_root / f"{_slug(source_video.stem)}-{ingest_key[:12]}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return slug or "video"
