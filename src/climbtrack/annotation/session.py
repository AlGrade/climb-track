"""Versioned, editable annotation sessions initialized from raw pose predictions."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from climbtrack.annotation.keypoints import annotation_keypoints
from climbtrack.annotation.paths import annotation_session_dir
from climbtrack.annotation.selection import select_annotation_frames
from climbtrack.cache import CacheResult
from climbtrack.config import AppConfig
from climbtrack.errors import ClimbTrackError
from climbtrack.schema.crops import read_pose_crops
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.keypoints import read_registry
from climbtrack.schema.pose import read_pose_parquet

SESSION_SCHEMA_VERSION = "1.0.0"


class SessionModel(BaseModel):
    """Strict mutable base model for user-edited annotation state."""

    model_config = ConfigDict(extra="forbid")


class AnnotationCrop(SessionModel):
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationPoint(SessionModel):
    name: str
    group: str
    predicted_x: float
    predicted_y: float
    predicted_confidence: float
    x: float | None
    y: float | None
    visible: bool = True

    @model_validator(mode="after")
    def coordinates_match_visibility(self) -> "AnnotationPoint":
        """Visible points need coordinates; occluded points must use nulls."""
        if self.visible and (self.x is None or self.y is None):
            raise ValueError("Visible annotation points require x and y")
        if not self.visible and (self.x is not None or self.y is not None):
            raise ValueError("Occluded annotation points must store null x and y")
        return self


class AnnotationFrame(SessionModel):
    frame_idx: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    image_path: str
    crop: AnnotationCrop
    reviewed: bool = False
    points: dict[str, AnnotationPoint]


class AnnotationSession(SessionModel):
    schema_version: str = SESSION_SCHEMA_VERSION
    created_at: str
    source_video_name: str
    ingest_cache_key: str
    selection_cache_key: str
    pose_cache_key: str
    keypoint_names: list[str]
    frames: list[AnnotationFrame]


def prepare_session(
    ingest: CacheResult,
    selection: CacheResult,
    pose: CacheResult,
    *,
    config: AppConfig,
    annotation_root: Path,
) -> tuple[Path, AnnotationSession, bool]:
    """Create or resume a deterministic ten-frame ground-truth session."""
    registry = read_registry(pose.path / "keypoints.json")
    entries = annotation_keypoints(registry)
    keypoint_names = [str(entry["name"]) for entry in entries]
    frames = read_frame_index(ingest.path / "frames.parquet")
    pose_rows = read_pose_parquet(pose.path / "pose_raw.parquet")
    selected = select_annotation_frames(
        frames,
        pose_rows,
        set(keypoint_names),
        count=config.annotation.sample_count,
        minimum_spacing_seconds=config.annotation.minimum_spacing_seconds,
    )
    source_video = Path(str(ingest.manifest.input_fingerprint["path"]))
    session_dir = annotation_session_dir(
        annotation_root,
        source_video,
        ingest.manifest.cache_key,
    )
    session_path = session_dir / "ground_truth.json"
    if session_path.is_file():
        existing = load_session(session_path)
        _validate_identity(existing, ingest, selection, pose, keypoint_names)
        return session_path, existing, False

    frame_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    crops = {
        int(crop["frame_idx"]): crop
        for crop in read_pose_crops(selection.path / "pose_crops.parquet")
    }
    rows_by_frame: dict[int, dict[str, dict[str, Any]]] = {}
    for row in pose_rows:
        frame_idx = int(row["frame_idx"])
        name = str(row["keypoint_name"])
        if name in keypoint_names:
            rows_by_frame.setdefault(frame_idx, {})[name] = row

    annotation_frames = [
        _annotation_frame(
            frame_by_idx[frame_idx],
            crops[frame_idx],
            rows_by_frame[frame_idx],
            entries,
        )
        for frame_idx in selected
    ]
    session = AnnotationSession(
        created_at=datetime.now(UTC).isoformat(),
        source_video_name=source_video.name,
        ingest_cache_key=ingest.manifest.cache_key,
        selection_cache_key=selection.manifest.cache_key,
        pose_cache_key=pose.manifest.cache_key,
        keypoint_names=keypoint_names,
        frames=annotation_frames,
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    save_session(session, session_path)
    return session_path, session, True


def save_session(session: AnnotationSession, path: Path) -> None:
    """Atomically persist user edits after every reviewed frame."""
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_session(path: Path) -> AnnotationSession:
    """Load and strictly validate one editable ground-truth session."""
    try:
        return AnnotationSession.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClimbTrackError(f"Invalid annotation session: {path}") from exc


def _annotation_frame(
    frame: dict[str, Any],
    crop: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    entries: tuple[dict[str, Any], ...],
) -> AnnotationFrame:
    missing = {str(entry["name"]) for entry in entries} - rows.keys()
    if missing:
        raise ClimbTrackError(
            f"Frame {frame['frame_idx']} lacks pose observations: {sorted(missing)}"
        )
    points = {}
    for entry in entries:
        name = str(entry["name"])
        row = rows[name]
        x, y = float(row["x"]), float(row["y"])
        points[name] = AnnotationPoint(
            name=name,
            group=str(entry["group"]),
            predicted_x=x,
            predicted_y=y,
            predicted_confidence=float(row["confidence"]),
            x=x,
            y=y,
        )
    return AnnotationFrame(
        frame_idx=int(frame["frame_idx"]),
        timestamp=float(frame["timestamp"]),
        image_path=str(frame["image_path"]),
        crop=AnnotationCrop(**{name: float(crop[name]) for name in ("x1", "y1", "x2", "y2")}),
        points=points,
    )


def _validate_identity(
    session: AnnotationSession,
    ingest: CacheResult,
    selection: CacheResult,
    pose: CacheResult,
    keypoint_names: list[str],
) -> None:
    expected = (
        ingest.manifest.cache_key,
        selection.manifest.cache_key,
        pose.manifest.cache_key,
        keypoint_names,
    )
    actual = (
        session.ingest_cache_key,
        session.selection_cache_key,
        session.pose_cache_key,
        session.keypoint_names,
    )
    if actual != expected:
        raise ClimbTrackError(
            "Existing annotations belong to different cached inputs; move the annotation "
            "directory aside before starting a new session."
        )
