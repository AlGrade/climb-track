"""Versioned and atomically persisted manual move ground truth."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from climbtrack.annotation.paths import annotation_session_dir
from climbtrack.cache import CacheResult
from climbtrack.errors import ClimbTrackError
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.moves import write_moves_parquet

MOVE_SESSION_SCHEMA_VERSION = "1.1.0"


class MoveModel(BaseModel):
    """Strict base for user-editable move data."""

    model_config = ConfigDict(extra="forbid")


class MovingHand(StrEnum):
    """Hand assignment for one move segment."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class MoveEdit(MoveModel):
    """Small browser payload; timestamps are resolved authoritatively on the server."""

    move_id: int | None = Field(default=None, ge=1)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    moving_hand: MovingHand
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["manual", "automatic", "corrected"] = "manual"
    is_reviewed: bool = True
    outcome: Literal["completed", "fall"] = "completed"

    @model_validator(mode="after")
    def require_forward_range(self) -> "MoveEdit":
        if self.end_frame <= self.start_frame:
            raise ValueError("A move must end after it starts")
        return self


class MoveAnnotation(MoveModel):
    """One reviewed hand move aligned to exact source frames."""

    move_id: int = Field(ge=1)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_timestamp: float = Field(ge=0.0)
    end_timestamp: float = Field(ge=0.0)
    moving_hand: MovingHand
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(pattern="^(manual|automatic|corrected)$")
    is_reviewed: bool
    outcome: Literal["completed", "fall"] = "completed"

    @model_validator(mode="after")
    def require_forward_range(self) -> "MoveAnnotation":
        if self.end_frame <= self.start_frame or self.end_timestamp <= self.start_timestamp:
            raise ValueError("A move must end after it starts")
        return self


class MoveSession(MoveModel):
    """Identity and revision metadata for one video's move annotations."""

    schema_version: str = MOVE_SESSION_SCHEMA_VERSION
    created_at: str
    updated_at: str
    source_video_name: str
    ingest_cache_key: str
    automatic_moves_cache_key: str | None = None
    frame_count: int = Field(ge=2)
    first_timestamp: float = Field(ge=0.0)
    last_timestamp: float = Field(ge=0.0)
    revision: int = Field(default=0, ge=0)
    moves: list[MoveAnnotation]

    @model_validator(mode="after")
    def validate_timeline(self) -> "MoveSession":
        if self.last_timestamp <= self.first_timestamp:
            raise ValueError("Move session requires a forward video timeline")
        if [move.move_id for move in self.moves] != list(range(1, len(self.moves) + 1)):
            raise ValueError("Move IDs must be contiguous and start at one")
        starts = [move.start_timestamp for move in self.moves]
        if starts != sorted(starts):
            raise ValueError("Moves must be sorted by start timestamp")
        return self


def prepare_move_session(
    ingest: CacheResult,
    *,
    annotation_root: Path,
    automatic_moves: list[dict[str, Any]] | None = None,
    automatic_moves_cache_key: str | None = None,
) -> tuple[Path, MoveSession, bool]:
    """Create or resume the manual move session for one ingested video."""
    frames = read_frame_index(ingest.path / "frames.parquet")
    source_video = Path(str(ingest.manifest.input_fingerprint["path"]))
    session_dir = annotation_session_dir(
        annotation_root,
        source_video,
        ingest.manifest.cache_key,
    )
    session_path = session_dir / "moves_ground_truth.json"
    if session_path.is_file():
        existing = load_move_session(session_path)
        _validate_identity(existing, ingest.manifest.cache_key, source_video.name, frames)
        automatic_only = all(move.source == "automatic" for move in existing.moves)
        should_refresh = (
            existing.revision == 0
            and automatic_moves
            and automatic_only
            and (
                not existing.moves
                or existing.automatic_moves_cache_key != automatic_moves_cache_key
            )
        )
        if should_refresh:
            existing = existing.model_copy(
                update={
                    "schema_version": MOVE_SESSION_SCHEMA_VERSION,
                    "automatic_moves_cache_key": automatic_moves_cache_key,
                    "moves": _automatic_annotations(automatic_moves),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            save_move_session(existing, session_path)
        return session_path, existing, False

    now = datetime.now(UTC).isoformat()
    session = MoveSession(
        created_at=now,
        updated_at=now,
        source_video_name=source_video.name,
        ingest_cache_key=ingest.manifest.cache_key,
        automatic_moves_cache_key=automatic_moves_cache_key,
        frame_count=len(frames),
        first_timestamp=float(frames[0]["timestamp"]),
        last_timestamp=float(frames[-1]["timestamp"]),
        moves=_automatic_annotations(automatic_moves or []),
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    save_move_session(session, session_path)
    return session_path, session, True


def apply_move_edits(
    session: MoveSession,
    edits: list[MoveEdit],
    frames: list[dict[str, Any]],
    *,
    expected_revision: int,
) -> MoveSession:
    """Validate browser edits, snap to source timestamps, and create a new revision."""
    if expected_revision != session.revision:
        raise ClimbTrackError(
            "Move annotations changed in another browser tab; reload before saving again."
        )
    if len(edits) > 10_000:
        raise ClimbTrackError("A move session cannot contain more than 10,000 moves")
    timestamps = {int(frame["frame_idx"]): float(frame["timestamp"]) for frame in frames}
    annotations: list[MoveAnnotation] = []
    seen: set[tuple[int, int, MovingHand]] = set()
    for edit in edits:
        try:
            start_timestamp = timestamps[edit.start_frame]
            end_timestamp = timestamps[edit.end_frame]
        except KeyError as exc:
            raise ClimbTrackError(f"Move references unknown frame {exc.args[0]}") from exc
        signature = (edit.start_frame, edit.end_frame, edit.moving_hand)
        if signature in seen:
            raise ClimbTrackError("Duplicate move annotations are not allowed")
        seen.add(signature)
        annotations.append(
            MoveAnnotation(
                move_id=1,
                start_frame=edit.start_frame,
                end_frame=edit.end_frame,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                moving_hand=edit.moving_hand,
                confidence=edit.confidence,
                source=edit.source,
                is_reviewed=edit.is_reviewed,
                outcome=edit.outcome,
            )
        )
    annotations.sort(key=lambda move: (move.start_timestamp, move.end_timestamp, move.moving_hand))
    annotations = [
        move.model_copy(update={"move_id": index}) for index, move in enumerate(annotations, 1)
    ]
    return session.model_copy(
        update={
            "updated_at": datetime.now(UTC).isoformat(),
            "revision": session.revision + 1,
            "moves": annotations,
        }
    )


def save_move_session(session: MoveSession, path: Path) -> None:
    """Atomically save JSON ground truth and its canonical Parquet projection."""
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    write_moves_parquet(
        (move.model_dump(mode="json") for move in session.moves),
        path.with_name("moves.parquet"),
    )


def load_move_session(path: Path) -> MoveSession:
    """Load and strictly validate one move session."""
    try:
        return MoveSession.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClimbTrackError(f"Invalid move annotation session: {path}") from exc


def _validate_identity(
    session: MoveSession,
    ingest_key: str,
    source_video_name: str,
    frames: list[dict[str, Any]],
) -> None:
    expected = (
        ingest_key,
        source_video_name,
        len(frames),
        float(frames[0]["timestamp"]),
        float(frames[-1]["timestamp"]),
    )
    actual = (
        session.ingest_cache_key,
        session.source_video_name,
        session.frame_count,
        session.first_timestamp,
        session.last_timestamp,
    )
    if actual != expected:
        raise ClimbTrackError(
            "Existing move annotations belong to a different video timeline; move the annotation "
            "directory aside before starting a new session."
        )


def _automatic_annotations(rows: list[dict[str, Any]]) -> list[MoveAnnotation]:
    annotations = [MoveAnnotation.model_validate(row) for row in rows]
    annotations.sort(key=lambda move: (move.start_timestamp, move.end_timestamp))
    return [move.model_copy(update={"move_id": index}) for index, move in enumerate(annotations, 1)]
