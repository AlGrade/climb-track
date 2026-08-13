"""Stage orchestration shared by every entry point.

The stages in `climbtrack.stages` are deliberately independent: each one takes its
upstream results explicitly and decides on its own whether the cache already holds
the answer. This module holds the dependency order between them, so that callers
can ask for "the refined pose of this video" without restating the chain.

Nothing here writes to a terminal. The CLI adds reporting on top; keeping the two
apart is what makes the pipeline usable from tests and from any future front end.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from climbtrack.cache import CacheResult
from climbtrack.config import (
    AppConfig,
    load_config,
    resolve_cache_dir,
    resolve_project_path,
)
from climbtrack.device import seed_torch
from climbtrack.errors import ClimbTrackError
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.tracks import read_tracks
from climbtrack.selection.click import choose_track_by_click
from climbtrack.stages.detect import detect_people
from climbtrack.stages.ingest import ingest_video
from climbtrack.stages.move_metrics import measure_moves
from climbtrack.stages.moves import detect_moves
from climbtrack.stages.pose import estimate_pose
from climbtrack.stages.refine import refine_pose
from climbtrack.stages.select import select_climber
from climbtrack.stages.track import track_people


@dataclass(frozen=True)
class PipelineContext:
    """Everything the stages need to locate configuration, cache, and project."""

    config: AppConfig
    config_path: Path
    project_root: Path
    cache_root: Path


def build_context(config_path: Path) -> PipelineContext:
    """Load the configuration and seed every random source before any stage runs."""
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    seed_everything(config)
    return PipelineContext(
        config=config,
        config_path=config_path,
        project_root=config_path.parent.parent,
        cache_root=resolve_cache_dir(config, config_path),
    )


def seed_everything(config: AppConfig) -> None:
    """Seed Python, NumPy, and torch from the single configured seed."""
    random.seed(config.project.seed)
    np.random.seed(config.project.seed)
    seed_torch(config.project.seed)


def run_ingest(video: Path, context: PipelineContext, force: bool = False) -> CacheResult:
    """Stage 00: lossless frames plus true source timestamps."""
    return ingest_video(
        video.expanduser(),
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def run_detect(
    ingest_result: CacheResult, context: PipelineContext, force: bool = False
) -> CacheResult:
    """Stage 10: YOLO11x person boxes."""
    model_path = resolve_project_path(context.config.detection.model_path, context.config_path)
    if not model_path.is_file():
        raise ClimbTrackError(
            f"YOLO11x checkpoint is missing: {model_path}. Run 'climbtrack download-yolo'."
        )
    return detect_people(
        ingest_result,
        model_path=model_path,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def run_to_tracks(
    video: Path, config_path: Path, force: bool = False
) -> tuple[PipelineContext, CacheResult, CacheResult, CacheResult]:
    """Run stages 00-20 and return context, ingest, tracks, and detections."""
    context = build_context(config_path)
    ingest_result = run_ingest(video, context, force)
    detections = run_detect(ingest_result, context, force)
    tracks = track_people(
        ingest_result,
        detections,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )
    return context, ingest_result, tracks, detections


def run_to_selection(
    video: Path,
    config_path: Path,
    *,
    track_id: int | None = None,
    click: bool = False,
) -> tuple[PipelineContext, CacheResult, CacheResult]:
    """Run stages 00-25 and return context, ingest, and the chosen climber."""
    context, ingest_result, tracks_result, _ = run_to_tracks(video, config_path, False)
    chosen = resolve_manual_track(track_id, click, ingest_result, tracks_result)
    selection = select_climber(
        ingest_result,
        tracks_result,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        manual_track_id=chosen,
    )
    return context, ingest_result, selection


def run_pose(
    ingest_result: CacheResult,
    selection: CacheResult,
    context: PipelineContext,
    force: bool = False,
) -> CacheResult:
    """Stage 30: raw Sapiens2 keypoints."""
    model_dir = resolve_project_path(context.config.models.sapiens2.model_dir, context.config_path)
    return estimate_pose(
        ingest_result,
        selection,
        model_dir=model_dir,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def run_refine(
    selection: CacheResult,
    pose_result: CacheResult,
    context: PipelineContext,
    force: bool = False,
) -> CacheResult:
    """Stage 40: temporal repair of the raw keypoints."""
    return refine_pose(
        selection,
        pose_result,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def run_moves(
    refined: CacheResult,
    context: PipelineContext,
    force: bool = False,
) -> CacheResult:
    """Stage 70: automatic hand-move segmentation."""
    return detect_moves(
        refined,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def run_move_metrics(
    refined: CacheResult,
    moves_path: Path,
    context: PipelineContext,
    force: bool = False,
) -> CacheResult:
    """Stage 80: per-move speed, posture, and coordination metrics."""
    return measure_moves(
        refined,
        moves_path,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def resolve_manual_track(
    track_id: int | None,
    click: bool,
    ingest_result: CacheResult,
    tracks_result: CacheResult,
) -> int | None:
    """Return the manually chosen track ID, or None to let scoring decide."""
    if track_id is not None and click:
        raise ClimbTrackError("Use either --track-id or --click, not both")
    if not click:
        return track_id
    frames = read_frame_index(ingest_result.path / "frames.parquet")
    tracks = read_tracks(tracks_result.path / "tracks.parquet")
    return choose_track_by_click(ingest_result.path, frames, tracks)
