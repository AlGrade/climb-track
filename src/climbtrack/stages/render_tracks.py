"""Milestone-2 quality-control video for detections and track IDs."""

import json
import shutil
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.errors import ExternalToolError
from climbtrack.hashing import hash_json
from climbtrack.provenance import executable_version, git_state, runtime_state
from climbtrack.rendering.video import encode_overlay, frame_durations, write_concat_manifest
from climbtrack.schema.crops import read_pose_crops
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.tracks import read_tracks

STAGE_NAME = "50_render_tracks"
STAGE_VERSION = "2.0.1"


def render_tracking_overlay(
    ingest: CacheResult,
    tracks: CacheResult,
    selection: CacheResult | None,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Render a VFR-aware tracking overlay and preserve source audio when present."""
    selected_track_id = _selected_track_id(selection)
    effective_config = {
        **config.render.model_dump(mode="json"),
        "selected_track_id": selected_track_id,
    }
    tools = {
        "ffmpeg": executable_version(config.render.ffmpeg_path),
        "opencv-python": version("opencv-python"),
    }
    input_fingerprint = {
        "tracks": upstream_fingerprint(tracks.manifest),
        "selection": upstream_fingerprint(selection.manifest) if selection else None,
    }
    runtime = runtime_state()
    git = git_state(project_root)
    cache = StageCache(cache_root, STAGE_NAME)
    cache_key = cache.make_key(
        stage=STAGE_NAME,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
    )

    def build(output: Path) -> None:
        import cv2

        frames = read_frame_index(ingest.path / "frames.parquet")
        track_rows = read_tracks(tracks.path / "tracks.parquet")
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in track_rows:
            by_frame[int(row["frame_idx"])].append(row)
        crops_by_frame = _pose_crops_by_frame(selection)

        overlay_dir = output / ".overlay-frames"
        overlay_dir.mkdir()
        sample_indices = {0, len(frames) // 2, len(frames) - 1}
        overlay_paths: list[Path] = []
        with Progress(TextColumn("50_render"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("overlay", total=len(frames))
            for frame in frames:
                frame_idx = int(frame["frame_idx"])
                source = ingest.path / str(frame["image_path"])
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise ExternalToolError(f"Could not read decoded frame: {source}")
                _draw_overlay(
                    image,
                    frame_idx,
                    by_frame.get(frame_idx, []),
                    selected_track_id,
                    crops_by_frame.get(frame_idx),
                    config,
                )
                overlay_path = overlay_dir / f"{frame_idx:09d}.jpg"
                if not cv2.imwrite(str(overlay_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise ExternalToolError(f"Could not write overlay frame: {overlay_path}")
                overlay_paths.append(overlay_path)
                if frame_idx in sample_indices:
                    shutil.copy2(overlay_path, output / f"preview_{frame_idx:09d}.jpg")
                progress.advance(task)

        durations = frame_durations(frames)
        concat_path = output / ".overlay-concat.txt"
        write_concat_manifest(concat_path, overlay_paths, durations)

        metadata = json.loads((ingest.path / "metadata.json").read_text(encoding="utf-8"))
        duration_seconds = float(metadata["video"]["duration_seconds"])
        source_video = Path(str(ingest.manifest.input_fingerprint["path"]))
        output_video = output / "tracking_overlay.mp4"
        encode_overlay(
            concat_path,
            source_video,
            output_video,
            duration_seconds,
            config,
        )
        shutil.rmtree(overlay_dir)
        concat_path.unlink()
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "frames": len(frames),
            "duration_seconds": duration_seconds,
            "selected_track_id": selected_track_id,
            "timing": "source_frame_timestamps_vfr",
            "source_audio": "preserved_when_present",
            "config_hash": hash_json(effective_config),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return cache.materialize(
        cache_key=cache_key,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
        runtime=runtime,
        git=git,
        verify_checksums=True,
        force=force,
        builder=build,
    )


def _draw_overlay(
    image: Any,
    frame_idx: int,
    rows: list[dict[str, Any]],
    selected_track_id: int | None,
    pose_crop: dict[str, Any] | None,
    config: AppConfig,
) -> None:
    import cv2

    thickness = config.render.line_thickness
    cv2.putText(
        image,
        f"frame {frame_idx}",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.render.font_scale,
        (255, 255, 255),
        max(2, thickness // 2),
        cv2.LINE_AA,
    )
    for row in rows:
        track_id = int(row["track_id"])
        if not config.render.show_all_tracks and track_id != selected_track_id:
            continue
        selected = track_id == selected_track_id
        if selected and pose_crop is not None:
            continue
        color = (0, 255, 0) if selected else _track_color(track_id)
        x1, y1, x2, y2 = (round(float(row[name])) for name in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"ID {track_id}  {float(row['confidence']):.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(30, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.render.font_scale,
            color,
            max(2, thickness // 2),
            cv2.LINE_AA,
        )
    if pose_crop is not None:
        x1, y1, x2, y2 = (round(float(pose_crop[name])) for name in ("x1", "y1", "x2", "y2"))
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        suffix = " interpolated" if pose_crop["is_interpolated"] else ""
        cv2.putText(
            image,
            f"ID {pose_crop['track_id']} pose crop{suffix}",
            (x1, max(30, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.render.font_scale,
            color,
            max(2, thickness // 2),
            cv2.LINE_AA,
        )


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 97) % 192,
        64 + (track_id * 57) % 192,
        64 + (track_id * 137) % 192,
    )


def _selected_track_id(selection: CacheResult | None) -> int | None:
    if selection is None:
        return None
    payload = json.loads((selection.path / "selection.json").read_text(encoding="utf-8"))
    return int(payload["track_id"])


def _pose_crops_by_frame(selection: CacheResult | None) -> dict[int, dict[str, Any]]:
    if selection is None:
        return {}
    rows = read_pose_crops(selection.path / "pose_crops.parquet")
    return {int(row["frame_idx"]): row for row in rows}
