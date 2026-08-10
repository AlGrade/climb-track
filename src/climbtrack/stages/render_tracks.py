"""Milestone-2 quality-control video for detections and track IDs."""

import json
import shutil
import statistics
import subprocess
from collections import defaultdict
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.errors import ExternalToolError
from climbtrack.hashing import hash_json
from climbtrack.provenance import executable_version, git_state, resolve_executable, runtime_state
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.tracks import read_tracks

STAGE_NAME = "50_render_tracks"
STAGE_VERSION = "1.0.0"


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
                    config,
                )
                overlay_path = overlay_dir / f"{frame_idx:09d}.jpg"
                if not cv2.imwrite(str(overlay_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise ExternalToolError(f"Could not write overlay frame: {overlay_path}")
                overlay_paths.append(overlay_path)
                if frame_idx in sample_indices:
                    shutil.copy2(overlay_path, output / f"preview_{frame_idx:09d}.jpg")
                progress.advance(task)

        durations = _frame_durations(frames)
        concat_path = output / ".overlay-concat.txt"
        lines = ["ffconcat version 1.0"]
        for path, duration in zip(overlay_paths, durations, strict=True):
            lines.append(f"file '{_escape_concat_path(path)}'")
            lines.append(f"duration {duration:.9f}")
        lines.append(f"file '{_escape_concat_path(overlay_paths[-1])}'")
        concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        metadata = json.loads((ingest.path / "metadata.json").read_text(encoding="utf-8"))
        duration_seconds = float(metadata["video"]["duration_seconds"])
        source_video = Path(str(ingest.manifest.input_fingerprint["path"]))
        output_video = output / "tracking_overlay.mp4"
        _encode_overlay(
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


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 97) % 192,
        64 + (track_id * 57) % 192,
        64 + (track_id * 137) % 192,
    )


def _frame_durations(frames: list[dict[str, Any]]) -> list[float]:
    timestamps = [float(frame["timestamp"]) for frame in frames]
    deltas = [right - left for left, right in pairwise(timestamps)]
    fallback = statistics.median(deltas)
    last = frames[-1]["duration"]
    last_duration = float(last) if last is not None and float(last) > 0 else fallback
    return [*deltas, last_duration]


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _encode_overlay(
    concat_path: Path,
    source_video: Path,
    output_video: Path,
    duration_seconds: float,
    config: AppConfig,
) -> None:
    executable = resolve_executable(config.render.ffmpeg_path)
    command = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        config.render.codec,
        "-preset",
        config.render.preset,
        "-crf",
        str(config.render.crf),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "vfr",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration_seconds:.9f}",
        "-movflags",
        "+faststart",
        "-y",
        str(output_video),
    ]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        detail = process.stderr.strip() or "no error output"
        raise ExternalToolError(f"ffmpeg overlay encoding failed: {detail}")


def _selected_track_id(selection: CacheResult | None) -> int | None:
    if selection is None:
        return None
    payload = json.loads((selection.path / "selection.json").read_text(encoding="utf-8"))
    return int(payload["track_id"])
