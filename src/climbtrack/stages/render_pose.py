"""Stage 50: VFR-aware visual verification of raw pose observations."""

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
from climbtrack.errors import ClimbTrackError, ExternalToolError
from climbtrack.hashing import hash_json
from climbtrack.provenance import executable_version, git_state, runtime_state
from climbtrack.schema.crops import read_pose_crops
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.keypoints import read_registry
from climbtrack.schema.pose import read_pose_parquet
from climbtrack.stages.render_tracks import _encode_overlay, _escape_concat_path, _frame_durations

STAGE_NAME = "50_render_pose"
STAGE_VERSION = "1.0.0"


def render_pose_overlay(
    ingest: CacheResult,
    selection: CacheResult,
    pose: CacheResult,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Render the unrefined skeleton over the original VFR video."""
    effective_config = {
        "render": config.render.model_dump(mode="json"),
        "pose_render": config.pose_render.model_dump(mode="json"),
        "mode": "raw",
    }
    tools = {
        "ffmpeg": executable_version(config.render.ffmpeg_path),
        "opencv-python": version("opencv-python"),
    }
    input_fingerprint = {
        "ingest": upstream_fingerprint(ingest.manifest),
        "selection": upstream_fingerprint(selection.manifest),
        "pose": upstream_fingerprint(pose.manifest),
    }
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
        rows = read_pose_parquet(pose.path / "pose_raw.parquet")
        registry = read_registry(pose.path / "keypoints.json")
        crops = {
            int(row["frame_idx"]): row
            for row in read_pose_crops(selection.path / "pose_crops.parquet")
        }
        by_frame: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_frame[int(row["frame_idx"])][str(row["keypoint_name"])] = row
        if len(by_frame) != len(frames):
            raise ClimbTrackError(
                f"Pose overlay expected {len(frames)} frames, found {len(by_frame)}"
            )

        overlay_dir = output / ".overlay-frames"
        overlay_dir.mkdir()
        sample_indices = _sample_indices(frames)
        overlay_paths: list[Path] = []
        with Progress(TextColumn("50_render_pose"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("raw skeleton", total=len(frames))
            for frame in frames:
                frame_idx = int(frame["frame_idx"])
                source = ingest.path / str(frame["image_path"])
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise ExternalToolError(f"Could not read decoded frame: {source}")
                _draw_pose(
                    image,
                    frame_idx,
                    by_frame[frame_idx],
                    registry,
                    crops.get(frame_idx),
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
        output_video = output / "skeleton_raw_overlay.mp4"
        _encode_overlay(concat_path, source_video, output_video, duration_seconds, config)
        shutil.rmtree(overlay_dir)
        concat_path.unlink()
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "frames": len(frames),
            "mode": "raw_unrefined",
            "confidence_encoding": "red_to_green_and_alpha",
            "face_keypoints_visible": config.pose_render.show_face_keypoints,
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
        runtime=runtime_state(),
        git=git_state(project_root),
        verify_checksums=True,
        force=force,
        builder=build,
    )


def _draw_pose(
    image: Any,
    frame_idx: int,
    rows: dict[str, dict[str, Any]],
    registry: dict[str, Any],
    crop: dict[str, Any] | None,
    config: AppConfig,
) -> None:
    import cv2

    threshold = config.pose_render.confidence_threshold
    groups = {entry["name"]: entry["group"] for entry in registry["keypoints"]}
    primitives: list[tuple[str, tuple[Any, ...], float]] = []
    for left, right in registry["skeleton_edges"]:
        left_row, right_row = rows[left], rows[right]
        confidence = min(float(left_row["confidence"]), float(right_row["confidence"]))
        if confidence < threshold:
            continue
        primitives.append(
            (
                "line",
                (
                    (round(float(left_row["x"])), round(float(left_row["y"]))),
                    (round(float(right_row["x"])), round(float(right_row["y"]))),
                ),
                confidence,
            )
        )
    for name, row in rows.items():
        if groups[name] == "face" and not config.pose_render.show_face_keypoints:
            continue
        confidence = float(row["confidence"])
        if confidence >= threshold:
            primitives.append(
                (
                    "point",
                    ((round(float(row["x"])), round(float(row["y"]))),),
                    confidence,
                )
            )

    for bucket in range(5):
        selected = [
            item for item in primitives if min(4, int(min(1.0, max(0.0, item[2])) * 5)) == bucket
        ]
        if not selected:
            continue
        confidence = (bucket + 0.5) / 5
        color = (0, round(255 * confidence), round(255 * (1.0 - confidence)))
        layer = image.copy()
        for kind, points, _ in selected:
            if kind == "line":
                cv2.line(
                    layer,
                    points[0],
                    points[1],
                    color,
                    config.pose_render.line_thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    layer,
                    points[0],
                    config.pose_render.point_radius,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
        alpha = 0.25 + 0.75 * confidence
        cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0.0, image)

    if crop is not None and config.pose_render.show_pose_crop:
        x1, y1, x2, y2 = (round(float(crop[name])) for name in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    track_id = next(iter(rows.values()))["track_id"]
    cv2.putText(
        image,
        f"RAW Sapiens2-1B  ID {track_id}  frame {frame_idx}",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.render.font_scale,
        (255, 255, 255),
        max(2, config.pose_render.line_thickness // 2),
        cv2.LINE_AA,
    )


def _sample_indices(frames: list[dict[str, Any]]) -> set[int]:
    targets = (0.0, 1.0, 9.0, 14.0, 17.0, float(frames[-1]["timestamp"]))
    return {
        int(min(frames, key=lambda frame: abs(float(frame["timestamp"]) - target))["frame_idx"])
        for target in targets
    }
