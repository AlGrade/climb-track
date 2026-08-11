"""Side-by-side raw-versus-refined quality-control video."""

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
from climbtrack.rendering.pose import draw_pose
from climbtrack.rendering.video import encode_overlay, frame_durations, write_concat_manifest
from climbtrack.schema.crops import read_pose_crops
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.keypoints import read_registry
from climbtrack.schema.pose import read_pose_parquet

STAGE_NAME = "50_render_compare"
STAGE_VERSION = "1.0.0"


def render_pose_comparison(
    ingest: CacheResult,
    selection: CacheResult,
    pose: CacheResult,
    refined: CacheResult,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Render synchronized raw and refined skeletons at a review-friendly size."""
    effective_config = {
        "render": config.render.model_dump(mode="json"),
        "pose_render": config.pose_render.model_dump(mode="json"),
        "mode": "raw_vs_refined",
    }
    tools = {
        "ffmpeg": executable_version(config.render.ffmpeg_path),
        "opencv-python": version("opencv-python"),
    }
    input_fingerprint = {
        "ingest": upstream_fingerprint(ingest.manifest),
        "selection": upstream_fingerprint(selection.manifest),
        "pose": upstream_fingerprint(pose.manifest),
        "refined": upstream_fingerprint(refined.manifest),
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
        registry = read_registry(pose.path / "keypoints.json")
        included_names = {
            str(entry["name"])
            for entry in registry["keypoints"]
            if config.pose_render.show_face_keypoints or entry["group"] != "face"
        }
        raw_by_frame = _index_rows(
            read_pose_parquet(pose.path / "pose_raw.parquet"), included_names
        )
        refined_by_frame = _index_rows(
            read_pose_parquet(refined.path / "pose_refined.parquet"), included_names
        )
        if len(raw_by_frame) != len(frames) or len(refined_by_frame) != len(frames):
            raise ClimbTrackError("Comparison render requires raw and refined rows for every frame")
        crops = {
            int(row["frame_idx"]): row
            for row in read_pose_crops(selection.path / "pose_crops.parquet")
        }
        overlay_dir = output / ".overlay-frames"
        overlay_dir.mkdir()
        sample_indices = _sample_indices(frames)
        overlay_paths: list[Path] = []
        panel_size: tuple[int, int] | None = None
        with Progress(TextColumn(STAGE_NAME), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("raw vs refined", total=len(frames))
            for frame in frames:
                frame_idx = int(frame["frame_idx"])
                source = ingest.path / str(frame["image_path"])
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise ExternalToolError(f"Could not read decoded frame: {source}")
                if panel_size is None:
                    panel_size = _panel_size(image, config.pose_render.comparison_panel_width)
                raw_image, refined_image = image.copy(), image.copy()
                draw_pose(
                    raw_image,
                    frame_idx,
                    raw_by_frame[frame_idx],
                    registry,
                    crops.get(frame_idx),
                    config,
                    label="RAW",
                )
                draw_pose(
                    refined_image,
                    frame_idx,
                    refined_by_frame[frame_idx],
                    registry,
                    crops.get(frame_idx),
                    config,
                    label="REFINED",
                )
                raw_panel = cv2.resize(raw_image, panel_size, interpolation=cv2.INTER_AREA)
                refined_panel = cv2.resize(refined_image, panel_size, interpolation=cv2.INTER_AREA)
                comparison = cv2.hconcat((raw_panel, refined_panel))
                cv2.line(
                    comparison,
                    (panel_size[0], 0),
                    (panel_size[0], panel_size[1]),
                    (255, 255, 255),
                    2,
                )
                overlay_path = overlay_dir / f"{frame_idx:09d}.jpg"
                if not cv2.imwrite(str(overlay_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise ExternalToolError(f"Could not write comparison frame: {overlay_path}")
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
        output_video = output / "raw_vs_refined.mp4"
        encode_overlay(concat_path, source_video, output_video, duration_seconds, config)
        shutil.rmtree(overlay_dir)
        concat_path.unlink()
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "frames": len(frames),
            "layout": "raw_left_refined_right",
            "panel_size": list(panel_size or (0, 0)),
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


def _index_rows(
    rows: list[dict[str, Any]], included_names: set[str]
) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        name = str(row["keypoint_name"])
        if name in included_names:
            result[int(row["frame_idx"])][name] = row
    return result


def _panel_size(image: Any, maximum_width: int) -> tuple[int, int]:
    height, width = image.shape[:2]
    panel_width = min(maximum_width, width)
    panel_height = round(height * panel_width / width)
    if panel_height % 2:
        panel_height += 1
    return panel_width, panel_height


def _sample_indices(frames: list[dict[str, Any]]) -> set[int]:
    targets = (0.0, 1.0, 9.0, 14.0, 17.0, 25.8, 26.3, float(frames[-1]["timestamp"]))
    return {
        int(min(frames, key=lambda frame: abs(float(frame["timestamp"]) - target))["frame_idx"])
        for target in targets
    }
