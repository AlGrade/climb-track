"""Stage 30: raw Sapiens2 pose observations without temporal refinement."""

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from climbtrack.backends.sapiens2 import Sapiens2Backend
from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.errors import ClimbTrackError, ExternalToolError
from climbtrack.hashing import fingerprint_file, hash_json
from climbtrack.provenance import git_state, runtime_state
from climbtrack.schema.crops import read_pose_crops
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.pose import write_pose_parquet

STAGE_NAME = "30_pose"
STAGE_VERSION = "1.0.0"


def estimate_pose(
    ingest: CacheResult,
    selection: CacheResult,
    *,
    model_dir: Path,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Run pinned local Sapiens2-1B on the selected climber crop for every frame."""
    required = (
        model_dir / "config.json",
        model_dir / "preprocessor_config.json",
        model_dir / config.models.sapiens2.checkpoint_filename,
        model_dir / "keypoints.json",
        model_dir / "download.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ClimbTrackError(
            f"Sapiens2-1B is missing ({missing[0]}). Run 'climbtrack download-sapiens'."
        )
    model_files = {path.name: fingerprint_file(path) for path in required}
    effective_config = {
        "pose": config.pose.model_dump(mode="json"),
        "model": config.models.sapiens2.model_dump(mode="json"),
    }
    tools = {
        "torch": version("torch"),
        "transformers": version("transformers"),
        "model_files": model_files,
    }
    input_fingerprint = {
        "ingest": upstream_fingerprint(ingest.manifest),
        "selection": upstream_fingerprint(selection.manifest),
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
        crops = read_pose_crops(selection.path / "pose_crops.parquet")
        if len(frames) != len(crops):
            raise ClimbTrackError(
                f"Pose requires one crop per frame; found {len(crops)} for {len(frames)} frames"
            )
        backend = Sapiens2Backend(model_dir, device=config.project.device.value, config=config.pose)
        keypoints = backend.registry["keypoints"]
        records: list[dict[str, Any]] = []
        confidence_sums = {entry["group"]: 0.0 for entry in keypoints}
        confidence_counts = {entry["group"]: 0 for entry in keypoints}
        batch_size = config.pose.batch_size
        with Progress(
            TextColumn("30_pose"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Sapiens2-1B", total=len(frames))
            for start in range(0, len(frames), batch_size):
                frame_batch = frames[start : start + batch_size]
                crop_batch = crops[start : start + batch_size]
                images = []
                boxes = []
                for frame, crop in zip(frame_batch, crop_batch, strict=True):
                    if int(frame["frame_idx"]) != int(crop["frame_idx"]):
                        raise ClimbTrackError("Frame and pose-crop indices are not aligned")
                    source = ingest.path / str(frame["image_path"])
                    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ExternalToolError(f"Could not read decoded frame: {source}")
                    images.append(image)
                    boxes.append(_xywh(crop))
                predictions = backend.infer_batch(images, boxes)
                for frame, crop, prediction in zip(
                    frame_batch, crop_batch, predictions, strict=True
                ):
                    for entry, coordinate, confidence in zip(
                        keypoints,
                        prediction.coordinates,
                        prediction.confidence,
                        strict=True,
                    ):
                        value = float(confidence)
                        group = str(entry["group"])
                        confidence_sums[group] += value
                        confidence_counts[group] += 1
                        records.append(
                            {
                                "frame_idx": int(frame["frame_idx"]),
                                "timestamp": float(frame["timestamp"]),
                                "track_id": int(crop["track_id"]),
                                "keypoint_name": str(entry["name"]),
                                "x": float(coordinate[0]),
                                "y": float(coordinate[1]),
                                "confidence": value,
                                "is_missing": False,
                                "is_interpolated": False,
                                "source_backend": backend.source_backend,
                            }
                        )
                progress.advance(task, len(frame_batch))
        write_pose_parquet(records, output / "pose_raw.parquet")
        (output / "keypoints.json").write_text(
            json.dumps(backend.registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "backend": backend.source_backend,
            "frames": len(frames),
            "keypoints_per_frame": len(keypoints),
            "observations": len(records),
            "mean_confidence_by_group": {
                group: confidence_sums[group] / confidence_counts[group]
                for group in sorted(confidence_sums)
            },
            "raw_unrefined": True,
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


def _xywh(crop: dict[str, Any]) -> list[float]:
    x1, y1 = float(crop["x1"]), float(crop["y1"])
    return [x1, y1, float(crop["x2"]) - x1, float(crop["y2"]) - y1]
