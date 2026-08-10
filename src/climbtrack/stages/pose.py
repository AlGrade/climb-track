"""Stage 30: resumable raw Sapiens2 pose observations without refinement."""

import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from climbtrack.backends.sapiens2 import PosePrediction, Sapiens2Backend
from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.errors import ClimbTrackError, ExternalToolError
from climbtrack.hashing import fingerprint_file, hash_json
from climbtrack.provenance import git_state, runtime_state
from climbtrack.schema.crops import read_pose_crops
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.keypoints import read_registry
from climbtrack.schema.pose import (
    combine_pose_parquet,
    read_pose_table,
    write_pose_parquet,
)

STAGE_NAME = "30_pose"
STAGE_VERSION = "1.1.0"


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
    """Run pinned local Sapiens2-1B and checkpoint every completed frame."""
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
        if not frames:
            raise ClimbTrackError("Pose requires at least one decoded frame")
        registry = read_registry(model_dir / "keypoints.json")
        keypoints = registry["keypoints"]
        expected_names = {str(entry["name"]) for entry in keypoints}
        expected_observations = len(frames) * len(keypoints)
        parts_dir = output / ".pose-parts"
        if _final_outputs_complete(
            output,
            expected_observations,
            effective_config,
            registry,
        ):
            if parts_dir.exists():
                shutil.rmtree(parts_dir)
            return

        parts_dir.mkdir(exist_ok=True)
        completed = _completed_parts(parts_dir, frames, expected_names)
        backend: Sapiens2Backend | None = None
        batch_size = config.pose.batch_size
        with Progress(
            TextColumn("30_pose"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            description = "Sapiens2-1B"
            if completed:
                description += f" ({len(completed)} frames resumed)"
            task = progress.add_task(
                description,
                total=len(frames),
                completed=len(completed),
            )
            for start in range(0, len(frames), batch_size):
                indices = [
                    index
                    for index in range(start, min(start + batch_size, len(frames)))
                    if int(frames[index]["frame_idx"]) not in completed
                ]
                if not indices:
                    continue
                if backend is None:
                    backend = Sapiens2Backend(
                        model_dir,
                        device=config.project.device.value,
                        config=config.pose,
                    )
                frame_batch = [frames[index] for index in indices]
                crop_batch = [crops[index] for index in indices]
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
                    part = _part_path(parts_dir, int(frame["frame_idx"]))
                    temporary = part.with_suffix(".tmp")
                    write_pose_parquet(
                        _prediction_records(frame, crop, prediction, keypoints),
                        temporary,
                    )
                    temporary.replace(part)
                progress.advance(task, len(frame_batch))

        part_paths = [_part_path(parts_dir, int(frame["frame_idx"])) for frame in frames]
        observations = combine_pose_parquet(part_paths, output / "pose_raw.parquet")
        confidence_sums, confidence_counts = _confidence_totals(
            part_paths,
            {str(entry["name"]): str(entry["group"]) for entry in keypoints},
        )
        (output / "keypoints.json").write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "backend": Sapiens2Backend.source_backend,
            "frames": len(frames),
            "keypoints_per_frame": len(keypoints),
            "observations": observations,
            "mean_confidence_by_group": {
                group: confidence_sums[group] / confidence_counts[group]
                for group in sorted(confidence_sums)
            },
            "raw_unrefined": True,
            "resumable_frame_checkpoints": True,
            "config_hash": hash_json(effective_config),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.rmtree(parts_dir)

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
        resume_incomplete=True,
        builder=build,
    )


def _xywh(crop: dict[str, Any]) -> list[float]:
    x1, y1 = float(crop["x1"]), float(crop["y1"])
    return [x1, y1, float(crop["x2"]) - x1, float(crop["y2"]) - y1]


def _part_path(parts_dir: Path, frame_idx: int) -> Path:
    return parts_dir / f"{frame_idx:09d}.parquet"


def _completed_parts(
    parts_dir: Path,
    frames: list[dict[str, Any]],
    expected_names: set[str],
) -> set[int]:
    completed: set[int] = set()
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        path = _part_path(parts_dir, frame_idx)
        if not path.is_file():
            continue
        table = read_pose_table(path)
        names = set(table.column("keypoint_name").to_pylist())
        frame_indices = set(table.column("frame_idx").to_pylist())
        if table.num_rows != len(expected_names) or names != expected_names:
            raise ClimbTrackError(f"Resumable pose part is incomplete or invalid: {path}")
        if frame_indices != {frame_idx}:
            raise ClimbTrackError(f"Resumable pose part has the wrong frame index: {path}")
        completed.add(frame_idx)
    return completed


def _prediction_records(
    frame: dict[str, Any],
    crop: dict[str, Any],
    prediction: PosePrediction,
    keypoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "frame_idx": int(frame["frame_idx"]),
            "timestamp": float(frame["timestamp"]),
            "track_id": int(crop["track_id"]),
            "keypoint_name": str(entry["name"]),
            "x": float(coordinate[0]),
            "y": float(coordinate[1]),
            "confidence": float(confidence),
            "is_missing": False,
            "is_interpolated": False,
            "source_backend": Sapiens2Backend.source_backend,
        }
        for entry, coordinate, confidence in zip(
            keypoints,
            prediction.coordinates,
            prediction.confidence,
            strict=True,
        )
    ]


def _confidence_totals(
    part_paths: list[Path],
    groups: dict[str, str],
) -> tuple[dict[str, float], dict[str, int]]:
    sums = {group: 0.0 for group in set(groups.values())}
    counts = {group: 0 for group in sums}
    for path in part_paths:
        table = read_pose_table(path)
        names = table.column("keypoint_name").to_pylist()
        confidence = table.column("confidence").to_pylist()
        for name, value in zip(names, confidence, strict=True):
            group = groups[str(name)]
            sums[group] += float(value)
            counts[group] += 1
    return sums, counts


def _final_outputs_complete(
    output: Path,
    expected_observations: int,
    effective_config: dict[str, Any],
    registry: dict[str, Any],
) -> bool:
    pose_path = output / "pose_raw.parquet"
    registry_path = output / "keypoints.json"
    summary_path = output / "summary.json"
    if not pose_path.is_file() or not registry_path.is_file() or not summary_path.is_file():
        return False
    try:
        table = read_pose_table(pose_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        output_registry = read_registry(registry_path)
    except (ClimbTrackError, OSError, ValueError):
        return False
    return (
        table.num_rows == expected_observations
        and summary.get("observations") == expected_observations
        and summary.get("config_hash") == hash_json(effective_config)
        and output_registry == registry
    )
