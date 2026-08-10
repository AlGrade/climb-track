"""Stage 10: YOLO11x person detection."""

import json
from importlib.metadata import version
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from climbtrack.backends.yolo11 import Yolo11PersonDetector
from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.device import configure_ultralytics
from climbtrack.hashing import fingerprint_file
from climbtrack.provenance import git_state, runtime_state
from climbtrack.schema.detections import write_detections
from climbtrack.schema.frames import read_frame_index

STAGE_NAME = "10_detect"
STAGE_VERSION = "1.0.0"


def detect_people(
    ingest: CacheResult,
    *,
    model_path: Path,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Run YOLO11x on every cached frame and retain only COCO person boxes."""
    configure_ultralytics(cache_root)
    model_fingerprint = fingerprint_file(model_path)
    effective_config = {
        **config.detection.model_dump(mode="json"),
        "device": config.project.device.value,
        "class_ids": [0],
        "seed": config.project.seed,
    }
    tools = {
        "model": model_fingerprint,
        "torch": version("torch"),
        "torchvision": version("torchvision"),
        "ultralytics": version("ultralytics"),
    }
    input_fingerprint = upstream_fingerprint(ingest.manifest)
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
        frames = read_frame_index(ingest.path / "frames.parquet")
        image_paths = [ingest.path / str(frame["image_path"]) for frame in frames]
        detector = Yolo11PersonDetector(
            model_path,
            config.detection,
            config.project.device,
            config.project.seed,
        )
        records: list[dict[str, object]] = []
        processed = 0
        with Progress(
            TextColumn("10_detect"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("detect", total=len(frames))
            for frame, detections in zip(frames, detector.predict(image_paths), strict=True):
                for detection in detections:
                    records.append(
                        {
                            "frame_idx": frame["frame_idx"],
                            "timestamp": frame["timestamp"],
                            "detection_idx": detection.detection_idx,
                            "x1": detection.x1,
                            "y1": detection.y1,
                            "x2": detection.x2,
                            "y2": detection.y2,
                            "confidence": detection.confidence,
                            "class_id": detection.class_id,
                            "class_name": "person",
                        }
                    )
                processed += 1
                progress.advance(task)
        if processed != len(frames):
            raise RuntimeError(f"YOLO returned {processed} frame results for {len(frames)} inputs")
        write_detections(records, output / "detections.parquet")
        frames_with_people = len({int(record["frame_idx"]) for record in records})
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "frames": len(frames),
            "frames_with_people": frames_with_people,
            "detections": len(records),
            "model": model_fingerprint,
            "effective_config": effective_config,
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
