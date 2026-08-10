"""Stage 20: ByteTrack person tracklets from cached detections."""

import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from climbtrack.backends.bytetrack import ByteTrackAdapter, containment_keep_indices
from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.device import configure_ultralytics
from climbtrack.provenance import git_state, runtime_state
from climbtrack.schema.detections import read_detections
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.tracks import write_tracks

STAGE_NAME = "20_track"
STAGE_VERSION = "1.0.0"


def track_people(
    ingest: CacheResult,
    detections: CacheResult,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Associate Stage-10 boxes in exact timestamp/frame order."""
    configure_ultralytics(cache_root)
    effective_config = config.tracking.model_dump(mode="json")
    tools = {
        "lap": version("lap"),
        "numpy": version("numpy"),
        "ultralytics": version("ultralytics"),
    }
    input_fingerprint = upstream_fingerprint(detections.manifest)
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
        detection_rows = read_detections(detections.path / "detections.parquet")
        by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in detection_rows:
            by_frame[int(row["frame_idx"])].append(row)
        metadata = json.loads((ingest.path / "metadata.json").read_text(encoding="utf-8"))
        image_shape = (
            int(metadata["video"]["display_height"]),
            int(metadata["video"]["display_width"]),
        )
        tracker = ByteTrackAdapter(config.tracking, image_shape)
        records: list[dict[str, object]] = []

        with Progress(TextColumn("20_track"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("track", total=len(frames))
            for frame in frames:
                frame_idx = int(frame["frame_idx"])
                frame_detections = sorted(
                    by_frame.get(frame_idx, []), key=lambda row: int(row["detection_idx"])
                )
                matrix = np.asarray(
                    [
                        [
                            row["x1"],
                            row["y1"],
                            row["x2"],
                            row["y2"],
                            row["confidence"],
                            row["class_id"],
                        ]
                        for row in frame_detections
                    ],
                    dtype=np.float32,
                )
                if not frame_detections:
                    matrix = np.empty((0, 6), dtype=np.float32)
                keep = containment_keep_indices(matrix, config.tracking.containment_threshold)
                matrix = matrix[list(keep)]
                frame_detections = [frame_detections[index] for index in keep]
                for tracked in tracker.update(matrix):
                    if tracked.detection_index >= len(frame_detections):
                        raise RuntimeError(
                            "ByteTrack returned a detection index outside the input frame"
                        )
                    source = frame_detections[tracked.detection_index]
                    records.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp": frame["timestamp"],
                            "track_id": tracked.track_id,
                            "detection_idx": source["detection_idx"],
                            "x1": tracked.x1,
                            "y1": tracked.y1,
                            "x2": tracked.x2,
                            "y2": tracked.y2,
                            "confidence": tracked.confidence,
                            "class_id": tracked.class_id,
                        }
                    )
                progress.advance(task)

        write_tracks(records, output / "tracks.parquet")
        track_ids = sorted({int(record["track_id"]) for record in records})
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "frames": len(frames),
            "observations": len(records),
            "track_count": len(track_ids),
            "track_ids": track_ids,
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
