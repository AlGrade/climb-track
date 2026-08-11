"""Stage 40: conservative temporal repair and One Euro smoothing."""

import json
import shutil
from collections import Counter
from pathlib import Path

from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.hashing import hash_json
from climbtrack.provenance import git_state, runtime_state
from climbtrack.refinement import refine_pose_records
from climbtrack.schema.crops import read_pose_crops
from climbtrack.schema.keypoints import read_registry
from climbtrack.schema.pose import read_pose_parquet, write_pose_parquet

STAGE_NAME = "40_refine"
STAGE_VERSION = "1.0.0"


def refine_pose(
    selection: CacheResult,
    pose: CacheResult,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Refine cached raw observations without invoking the pose model again."""
    effective_config = {"refine": config.refine.model_dump(mode="json")}
    input_fingerprint = {
        "selection": upstream_fingerprint(selection.manifest),
        "pose": upstream_fingerprint(pose.manifest),
    }
    cache = StageCache(cache_root, STAGE_NAME)
    cache_key = cache.make_key(
        stage=STAGE_NAME,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools={},
    )

    def build(output: Path) -> None:
        raw = read_pose_parquet(pose.path / "pose_raw.parquet")
        registry = read_registry(pose.path / "keypoints.json")
        crop_scales = {
            int(crop["frame_idx"]): max(
                float(crop["x2"]) - float(crop["x1"]),
                float(crop["y2"]) - float(crop["y1"]),
            )
            for crop in read_pose_crops(selection.path / "pose_crops.parquet")
        }
        result = refine_pose_records(raw, registry, crop_scales, config.refine)
        write_pose_parquet(result.records, output / "pose_refined.parquet")
        shutil.copy2(pose.path / "keypoints.json", output / "keypoints.json")

        groups = {entry["name"]: entry["group"] for entry in registry["keypoints"]}
        missing_by_group = Counter(
            groups[record["keypoint_name"]]
            for record in result.records
            if bool(record["is_missing"])
        )
        interpolated_by_group = Counter(
            groups[record["keypoint_name"]]
            for record in result.records
            if bool(record["is_interpolated"])
        )
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "frames": len(crop_scales),
            "observations": len(result.records),
            "diagnostics": result.diagnostics,
            "missing_by_group": dict(sorted(missing_by_group.items())),
            "interpolated_by_group": dict(sorted(interpolated_by_group.items())),
            "filter": "one_euro_2d",
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
        tools={},
        runtime=runtime_state(),
        git=git_state(project_root),
        verify_checksums=True,
        force=force,
        builder=build,
    )
