"""Stage 70: automatic hand-move segmentation from refined pose observations."""

import json
from pathlib import Path

from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.hashing import hash_json
from climbtrack.moves import detect_hand_moves
from climbtrack.provenance import git_state, runtime_state
from climbtrack.schema.moves import write_moves_parquet
from climbtrack.schema.pose import read_pose_parquet

STAGE_NAME = "70_moves"
STAGE_VERSION = "1.1.0"


def detect_moves(
    refined: CacheResult,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Detect moves deterministically without invoking a neural model."""
    effective_config = {"move_detection": config.move_detection.model_dump(mode="json")}
    input_fingerprint = {"refined_pose": upstream_fingerprint(refined.manifest)}
    cache = StageCache(cache_root, STAGE_NAME)
    cache_key = cache.make_key(
        stage=STAGE_NAME,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools={},
    )

    def build(output: Path) -> None:
        records = read_pose_parquet(refined.path / "pose_refined.parquet")
        result = detect_hand_moves(records, config.move_detection)
        write_moves_parquet(result.moves, output / "moves_auto.parquet")
        summary = {
            "schema_version": "1.1.0",
            "stage": STAGE_NAME,
            "config_hash": hash_json(effective_config),
            "diagnostics": result.diagnostics,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
