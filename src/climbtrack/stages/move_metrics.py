"""Stage 80: per-move hand and body kinematics."""

import json
from pathlib import Path

from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.hashing import fingerprint_file, hash_json
from climbtrack.moves.metrics import calculate_move_metrics
from climbtrack.provenance import git_state, runtime_state
from climbtrack.schema.move_metrics import (
    write_move_metrics_parquet,
    write_move_speed_timeline_parquet,
)
from climbtrack.schema.moves import read_moves_parquet
from climbtrack.schema.pose import read_pose_parquet

STAGE_NAME = "80_move_metrics"
STAGE_VERSION = "2.1.0"


def measure_moves(
    refined: CacheResult,
    moves_path: Path,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Calculate deterministic 2D kinematics for the current editable move set."""
    effective_config = {"move_metrics": config.move_metrics.model_dump(mode="json")}
    input_fingerprint = {
        "refined_pose": upstream_fingerprint(refined.manifest),
        "moves": fingerprint_file(moves_path),
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
        pose_records = read_pose_parquet(refined.path / "pose_refined.parquet")
        moves = read_moves_parquet(moves_path)
        result = calculate_move_metrics(pose_records, moves, config.move_metrics)
        write_move_metrics_parquet(result.metrics, output / "move_metrics.parquet")
        write_move_speed_timeline_parquet(
            result.speed_timeline,
            output / "move_speed_timeline.parquet",
        )
        (output / "move_metrics.json").write_text(
            json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": "1.0.0",
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
