"""Canonical Parquet schema for per-move 2D kinematics."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from climbtrack.errors import SchemaValidationError

MOVE_METRICS_SCHEMA_VERSION = "1.1.0"
MOVE_METRICS_SCHEMA = pa.schema(
    [
        pa.field("move_id", pa.int64(), nullable=False),
        pa.field("moving_hand", pa.string(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("start_timestamp", pa.float64(), nullable=False),
        pa.field("end_timestamp", pa.float64(), nullable=False),
        pa.field("duration_seconds", pa.float64(), nullable=False),
        pa.field("body_length_px", pa.float64(), nullable=False),
        pa.field("hand_valid_fraction", pa.float64(), nullable=False),
        pa.field("hand_horizontal_displacement_px", pa.float64(), nullable=False),
        pa.field("hand_vertical_gain_px", pa.float64(), nullable=False),
        pa.field("hand_horizontal_path_px", pa.float64(), nullable=False),
        pa.field("hand_vertical_path_px", pa.float64(), nullable=False),
        pa.field("hand_direct_displacement_px", pa.float64(), nullable=False),
        pa.field("hand_path_length_px", pa.float64(), nullable=False),
        pa.field("hand_mean_speed_px_s", pa.float64(), nullable=False),
        pa.field("hand_max_speed_px_s", pa.float64(), nullable=False),
        pa.field("hand_mean_speed_body_lengths_s", pa.float64(), nullable=False),
        pa.field("hand_max_speed_body_lengths_s", pa.float64(), nullable=False),
        pa.field("hand_peak_speed_timestamp", pa.float64(), nullable=False),
        pa.field("hand_peak_speed_offset_seconds", pa.float64(), nullable=False),
        pa.field("body_valid_fraction", pa.float64(), nullable=False),
        pa.field("body_horizontal_displacement_px", pa.float64(), nullable=False),
        pa.field("body_vertical_gain_px", pa.float64(), nullable=False),
        pa.field("body_horizontal_path_px", pa.float64(), nullable=False),
        pa.field("body_vertical_path_px", pa.float64(), nullable=False),
        pa.field("body_direct_displacement_px", pa.float64(), nullable=False),
        pa.field("body_path_length_px", pa.float64(), nullable=False),
        pa.field("body_mean_speed_px_s", pa.float64(), nullable=False),
        pa.field("body_max_speed_px_s", pa.float64(), nullable=False),
        pa.field("body_mean_speed_body_lengths_s", pa.float64(), nullable=False),
        pa.field("body_max_speed_body_lengths_s", pa.float64(), nullable=False),
        pa.field("body_peak_speed_timestamp", pa.float64(), nullable=False),
        pa.field("body_peak_speed_offset_seconds", pa.float64(), nullable=False),
        pa.field("support_hand_relative_displacement_px", pa.float64(), nullable=True),
        pa.field("support_hand_relative_path_length_px", pa.float64(), nullable=True),
        # Where the moving hand comes to rest: the grasp for a completed move,
        # the bottom of the fall for a failed one.
        pa.field("hand_settle_frame", pa.int64(), nullable=False),
        pa.field("hand_settle_timestamp", pa.float64(), nullable=False),
        pa.field("hand_settle_offset_seconds", pa.float64(), nullable=False),
        pa.field("hip_rise_body_lengths", pa.float64(), nullable=False),
        pa.field("hip_below_hand_body_lengths", pa.float64(), nullable=False),
        # Lead of the torso over the moving hand. Null when one of the two speed
        # curves is flat, which leaves the lag undefined rather than zero.
        pa.field("coordination_lag_seconds", pa.float64(), nullable=True),
        pa.field("coordination_correlation", pa.float64(), nullable=True),
    ],
    metadata={b"climbtrack.schema": MOVE_METRICS_SCHEMA_VERSION.encode()},
)

MOVE_SPEED_TIMELINE_SCHEMA_VERSION = "1.0.0"
MOVE_SPEED_TIMELINE_SCHEMA = pa.schema(
    [
        pa.field("move_id", pa.int64(), nullable=False),
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=False),
        pa.field("offset_seconds", pa.float64(), nullable=False),
        pa.field("hand_speed_px_s", pa.float64(), nullable=False),
        pa.field("hand_speed_body_lengths_s", pa.float64(), nullable=False),
        pa.field("body_speed_px_s", pa.float64(), nullable=False),
        pa.field("body_speed_body_lengths_s", pa.float64(), nullable=False),
    ],
    metadata={b"climbtrack.schema": MOVE_SPEED_TIMELINE_SCHEMA_VERSION.encode()},
)


def write_move_metrics_parquet(metrics: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Atomically write per-move kinematics."""
    table = pa.Table.from_pylist([dict(row) for row in metrics], schema=MOVE_METRICS_SCHEMA)
    temporary = path.with_name(f".{path.name}.part")
    pq.write_table(table, temporary, compression="zstd", version="2.6")
    temporary.replace(path)


def read_move_metrics_parquet(path: Path) -> list[dict[str, Any]]:
    """Read kinematics and reject schema drift."""
    table = pq.read_table(path)
    if not table.schema.equals(MOVE_METRICS_SCHEMA, check_metadata=True):
        raise SchemaValidationError(f"Unexpected move metrics schema: {path}")
    return table.to_pylist()


def write_move_speed_timeline_parquet(samples: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Atomically write per-frame hand and body speed samples."""
    table = pa.Table.from_pylist(
        [dict(sample) for sample in samples], schema=MOVE_SPEED_TIMELINE_SCHEMA
    )
    temporary = path.with_name(f".{path.name}.part")
    pq.write_table(table, temporary, compression="zstd", version="2.6")
    temporary.replace(path)


def read_move_speed_timeline_parquet(path: Path) -> list[dict[str, Any]]:
    """Read per-frame speeds and reject schema drift."""
    table = pq.read_table(path)
    if not table.schema.equals(MOVE_SPEED_TIMELINE_SCHEMA, check_metadata=True):
        raise SchemaValidationError(f"Unexpected move speed timeline schema: {path}")
    return table.to_pylist()
