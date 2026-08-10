"""Canonical selected-person crop schema."""

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from climbtrack.errors import SchemaValidationError

POSE_CROP_SCHEMA_VERSION = "1.0.0"
POSE_CROP_SCHEMA = pa.schema(
    [
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=False),
        pa.field("track_id", pa.int64(), nullable=False),
        pa.field("x1", pa.float32(), nullable=False),
        pa.field("y1", pa.float32(), nullable=False),
        pa.field("x2", pa.float32(), nullable=False),
        pa.field("y2", pa.float32(), nullable=False),
        pa.field("is_interpolated", pa.bool_(), nullable=False),
    ],
    metadata={b"climbtrack.schema": POSE_CROP_SCHEMA_VERSION.encode()},
)


def validate_pose_crop(record: Mapping[str, Any]) -> None:
    """Validate one selected-person crop."""
    values = [record[name] for name in ("timestamp", "x1", "y1", "x2", "y2")]
    if not all(math.isfinite(float(value)) for value in values):
        raise SchemaValidationError("Pose-crop values must be finite")
    if int(record["frame_idx"]) < 0 or int(record["track_id"]) <= 0:
        raise SchemaValidationError("Pose-crop frame and track IDs are invalid")
    width = float(record["x2"]) - float(record["x1"])
    height = float(record["y2"]) - float(record["y1"])
    if width <= 0 or height <= 0:
        raise SchemaValidationError("Pose crop must have positive width and height")
    if not math.isclose(width, height, rel_tol=1e-5, abs_tol=1e-3):
        raise SchemaValidationError("Pose crop must be square")


def write_pose_crops(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Validate and write selected-person crops."""
    materialized = [dict(record) for record in records]
    for record in materialized:
        validate_pose_crop(record)
    table = pa.Table.from_pylist(materialized, schema=POSE_CROP_SCHEMA)
    pq.write_table(table, path, compression="zstd", version="2.6")


def read_pose_crops(path: Path) -> list[dict[str, Any]]:
    """Read selected-person crops and verify the exact Arrow schema."""
    table = pq.read_table(path)
    if not table.schema.equals(POSE_CROP_SCHEMA, check_metadata=True):
        raise SchemaValidationError(f"Unexpected pose-crop schema: {path}")
    return table.to_pylist()
