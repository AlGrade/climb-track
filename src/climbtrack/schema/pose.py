"""Canonical long-form pose observation schema."""

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from climbtrack.errors import SchemaValidationError

POSE_SCHEMA_VERSION = "1.0.0"
POSE_SCHEMA = pa.schema(
    [
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=False),
        pa.field("track_id", pa.int64(), nullable=False),
        pa.field("keypoint_name", pa.string(), nullable=False),
        pa.field("x", pa.float32(), nullable=True),
        pa.field("y", pa.float32(), nullable=True),
        pa.field("confidence", pa.float32(), nullable=True),
        pa.field("is_missing", pa.bool_(), nullable=False),
        pa.field("is_interpolated", pa.bool_(), nullable=False),
        pa.field("source_backend", pa.string(), nullable=False),
    ],
    metadata={b"climbtrack.schema": POSE_SCHEMA_VERSION.encode()},
)


def validate_pose_record(record: Mapping[str, Any]) -> None:
    """Enforce missing-value and confidence invariants for one observation."""
    required = {field.name for field in POSE_SCHEMA}
    missing_fields = required - record.keys()
    if missing_fields:
        raise SchemaValidationError(f"Pose record lacks fields: {sorted(missing_fields)}")
    unexpected_fields = record.keys() - required
    if unexpected_fields:
        raise SchemaValidationError(f"Pose record has unknown fields: {sorted(unexpected_fields)}")
    if int(record["frame_idx"]) < 0 or int(record["track_id"]) < 0:
        raise SchemaValidationError("frame_idx and track_id must be non-negative")
    if not str(record["keypoint_name"]).strip() or not str(record["source_backend"]).strip():
        raise SchemaValidationError("keypoint_name and source_backend must be non-empty")
    if not math.isfinite(float(record["timestamp"])):
        raise SchemaValidationError("timestamp must be finite")

    values = (record["x"], record["y"], record["confidence"])
    if bool(record["is_missing"]):
        if any(value is not None for value in values):
            raise SchemaValidationError("Missing keypoints must store x, y and confidence as null")
        if bool(record["is_interpolated"]):
            raise SchemaValidationError("A missing keypoint cannot be marked interpolated")
        return

    if any(value is None for value in values):
        raise SchemaValidationError("Observed keypoints require x, y and confidence")
    if not all(math.isfinite(float(value)) for value in values):
        raise SchemaValidationError("Coordinates and confidence must be finite")
    confidence = float(record["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise SchemaValidationError("confidence must be in [0, 1]")


def write_pose_parquet(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Validate and write canonical observations with explicit Arrow nulls."""
    materialized = [dict(record) for record in records]
    for record in materialized:
        validate_pose_record(record)
    table = pa.Table.from_pylist(materialized, schema=POSE_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", version="2.6")
