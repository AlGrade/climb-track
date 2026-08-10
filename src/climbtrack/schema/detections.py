"""Canonical Stage-10 person-detection schema."""

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from climbtrack.errors import SchemaValidationError

DETECTION_SCHEMA_VERSION = "1.0.0"
DETECTION_SCHEMA = pa.schema(
    [
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=False),
        pa.field("detection_idx", pa.int32(), nullable=False),
        pa.field("x1", pa.float32(), nullable=False),
        pa.field("y1", pa.float32(), nullable=False),
        pa.field("x2", pa.float32(), nullable=False),
        pa.field("y2", pa.float32(), nullable=False),
        pa.field("confidence", pa.float32(), nullable=False),
        pa.field("class_id", pa.int16(), nullable=False),
        pa.field("class_name", pa.string(), nullable=False),
    ],
    metadata={b"climbtrack.schema": DETECTION_SCHEMA_VERSION.encode()},
)


def validate_detection(record: Mapping[str, Any]) -> None:
    """Validate one person-detection record."""
    values = [record[name] for name in ("timestamp", "x1", "y1", "x2", "y2", "confidence")]
    if not all(math.isfinite(float(value)) for value in values):
        raise SchemaValidationError("Detection values must be finite")
    if int(record["frame_idx"]) < 0 or int(record["detection_idx"]) < 0:
        raise SchemaValidationError("Detection indices must be non-negative")
    if float(record["x2"]) <= float(record["x1"]) or float(record["y2"]) <= float(record["y1"]):
        raise SchemaValidationError("Detection box must have positive width and height")
    if not 0.0 <= float(record["confidence"]) <= 1.0:
        raise SchemaValidationError("Detection confidence must be in [0, 1]")
    if int(record["class_id"]) != 0 or record["class_name"] != "person":
        raise SchemaValidationError("Stage 10 may contain only COCO person detections")


def write_detections(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Validate and write Stage-10 detections."""
    materialized = [dict(record) for record in records]
    for record in materialized:
        validate_detection(record)
    table = pa.Table.from_pylist(materialized, schema=DETECTION_SCHEMA)
    pq.write_table(table, path, compression="zstd", version="2.6")


def read_detections(path: Path) -> list[dict[str, Any]]:
    """Read Stage-10 detections and verify the exact Arrow schema."""
    table = pq.read_table(path)
    if not table.schema.equals(DETECTION_SCHEMA, check_metadata=True):
        raise SchemaValidationError(f"Unexpected detection schema: {path}")
    return table.to_pylist()
