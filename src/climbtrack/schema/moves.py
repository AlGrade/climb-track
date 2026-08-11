"""Canonical Parquet schema for manually reviewed climbing moves."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from climbtrack.errors import SchemaValidationError

MOVE_SCHEMA_VERSION = "1.1.0"
MOVE_SCHEMA = pa.schema(
    [
        pa.field("move_id", pa.int64(), nullable=False),
        pa.field("start_frame", pa.int64(), nullable=False),
        pa.field("end_frame", pa.int64(), nullable=False),
        pa.field("start_timestamp", pa.float64(), nullable=False),
        pa.field("end_timestamp", pa.float64(), nullable=False),
        pa.field("moving_hand", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("is_reviewed", pa.bool_(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
    ],
    metadata={b"climbtrack.schema": MOVE_SCHEMA_VERSION.encode()},
)


def write_moves_parquet(moves: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Atomically write canonical move rows."""
    records = [dict(move) for move in moves]
    table = pa.Table.from_pylist(records, schema=MOVE_SCHEMA)
    temporary = path.with_name(f".{path.name}.part")
    pq.write_table(table, temporary, compression="zstd", version="2.6")
    temporary.replace(path)


def read_moves_parquet(path: Path) -> list[dict[str, Any]]:
    """Read move rows and reject schema drift."""
    table = pq.read_table(path)
    if not table.schema.equals(MOVE_SCHEMA, check_metadata=True):
        raise SchemaValidationError(f"Unexpected move schema: {path}")
    return table.to_pylist()
