"""Frame timeline Parquet schema."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from climbtrack.errors import SchemaValidationError
from climbtrack.video.probe import FrameTiming

FRAME_SCHEMA_VERSION = "1.0.0"
FRAME_SCHEMA = pa.schema(
    [
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=False),
        pa.field("duration", pa.float64(), nullable=True),
        pa.field("source_pts", pa.int64(), nullable=True),
        pa.field("image_path", pa.string(), nullable=False),
    ],
    metadata={b"climbtrack.schema": FRAME_SCHEMA_VERSION.encode()},
)


def write_frame_index(frames: tuple[FrameTiming, ...], path: Path) -> None:
    """Write source timestamps and stable image names."""
    records = [
        {
            "frame_idx": frame.frame_idx,
            "timestamp": frame.timestamp,
            "duration": frame.duration,
            "source_pts": frame.source_pts,
            "image_path": f"frames/{frame.frame_idx:09d}.png",
        }
        for frame in frames
    ]
    table = pa.Table.from_pylist(records, schema=FRAME_SCHEMA)
    pq.write_table(table, path, compression="zstd", version="2.6")


def read_frame_index(path: Path) -> list[dict[str, Any]]:
    """Read Stage-00 frames and verify the exact Arrow schema."""
    table = pq.read_table(path)
    if not table.schema.equals(FRAME_SCHEMA, check_metadata=True):
        raise SchemaValidationError(f"Unexpected frame-index schema: {path}")
    return table.to_pylist()
