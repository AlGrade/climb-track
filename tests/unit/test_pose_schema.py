from pathlib import Path

import pyarrow.parquet as pq
import pytest

from climbtrack.errors import SchemaValidationError
from climbtrack.schema.pose import POSE_SCHEMA_VERSION, write_pose_parquet


def _record(**changes):
    record = {
        "frame_idx": 0,
        "timestamp": 0.0,
        "track_id": 1,
        "keypoint_name": "left_wrist",
        "x": 10.0,
        "y": 20.0,
        "confidence": 0.9,
        "is_missing": False,
        "is_interpolated": False,
        "source_backend": "sapiens2",
    }
    record.update(changes)
    return record


def test_round_trip_preserves_explicit_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "pose.parquet"
    write_pose_parquet(
        [
            _record(),
            _record(
                frame_idx=1,
                x=None,
                y=None,
                confidence=None,
                is_missing=True,
            ),
        ],
        path,
    )

    table = pq.read_table(path)
    assert table.schema.metadata[b"climbtrack.schema"].decode() == POSE_SCHEMA_VERSION
    assert table.column("x").null_count == 1
    assert table.column("is_missing").to_pylist() == [False, True]


def test_missing_keypoint_cannot_contain_zero_coordinates(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="must store"):
        write_pose_parquet(
            [_record(x=0.0, y=0.0, confidence=0.0, is_missing=True)],
            tmp_path / "invalid.parquet",
        )


def test_confidence_is_bounded(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match=r"\[0, 1\]"):
        write_pose_parquet([_record(confidence=1.2)], tmp_path / "invalid.parquet")
