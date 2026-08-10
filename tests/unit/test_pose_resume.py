from pathlib import Path

import pytest

from climbtrack.errors import ClimbTrackError
from climbtrack.schema.pose import write_pose_parquet
from climbtrack.stages.pose import _completed_parts


def _record(frame_idx: int, name: str) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp": float(frame_idx),
        "track_id": 1,
        "keypoint_name": name,
        "x": 10.0,
        "y": 20.0,
        "confidence": 0.9,
        "is_missing": False,
        "is_interpolated": False,
        "source_backend": "sapiens2",
    }


def test_completed_pose_parts_are_discovered(tmp_path: Path) -> None:
    frames = [{"frame_idx": 0}, {"frame_idx": 1}]
    write_pose_parquet(
        [_record(0, "left_wrist"), _record(0, "right_wrist")],
        tmp_path / "000000000.parquet",
    )

    completed = _completed_parts(tmp_path, frames, {"left_wrist", "right_wrist"})

    assert completed == {0}


def test_invalid_pose_part_is_never_silently_reused(tmp_path: Path) -> None:
    frames = [{"frame_idx": 0}]
    write_pose_parquet([_record(0, "left_wrist")], tmp_path / "000000000.parquet")

    with pytest.raises(ClimbTrackError, match="incomplete or invalid"):
        _completed_parts(tmp_path, frames, {"left_wrist", "right_wrist"})
