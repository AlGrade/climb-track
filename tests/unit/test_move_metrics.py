from pathlib import Path

import pytest

from climbtrack.config import MoveMetricsConfig
from climbtrack.moves import calculate_move_metrics
from climbtrack.schema.move_metrics import (
    read_move_metrics_parquet,
    write_move_metrics_parquet,
)


def _record(frame: int, name: str, x: float, y: float) -> dict[str, object]:
    return {
        "frame_idx": frame,
        "timestamp": frame * 0.1,
        "track_id": 1,
        "keypoint_name": name,
        "x": x,
        "y": y,
        "confidence": 0.95,
        "is_missing": False,
        "is_interpolated": False,
        "source_backend": "test",
    }


def test_calculates_vfr_ready_hand_and_body_speeds(tmp_path: Path) -> None:
    records = []
    anchors = (
        "wrist",
        "thumb_third_joint",
        "forefinger_third_joint",
        "middle_finger_third_joint",
        "ring_finger_third_joint",
        "pinky_finger_third_joint",
    )
    body_points = (
        ("left_shoulder", -50.0, 0.0),
        ("right_shoulder", 50.0, 0.0),
        ("left_hip", -40.0, 100.0),
        ("right_hip", 40.0, 100.0),
        ("left_knee", -40.0, 160.0),
        ("right_knee", 40.0, 160.0),
        ("left_ankle", -40.0, 220.0),
        ("right_ankle", 40.0, 220.0),
    )
    for frame in range(11):
        for side, x in (("left", frame * 10.0), ("right", 0.0)):
            for anchor in anchors:
                records.append(_record(frame, f"{side}_{anchor}", x, 0.0))
        for name, x, y in body_points:
            records.append(_record(frame, name, x, y + frame * 2.0))

    result = calculate_move_metrics(
        records,
        [
            {
                "move_id": 1,
                "start_frame": 2,
                "end_frame": 8,
                "moving_hand": "left",
                "outcome": "completed",
            }
        ],
        MoveMetricsConfig(position_smoothing_radius=1, speed_window_radius=1),
    )

    assert result.diagnostics["body_length_px"] == pytest.approx(220.0)
    metric = result.metrics[0]
    assert metric["duration_seconds"] == pytest.approx(0.6)
    assert metric["hand_horizontal_displacement_px"] == pytest.approx(60.0)
    assert metric["hand_vertical_gain_px"] == pytest.approx(0.0)
    assert metric["hand_path_length_px"] == pytest.approx(60.0)
    assert metric["hand_mean_speed_px_s"] == pytest.approx(100.0)
    assert metric["hand_max_speed_px_s"] == pytest.approx(100.0)
    assert metric["hand_mean_speed_body_lengths_s"] == pytest.approx(100.0 / 220.0)
    assert metric["body_vertical_gain_px"] == pytest.approx(-12.0)
    assert metric["body_path_length_px"] == pytest.approx(12.0)
    assert metric["body_mean_speed_px_s"] == pytest.approx(20.0)
    assert metric["support_hand_relative_path_length_px"] == pytest.approx(12.0)

    path = tmp_path / "move_metrics.parquet"
    write_move_metrics_parquet(result.metrics, path)
    restored = read_move_metrics_parquet(path)
    assert restored[0]["move_id"] == 1
    assert restored[0]["hand_max_speed_px_s"] == pytest.approx(100.0)
