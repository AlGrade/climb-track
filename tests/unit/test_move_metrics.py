from pathlib import Path

import pytest

from climbtrack.config import MoveMetricsConfig
from climbtrack.moves import calculate_move_metrics
from climbtrack.schema.move_metrics import (
    read_move_metrics_parquet,
    read_move_speed_timeline_parquet,
    write_move_metrics_parquet,
    write_move_speed_timeline_parquet,
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

    assert result.diagnostics["body_length_px_median"] == pytest.approx(220.0)
    metric = result.metrics[0]
    assert metric["body_length_px"] == pytest.approx(220.0)
    assert metric["duration_seconds"] == pytest.approx(0.6)
    assert metric["hand_horizontal_displacement_px"] == pytest.approx(60.0)
    assert metric["hand_vertical_gain_px"] == pytest.approx(0.0)
    assert metric["hand_path_length_px"] == pytest.approx(60.0)
    assert metric["hand_mean_speed_px_s"] == pytest.approx(100.0)
    assert metric["hand_max_speed_px_s"] == pytest.approx(100.0)
    assert metric["hand_mean_speed_body_lengths_s"] == pytest.approx(100.0 / 220.0)
    assert metric["hand_max_speed_body_lengths_s"] == pytest.approx(100.0 / 220.0)
    assert metric["body_vertical_gain_px"] == pytest.approx(-12.0)
    assert metric["body_path_length_px"] == pytest.approx(12.0)
    assert metric["body_mean_speed_px_s"] == pytest.approx(20.0)
    assert metric["support_hand_relative_path_length_px"] == pytest.approx(12.0)
    assert len(result.speed_timeline) == 7
    assert result.diagnostics["speed_samples"] == 7
    assert result.speed_timeline[0]["frame_idx"] == 2
    assert result.speed_timeline[-1]["frame_idx"] == 8
    assert result.speed_timeline[3]["hand_speed_px_s"] == pytest.approx(100.0)

    path = tmp_path / "move_metrics.parquet"
    write_move_metrics_parquet(result.metrics, path)
    restored = read_move_metrics_parquet(path)
    assert restored[0]["move_id"] == 1
    assert restored[0]["hand_max_speed_px_s"] == pytest.approx(100.0)

    timeline_path = tmp_path / "move_speed_timeline.parquet"
    write_move_speed_timeline_parquet(result.speed_timeline, timeline_path)
    timeline = read_move_speed_timeline_parquet(timeline_path)
    assert timeline[0]["frame_idx"] == 2
    assert timeline[-1]["body_speed_px_s"] == pytest.approx(20.0)


def _scaled_body(frame: int, scale: float) -> list[dict[str, object]]:
    points = (
        ("left_shoulder", -50.0, 0.0),
        ("right_shoulder", 50.0, 0.0),
        ("left_hip", -40.0, 100.0),
        ("right_hip", 40.0, 100.0),
        ("left_knee", -40.0, 160.0),
        ("right_knee", 40.0, 160.0),
        ("left_ankle", -40.0, 220.0),
        ("right_ankle", 40.0, 220.0),
    )
    return [_record(frame, name, x * scale, y * scale) for name, x, y in points]


def test_normalizes_speed_by_the_local_body_length() -> None:
    """A climber who only grows in the image must not gain body lengths per second.

    A single body length for the whole video misreports every move recorded at a
    different distance from the camera, so the same real motion has to keep the
    same normalized speed while its pixel speed follows the apparent size.
    """
    anchors = (
        "wrist",
        "thumb_third_joint",
        "forefinger_third_joint",
        "middle_finger_third_joint",
        "ring_finger_third_joint",
        "pinky_finger_third_joint",
    )
    records: list[dict[str, object]] = []
    position = 0.0
    for frame in range(41):
        scale = 1.0 + frame * 0.02
        position += 10.0 * scale
        for anchor in anchors:
            records.append(_record(frame, f"left_{anchor}", position, 0.0))
            records.append(_record(frame, f"right_{anchor}", 0.0, 0.0))
        records.extend(_scaled_body(frame, scale))

    config = MoveMetricsConfig(
        position_smoothing_radius=1,
        speed_window_radius=1,
        body_length_smoothing_radius=2,
    )
    early = calculate_move_metrics(
        records,
        [{"move_id": 1, "start_frame": 5, "end_frame": 12, "moving_hand": "left"}],
        config,
    ).metrics[0]
    late = calculate_move_metrics(
        records,
        [{"move_id": 1, "start_frame": 28, "end_frame": 35, "moving_hand": "left"}],
        config,
    ).metrics[0]

    # The climber is visibly larger later, so the pixel speed must rise with them.
    assert late["hand_mean_speed_px_s"] > early["hand_mean_speed_px_s"] * 1.3
    assert late["body_length_px"] > early["body_length_px"] * 1.3
    # The real motion never changed, so the normalized speed has to stay put.
    assert late["hand_mean_speed_body_lengths_s"] == pytest.approx(
        early["hand_mean_speed_body_lengths_s"], rel=0.02
    )


def test_path_length_matches_the_travelled_steps() -> None:
    """The path column must count real travel, including motion that reverses.

    Integrating the smoothed speed instead reported a shorter path than the
    axis-wise sums printed next to it, so the two disagreed inside one row.
    """
    anchors = (
        "wrist",
        "thumb_third_joint",
        "forefinger_third_joint",
        "middle_finger_third_joint",
        "ring_finger_third_joint",
        "pinky_finger_third_joint",
    )
    records: list[dict[str, object]] = []
    for frame in range(21):
        # Out and back: 100 px away, 100 px home, with zero net displacement.
        offset = float(frame * 10) if frame <= 10 else float((20 - frame) * 10)
        for anchor in anchors:
            records.append(_record(frame, f"left_{anchor}", offset, 0.0))
            records.append(_record(frame, f"right_{anchor}", 0.0, 0.0))
        records.extend(_scaled_body(frame, 1.0))

    metric = calculate_move_metrics(
        records,
        [{"move_id": 1, "start_frame": 0, "end_frame": 20, "moving_hand": "left"}],
        MoveMetricsConfig(position_smoothing_radius=1, speed_window_radius=1),
    ).metrics[0]

    assert metric["hand_direct_displacement_px"] == pytest.approx(0.0, abs=1e-6)
    # The two path columns are the point: they must be one definition, not two.
    assert metric["hand_path_length_px"] == pytest.approx(metric["hand_horizontal_path_px"])
    # The hand travelled 200 px out and back. Median smoothing rounds the sharp
    # reversal and the segment ends, so the counted path settles just below that
    # figure while staying far away from the zero net displacement.
    assert 150.0 < metric["hand_path_length_px"] < 200.0
    assert metric["hand_mean_speed_px_s"] == pytest.approx(
        metric["hand_path_length_px"] / metric["duration_seconds"]
    )
