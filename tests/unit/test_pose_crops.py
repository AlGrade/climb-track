from pathlib import Path

import pytest

from climbtrack.config import PoseCropConfig
from climbtrack.schema.crops import read_pose_crops, write_pose_crops
from climbtrack.selection.crops import build_pose_crops


def _frame(frame_idx: int) -> dict[str, float | int | str | None]:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 60,
        "duration": 1 / 60,
        "source_pts": frame_idx,
        "image_path": f"frames/{frame_idx:09d}.png",
    }


def _track(frame_idx: int, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 60,
        "track_id": 1,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def test_pose_crop_is_square_padded_smoothed_and_fills_short_gap(tmp_path: Path) -> None:
    frames = [_frame(index) for index in range(5)]
    tracks = [
        _track(0, 40, 20, 60, 80),
        _track(1, 41, 20, 61, 80),
        _track(3, 80, 20, 100, 80),
        _track(4, 42, 20, 62, 80),
    ]
    config = PoseCropConfig(
        padding_scale=1.4,
        smoothing_window=3,
        maximum_interpolation_gap=2,
    )

    crops = build_pose_crops(
        frames,
        tracks,
        track_id=1,
        image_width=200,
        image_height=120,
        config=config,
    )

    assert len(crops) == 5
    assert crops[2]["is_interpolated"]
    assert crops[1]["x2"] - crops[1]["x1"] == pytest.approx(crops[1]["y2"] - crops[1]["y1"])
    assert crops[1]["x2"] - crops[1]["x1"] >= 60 * 1.4
    assert (crops[2]["x1"] + crops[2]["x2"]) / 2 == pytest.approx(70.5)

    path = tmp_path / "pose_crops.parquet"
    write_pose_crops(crops, path)
    assert read_pose_crops(path)[2]["is_interpolated"]


def test_pose_crop_stays_inside_image() -> None:
    crops = build_pose_crops(
        [_frame(0)],
        [_track(0, 0, 0, 30, 90)],
        track_id=1,
        image_width=100,
        image_height=120,
        config=PoseCropConfig(padding_scale=2.0, smoothing_window=1),
    )

    assert crops[0]["x1"] == 0
    assert crops[0]["y1"] >= 0
    assert crops[0]["x2"] <= 100
    assert crops[0]["y2"] <= 120
