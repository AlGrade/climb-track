import pytest

from climbtrack.config import SelectionConfig
from climbtrack.selection.scoring import rank_candidates


def _row(frame_idx: int, track_id: int, x: float, y: float) -> dict[str, float | int]:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 10,
        "track_id": track_id,
        "detection_idx": 0,
        "x1": x,
        "y1": y,
        "x2": x + 20,
        "y2": y + 40,
        "confidence": 0.9,
        "class_id": 0,
    }


def test_rank_candidates_rewards_long_continuous_moving_track() -> None:
    rows = [_row(frame, 1, 40, 70 - frame) for frame in range(10)]
    rows.extend(_row(frame, 2, 5, 5) for frame in (0, 4, 9))
    config = SelectionConfig(minimum_observations=5, minimum_continuity=0.8)

    candidates = rank_candidates(rows, image_width=100, image_height=100, config=config)

    assert [candidate.track_id for candidate in candidates] == [1, 2]
    assert candidates[0].eligible
    assert not candidates[1].eligible
    assert candidates[0].continuity == pytest.approx(1.0)


def test_rank_candidates_handles_empty_input() -> None:
    assert rank_candidates([], image_width=100, image_height=100, config=SelectionConfig()) == []
