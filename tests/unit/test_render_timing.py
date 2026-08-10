import pytest

from climbtrack.stages.render_tracks import _frame_durations


def test_frame_durations_preserve_variable_timestamps() -> None:
    frames = [
        {"timestamp": 0.0, "duration": 0.04},
        {"timestamp": 0.04, "duration": 0.06},
        {"timestamp": 0.10, "duration": None},
    ]

    durations = _frame_durations(frames)

    assert durations == pytest.approx([0.04, 0.06, 0.05])
