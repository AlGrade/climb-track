import pytest

from climbtrack.rendering.video import frame_durations


def test_frame_durations_preserve_variable_timestamps() -> None:
    frames = [
        {"timestamp": 0.0, "duration": 0.04},
        {"timestamp": 0.04, "duration": 0.06},
        {"timestamp": 0.10, "duration": None},
    ]

    durations = frame_durations(frames)

    assert durations == pytest.approx([0.04, 0.06, 0.05])


def test_single_frame_requires_its_own_duration() -> None:
    assert frame_durations([{"timestamp": 0.0, "duration": 0.25}]) == [0.25]

    with pytest.raises(ValueError, match="positive duration"):
        frame_durations([{"timestamp": 0.0, "duration": None}])
