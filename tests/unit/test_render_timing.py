from pathlib import Path

import pytest

from climbtrack.rendering.video import frame_durations, write_concat_manifest


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


def test_concat_manifest_overrides_image_demuxer_framerate(tmp_path: Path) -> None:
    frames = [tmp_path / "000.jpg", tmp_path / "001.jpg"]
    manifest = tmp_path / "frames.ffconcat"

    write_concat_manifest(manifest, frames, [0.0167, 0.0168])

    text = manifest.read_text(encoding="utf-8")
    assert text.count("option framerate 1000000") == 3
    assert "duration 0.016700000" in text
    assert text.count(f"file '{frames[-1]}'") == 2
