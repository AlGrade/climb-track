import pytest

from climbtrack.errors import ExternalToolError
from climbtrack.video.probe import parse_probe


def _probe(timestamps: list[float], *, rotation: int = 0) -> dict:
    return {
        "streams": [
            {
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "duration": "1.0",
                "tags": {"rotate": str(rotation)},
            }
        ],
        "frames": [
            {
                "best_effort_timestamp": index,
                "best_effort_timestamp_time": str(timestamp),
                "pkt_duration_time": "0.033333",
            }
            for index, timestamp in enumerate(timestamps)
        ],
        "format": {"duration": "1.0"},
    }


def test_rotation_changes_display_dimensions() -> None:
    result = parse_probe(_probe([0.0, 1 / 30, 2 / 30], rotation=90))

    assert result.metadata.display_width == 1080
    assert result.metadata.display_height == 1920
    assert result.metadata.rotation_degrees == 90


def test_detects_variable_frame_rate() -> None:
    result = parse_probe(_probe([0.0, 1 / 30, 0.09]))

    assert result.metadata.variable_frame_rate


def test_rejects_missing_timestamp() -> None:
    raw = _probe([0.0, 1 / 30])
    del raw["frames"][1]["best_effort_timestamp_time"]

    with pytest.raises(ExternalToolError, match="will not approximate"):
        parse_probe(raw)


def test_rejects_non_monotonic_timestamps() -> None:
    with pytest.raises(ExternalToolError, match="strictly increasing"):
        parse_probe(_probe([0.0, 0.04, 0.03]))
