from pathlib import Path

import pytest

from climbtrack.errors import SchemaValidationError
from climbtrack.schema.detections import read_detections, write_detections
from climbtrack.schema.tracks import read_tracks, write_tracks


def _detection(**changes):
    row = {
        "frame_idx": 0,
        "timestamp": 0.0,
        "detection_idx": 0,
        "x1": 10.0,
        "y1": 20.0,
        "x2": 30.0,
        "y2": 60.0,
        "confidence": 0.9,
        "class_id": 0,
        "class_name": "person",
    }
    row.update(changes)
    return row


def _track(**changes):
    row = _detection(track_id=1)
    row.pop("class_name")
    row.update(changes)
    return row


def test_detection_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "detections.parquet"
    write_detections([_detection()], path)

    rows = read_detections(path)

    assert rows[0]["class_name"] == "person"
    assert rows[0]["confidence"] == pytest.approx(0.9)


def test_track_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tracks.parquet"
    write_tracks([_track()], path)

    rows = read_tracks(path)

    assert rows[0]["track_id"] == 1
    assert rows[0]["x2"] == pytest.approx(30.0)


def test_schemas_reject_non_person_and_invalid_track(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="person"):
        write_detections([_detection(class_id=1)], tmp_path / "detections.parquet")
    with pytest.raises(SchemaValidationError, match="positive"):
        write_tracks([_track(track_id=0)], tmp_path / "tracks.parquet")
