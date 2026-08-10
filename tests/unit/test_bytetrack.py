import numpy as np

from climbtrack.backends.bytetrack import ByteTrackAdapter, containment_keep_indices
from climbtrack.config import TrackingConfig


def test_bytetrack_keeps_identity_across_nearby_frames() -> None:
    tracker = ByteTrackAdapter(TrackingConfig(), (100, 100))

    first = tracker.update(np.asarray([[10, 10, 30, 50, 0.9, 0]], dtype=np.float32))
    second = tracker.update(np.asarray([[11, 10, 31, 50, 0.9, 0]], dtype=np.float32))

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id == 1
    assert first[0].detection_index == 0


def test_bytetrack_accepts_empty_frames() -> None:
    tracker = ByteTrackAdapter(TrackingConfig(), (100, 100))

    assert tracker.update(np.empty((0, 6), dtype=np.float32)) == ()


def test_containment_suppression_removes_nested_duplicate() -> None:
    detections = np.asarray(
        [
            [10, 10, 50, 90, 0.9, 0],
            [10, 10, 30, 90, 0.2, 0],
            [60, 10, 90, 90, 0.8, 0],
        ],
        dtype=np.float32,
    )

    assert containment_keep_indices(detections, 0.9) == (0, 2)


def test_containment_suppression_keeps_overlapping_people() -> None:
    detections = np.asarray([[10, 10, 50, 90, 0.9, 0], [30, 10, 70, 90, 0.8, 0]], dtype=np.float32)

    assert containment_keep_indices(detections, 0.9) == (0, 1)
