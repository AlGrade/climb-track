"""Pinned adapter around Ultralytics ByteTrack 8.4.104."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from climbtrack.config import TrackingConfig


@dataclass(frozen=True)
class TrackedBox:
    """One active ByteTrack result for a frame."""

    track_id: int
    detection_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class ByteTrackAdapter:
    """Version-isolated ByteTrack wrapper accepting plain NumPy detections."""

    def __init__(self, config: TrackingConfig, image_shape: tuple[int, int]) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_high_thresh=config.track_high_threshold,
            track_low_thresh=config.track_low_threshold,
            new_track_thresh=config.new_track_threshold,
            track_buffer=config.track_buffer,
            match_thresh=config.match_threshold,
            fuse_score=config.fuse_score,
        )
        self._tracker = BYTETracker(args)
        self._image_shape = image_shape

    def update(self, detections: np.ndarray) -> tuple[TrackedBox, ...]:
        """Advance one frame; input columns are x1,y1,x2,y2,confidence,class."""
        from ultralytics.engine.results import Boxes

        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float32)
        boxes = Boxes(detections.astype(np.float32, copy=False), self._image_shape)
        output = self._tracker.update(boxes)
        if output.size == 0:
            return ()
        return tuple(
            TrackedBox(
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                track_id=int(row[4]),
                confidence=float(row[5]),
                class_id=int(row[6]),
                detection_index=int(row[7]),
            )
            for row in output
        )


def containment_keep_indices(detections: np.ndarray, threshold: float) -> tuple[int, ...]:
    """Drop a lower-confidence box that is almost contained by a stronger box."""
    if len(detections) < 2:
        return tuple(range(len(detections)))
    areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
    keep: list[int] = []
    for index, candidate in enumerate(detections):
        suppressed = False
        for other_index, other in enumerate(detections):
            stronger = other[4] > candidate[4] or (other[4] == candidate[4] and other_index < index)
            if not stronger:
                continue
            intersection_width = max(0.0, min(candidate[2], other[2]) - max(candidate[0], other[0]))
            intersection_height = max(
                0.0, min(candidate[3], other[3]) - max(candidate[1], other[1])
            )
            intersection = intersection_width * intersection_height
            if intersection / min(areas[index], areas[other_index]) >= threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(index)
    return tuple(keep)
