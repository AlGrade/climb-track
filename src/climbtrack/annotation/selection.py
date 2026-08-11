"""Deterministic sampling of difficult and temporally representative frames."""

import math
from collections import defaultdict
from statistics import fmean, median
from typing import Any


def select_annotation_frames(
    frames: list[dict[str, Any]],
    pose_rows: list[dict[str, Any]],
    keypoint_names: set[str],
    *,
    count: int,
    minimum_spacing_seconds: float,
) -> list[int]:
    """Choose mostly difficult frames plus two timeline coverage anchors."""
    if not frames:
        raise ValueError("At least one video frame is required")
    if count > len(frames):
        count = len(frames)
    observations: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in pose_rows:
        name = str(row["keypoint_name"])
        if name in keypoint_names:
            observations[int(row["frame_idx"])][name] = row

    raw_confidence: dict[int, float] = {}
    raw_motion: dict[int, float] = {}
    previous: dict[str, dict[str, Any]] | None = None
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        current = observations.get(frame_idx, {})
        confidences = [float(row["confidence"]) for row in current.values()]
        raw_confidence[frame_idx] = 1.0 - fmean(confidences) if confidences else 1.0
        if previous:
            distances = [
                math.hypot(
                    float(current[name]["x"]) - float(previous[name]["x"]),
                    float(current[name]["y"]) - float(previous[name]["y"]),
                )
                for name in current.keys() & previous.keys()
            ]
            diagonal = _pose_diagonal((*current.values(), *previous.values()))
            raw_motion[frame_idx] = median(distances) / diagonal if distances else 0.0
        else:
            raw_motion[frame_idx] = 0.0
        previous = current

    confidence_rank = _percentile_ranks(raw_confidence)
    motion_rank = _percentile_ranks(raw_motion)
    difficulty = {
        frame_idx: 0.7 * confidence_rank[frame_idx] + 0.3 * motion_rank[frame_idx]
        for frame_idx in raw_confidence
    }
    timestamps = {int(frame["frame_idx"]): float(frame["timestamp"]) for frame in frames}
    ordered = [int(frame["frame_idx"]) for frame in frames]
    selected: list[int] = []

    start, end = timestamps[ordered[0]], timestamps[ordered[-1]]
    for fraction in (0.25, 0.75):
        target = start + (end - start) * fraction
        anchor = min(ordered, key=lambda idx: abs(timestamps[idx] - target))
        _try_add(anchor, selected, timestamps, 0.0)

    for frame_idx in sorted(ordered, key=lambda idx: (-difficulty[idx], idx)):
        if len(selected) >= count:
            break
        _try_add(frame_idx, selected, timestamps, minimum_spacing_seconds)

    for frame_idx in ordered:
        if len(selected) >= count:
            break
        _try_add(frame_idx, selected, timestamps, 0.0)
    return sorted(selected)


def _try_add(
    frame_idx: int,
    selected: list[int],
    timestamps: dict[int, float],
    minimum_spacing_seconds: float,
) -> None:
    if frame_idx in selected:
        return
    if any(
        abs(timestamps[frame_idx] - timestamps[existing]) < minimum_spacing_seconds
        for existing in selected
    ):
        return
    selected.append(frame_idx)


def _pose_diagonal(rows: tuple[dict[str, Any], ...]) -> float:
    if not rows:
        return 1.0
    xs = [float(row["x"]) for row in rows]
    ys = [float(row["y"]) for row in rows]
    return max(1.0, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))


def _percentile_ranks(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    denominator = max(1, len(ordered) - 1)
    return {frame_idx: rank / denominator for rank, (frame_idx, _) in enumerate(ordered)}
