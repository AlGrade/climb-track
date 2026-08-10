"""Offline temporal stabilization for the selected person's pose crop."""

from dataclasses import dataclass
from itertools import pairwise
from statistics import median
from typing import Any

from climbtrack.config import PoseCropConfig


@dataclass(frozen=True)
class _Measurement:
    center_x: float
    center_y: float
    side: float
    is_interpolated: bool


def build_pose_crops(
    frames: list[dict[str, Any]],
    track_rows: list[dict[str, Any]],
    *,
    track_id: int,
    image_width: int,
    image_height: int,
    config: PoseCropConfig,
) -> list[dict[str, Any]]:
    """Create padded square crops with centered median smoothing and short-gap filling."""
    selected = {
        int(row["frame_idx"]): row for row in track_rows if int(row["track_id"]) == track_id
    }
    if not selected:
        return []
    measurements = {
        frame_idx: _from_box(row, config.padding_scale) for frame_idx, row in selected.items()
    }
    _interpolate_short_gaps(measurements, config.maximum_interpolation_gap)
    half_window = config.smoothing_window // 2
    frame_lookup = {int(frame["frame_idx"]): frame for frame in frames}
    records: list[dict[str, Any]] = []
    for frame_idx in sorted(measurements):
        measurement = measurements[frame_idx]
        neighbors = [
            measurements[index]
            for index in range(frame_idx - half_window, frame_idx + half_window + 1)
            if index in measurements
        ]
        center_x = median(item.center_x for item in neighbors)
        center_y = median(item.center_y for item in neighbors)
        side = median(item.side for item in neighbors)
        source = selected.get(frame_idx)
        if source is not None:
            required_half = max(
                center_x - float(source["x1"]),
                float(source["x2"]) - center_x,
                center_y - float(source["y1"]),
                float(source["y2"]) - center_y,
            )
            side = max(side, required_half * 2.0 * 1.02)
        x1, y1, x2, y2 = _bounded_square(
            center_x,
            center_y,
            side,
            image_width=image_width,
            image_height=image_height,
        )
        frame = frame_lookup[frame_idx]
        records.append(
            {
                "frame_idx": frame_idx,
                "timestamp": frame["timestamp"],
                "track_id": track_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "is_interpolated": measurement.is_interpolated,
            }
        )
    return records


def _from_box(row: dict[str, Any], padding_scale: float) -> _Measurement:
    x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
    return _Measurement(
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        side=max(x2 - x1, y2 - y1) * padding_scale,
        is_interpolated=False,
    )


def _interpolate_short_gaps(measurements: dict[int, _Measurement], maximum_gap: int) -> None:
    observed = sorted(measurements)
    for left, right in pairwise(observed):
        gap = right - left - 1
        if gap <= 0 or gap > maximum_gap:
            continue
        start = measurements[left]
        end = measurements[right]
        for frame_idx in range(left + 1, right):
            fraction = (frame_idx - left) / (right - left)
            measurements[frame_idx] = _Measurement(
                center_x=start.center_x + fraction * (end.center_x - start.center_x),
                center_y=start.center_y + fraction * (end.center_y - start.center_y),
                side=start.side + fraction * (end.side - start.side),
                is_interpolated=True,
            )


def _bounded_square(
    center_x: float,
    center_y: float,
    side: float,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    side = min(side, float(image_width), float(image_height))
    x1 = min(max(center_x - side / 2.0, 0.0), image_width - side)
    y1 = min(max(center_y - side / 2.0, 0.0), image_height - side)
    return x1, y1, x1 + side, y1 + side
