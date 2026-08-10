"""Offline temporal stabilization for the selected person's pose crop."""

from dataclasses import dataclass
from itertools import pairwise
from statistics import fmean
from typing import Any

from climbtrack.config import PoseCropConfig


@dataclass(frozen=True)
class _Measurement:
    center_x: float
    center_y: float
    width: float
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
    """Create padded 3:4 crops from a local box envelope and centered smoothing."""
    selected = {
        int(row["frame_idx"]): row for row in track_rows if int(row["track_id"]) == track_id
    }
    if not selected:
        return []
    allowed = _allowed_frames(selected, config.maximum_interpolation_gap)
    context_radius = config.context_window // 2
    aspect = config.input_width / config.input_height
    measurements: dict[int, _Measurement] = {}
    for frame_idx, is_interpolated in allowed.items():
        neighbors = [
            row
            for index, row in selected.items()
            if frame_idx - context_radius <= index <= frame_idx + context_radius
        ]
        x1 = min(float(row["x1"]) for row in neighbors)
        y1 = min(float(row["y1"]) for row in neighbors)
        x2 = max(float(row["x2"]) for row in neighbors)
        y2 = max(float(row["y2"]) for row in neighbors)
        width = (x2 - x1) * config.padding_scale
        height = (y2 - y1) * config.padding_scale
        width = max(width, height * aspect)
        measurements[frame_idx] = _Measurement(
            center_x=(x1 + x2) / 2.0,
            center_y=(y1 + y2) / 2.0,
            width=width,
            is_interpolated=is_interpolated,
        )

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
        center_x = fmean(item.center_x for item in neighbors)
        center_y = fmean(item.center_y for item in neighbors)
        width = fmean(item.width for item in neighbors)
        source = selected.get(frame_idx)
        if source is not None:
            required_half_width = max(
                center_x - float(source["x1"]), float(source["x2"]) - center_x
            )
            required_half_height = max(
                center_y - float(source["y1"]), float(source["y2"]) - center_y
            )
            width = max(
                width,
                required_half_width * 2.0 * 1.02,
                required_half_height * 2.0 * aspect * 1.02,
            )
        x1, y1, x2, y2 = _bounded_aspect_box(
            center_x,
            center_y,
            width,
            aspect=aspect,
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


def _allowed_frames(selected: dict[int, dict[str, Any]], maximum_gap: int) -> dict[int, bool]:
    allowed = dict.fromkeys(selected, False)
    for left, right in pairwise(sorted(selected)):
        gap = right - left - 1
        if 0 < gap <= maximum_gap:
            for frame_idx in range(left + 1, right):
                allowed[frame_idx] = True
    return allowed


def _bounded_aspect_box(
    center_x: float,
    center_y: float,
    width: float,
    *,
    aspect: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    width = min(width, float(image_width), float(image_height) * aspect)
    height = width / aspect
    x1 = min(max(center_x - width / 2.0, 0.0), image_width - width)
    y1 = min(max(center_y - height / 2.0, 0.0), image_height - height)
    return x1, y1, x1 + width, y1 + height
