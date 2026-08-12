"""Per-move 2D kinematics from refined pose timelines."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from climbtrack.config import MoveMetricsConfig
from climbtrack.errors import ClimbTrackError

HAND_ANCHORS = {
    side: (
        f"{side}_wrist",
        f"{side}_thumb_third_joint",
        f"{side}_forefinger_third_joint",
        f"{side}_middle_finger_third_joint",
        f"{side}_ring_finger_third_joint",
        f"{side}_pinky_finger_third_joint",
    )
    for side in ("left", "right")
}
BODY_POINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class MoveMetricsResult:
    """Canonical summary rows, per-frame speeds, and calculation diagnostics."""

    metrics: list[dict[str, Any]]
    speed_timeline: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def calculate_move_metrics(
    pose_records: list[dict[str, Any]],
    moves: list[dict[str, Any]],
    config: MoveMetricsConfig,
) -> MoveMetricsResult:
    """Calculate smoothed hand and torso kinematics for every move."""
    if not pose_records:
        raise ClimbTrackError("Move metrics require refined pose observations")
    if not moves:
        raise ClimbTrackError("Move metrics require at least one move")

    timestamps, coordinates = _timeline(pose_records)
    palms: dict[str, np.ndarray] = {}
    palm_valid: dict[str, np.ndarray] = {}
    for side, anchors in HAND_ANCHORS.items():
        palms[side], palm_valid[side] = _robust_center(coordinates, anchors, minimum_points=1)
        palms[side] = _smooth(_interpolate(palms[side]), config.position_smoothing_radius)

    torso, torso_valid = _robust_center(
        coordinates,
        ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        minimum_points=2,
    )
    torso = _smooth(_interpolate(torso), config.position_smoothing_radius)
    body_lengths = _body_length_series(coordinates, config.body_length_smoothing_radius)
    hand_speeds = {
        side: _speed(values, timestamps, config.speed_window_radius)
        for side, values in palms.items()
    }
    torso_speed = _speed(torso, timestamps, config.speed_window_radius)

    rows = []
    speed_timeline = []
    for move in moves:
        start = int(move["start_frame"])
        end = int(move["end_frame"])
        if start < 0 or end >= len(timestamps) or end <= start:
            raise ClimbTrackError(f"Move {move['move_id']} has invalid frame boundaries")
        hand_side = str(move["moving_hand"])
        if hand_side == "both":
            hand = (palms["left"] + palms["right"]) / 2.0
            hand_speed = (hand_speeds["left"] + hand_speeds["right"]) / 2.0
            hand_valid_mask = palm_valid["left"] & palm_valid["right"]
        elif hand_side in palms:
            hand = palms[hand_side]
            hand_speed = hand_speeds[hand_side]
            hand_valid_mask = palm_valid[hand_side]
        else:
            raise ClimbTrackError(f"Move {move['move_id']} has unknown moving hand {hand_side!r}")

        hand_fraction = float(np.mean(hand_valid_mask[start : end + 1]))
        body_fraction = float(np.mean(torso_valid[start : end + 1]))
        if min(hand_fraction, body_fraction) < config.minimum_valid_fraction:
            raise ClimbTrackError(
                f"Move {move['move_id']} has too few valid pose observations "
                f"(hand={hand_fraction:.1%}, body={body_fraction:.1%})"
            )

        hand_metrics = _trajectory_metrics(hand, hand_speed, timestamps, body_lengths, start, end)
        body_metrics = _trajectory_metrics(torso, torso_speed, timestamps, body_lengths, start, end)
        support_displacement: float | None = None
        support_path: float | None = None
        if hand_side in ("left", "right"):
            support_side = "right" if hand_side == "left" else "left"
            relative = palms[support_side] - torso
            relative_speed = _speed(relative, timestamps, config.speed_window_radius)
            support = _trajectory_metrics(
                relative, relative_speed, timestamps, body_lengths, start, end
            )
            support_displacement = support["direct_displacement_px"]
            support_path = support["path_length_px"]

        row = {
            "move_id": int(move["move_id"]),
            "moving_hand": hand_side,
            "outcome": str(move.get("outcome", "completed")),
            "start_timestamp": float(timestamps[start]),
            "end_timestamp": float(timestamps[end]),
            "duration_seconds": float(timestamps[end] - timestamps[start]),
            "body_length_px": float(np.median(body_lengths[start : end + 1])),
            "hand_valid_fraction": hand_fraction,
            **_prefixed_metrics("hand", hand_metrics, float(timestamps[start])),
            "body_valid_fraction": body_fraction,
            **_prefixed_metrics("body", body_metrics, float(timestamps[start])),
            "support_hand_relative_displacement_px": support_displacement,
            "support_hand_relative_path_length_px": support_path,
        }
        rows.append(row)
        speed_timeline.extend(
            {
                "move_id": int(move["move_id"]),
                "frame_idx": frame,
                "timestamp": float(timestamps[frame]),
                "offset_seconds": float(timestamps[frame] - timestamps[start]),
                "hand_speed_px_s": float(hand_speed[frame]),
                "hand_speed_body_lengths_s": float(hand_speed[frame] / body_lengths[frame]),
                "body_speed_px_s": float(torso_speed[frame]),
                "body_speed_body_lengths_s": float(torso_speed[frame] / body_lengths[frame]),
            }
            for frame in range(start, end + 1)
        )

    return MoveMetricsResult(
        metrics=rows,
        speed_timeline=speed_timeline,
        diagnostics={
            "moves": len(rows),
            "speed_samples": len(speed_timeline),
            "frames": len(timestamps),
            "body_length_px_median": float(np.median(body_lengths)),
            "body_length_px_p10": float(np.quantile(body_lengths, 0.10)),
            "body_length_px_p90": float(np.quantile(body_lengths, 0.90)),
            "position_smoothing_radius": config.position_smoothing_radius,
            "speed_window_radius": config.speed_window_radius,
            "body_length_smoothing_radius": config.body_length_smoothing_radius,
            "timing": "source_frame_timestamps_vfr",
        },
    )


def _timeline(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_count = max(int(record["frame_idx"]) for record in records) + 1
    timestamps = np.full(frame_count, np.nan, dtype=np.float64)
    wanted = set(BODY_POINTS)
    for anchors in HAND_ANCHORS.values():
        wanted.update(anchors)
    coordinates = {name: np.full((frame_count, 2), np.nan) for name in wanted}
    for record in records:
        frame = int(record["frame_idx"])
        timestamp = float(record["timestamp"])
        if math.isfinite(timestamps[frame]) and not math.isclose(
            timestamps[frame], timestamp, abs_tol=1e-6
        ):
            raise ClimbTrackError(f"Pose frame {frame} contains inconsistent timestamps")
        timestamps[frame] = timestamp
        name = str(record["keypoint_name"])
        if name in coordinates and not bool(record["is_missing"]):
            coordinates[name][frame] = (float(record["x"]), float(record["y"]))
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ClimbTrackError("Move metrics require a complete forward timeline")
    return timestamps, coordinates


def _robust_center(
    coordinates: dict[str, np.ndarray],
    names: tuple[str, ...],
    *,
    minimum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([coordinates[name] for name in names], axis=1)
    counts = np.sum(np.isfinite(stack[:, :, 0]), axis=1)
    valid = counts >= minimum_points
    center = np.full((len(stack), 2), np.nan)
    if valid.any():
        center[valid] = np.nanmedian(stack[valid], axis=1)
    return center, valid


def _interpolate(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    indices = np.arange(len(result))
    for dimension in range(2):
        valid = np.isfinite(result[:, dimension])
        if not valid.any():
            raise ClimbTrackError("A required metric trajectory is entirely missing")
        result[:, dimension] = np.interp(indices, indices[valid], result[valid, dimension])
    return result


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        result[index] = np.median(values[start:end], axis=0)
    return result


def _speed(
    values: np.ndarray,
    timestamps: np.ndarray,
    radius: int,
) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values) - 1, index + radius)
        seconds = timestamps[end] - timestamps[start]
        if seconds > 0:
            result[index] = float(np.linalg.norm(values[end] - values[start])) / seconds
    return result


def _body_length_series(coordinates: dict[str, np.ndarray], radius: int) -> np.ndarray:
    """Estimate the apparent body length per frame in image pixels.

    A single median over the whole video misnormalizes every move recorded at a
    different distance from the camera: on the reference video the apparent body
    length spans 293 to 445 px, so one global value biases individual moves by
    well over ten percent. The chain is summed segment by segment, which keeps it
    stable when a knee bends in the image plane, and is smoothed over a wide
    window because depth changes slowly while single-frame foreshortening is noisy.
    """
    points = {name: _interpolate(coordinates[name]) for name in BODY_POINTS}
    shoulders = (points["left_shoulder"] + points["right_shoulder"]) / 2.0
    hips = (points["left_hip"] + points["right_hip"]) / 2.0
    torso = np.linalg.norm(shoulders - hips, axis=1)
    legs = []
    for side in ("left", "right"):
        thigh = np.linalg.norm(points[f"{side}_hip"] - points[f"{side}_knee"], axis=1)
        shin = np.linalg.norm(points[f"{side}_knee"] - points[f"{side}_ankle"], axis=1)
        legs.append(thigh + shin)
    series = torso + np.mean(np.stack(legs, axis=1), axis=1)
    smoothed = np.empty_like(series)
    for index in range(len(series)):
        start = max(0, index - radius)
        end = min(len(series), index + radius + 1)
        smoothed[index] = np.median(series[start:end])
    if not np.isfinite(smoothed).all() or smoothed.min() < 1.0:
        raise ClimbTrackError("Could not estimate body length for normalized move metrics")
    return smoothed


def _trajectory_metrics(
    values: np.ndarray,
    speed: np.ndarray,
    timestamps: np.ndarray,
    body_lengths: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    segment = values[start : end + 1]
    deltas = np.diff(segment, axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    displacement = segment[-1] - segment[0]
    duration = float(timestamps[end] - timestamps[start])
    segment_speed = speed[start : end + 1]
    segment_body = body_lengths[start : end + 1]
    # Every path column now uses the same definition, the summed frame-to-frame
    # steps. Integrating the smoothed speed instead disagreed with the axis-wise
    # path sums printed beside it by up to a quarter on the reference video.
    path_length = float(np.sum(step_lengths))
    normalized_speed = segment_speed / segment_body
    normalized_path = float(np.sum(step_lengths / segment_body[1:]))
    # Body length is smoothed over about a second, so both peaks fall on the same
    # frame in practice; the normalized series decides because BL/s is the unit
    # the results are read in.
    peak_offset = int(np.argmax(normalized_speed))
    return {
        "horizontal_displacement_px": float(displacement[0]),
        "vertical_gain_px": float(-displacement[1]),
        "horizontal_path_px": float(np.sum(np.abs(deltas[:, 0]))),
        "vertical_path_px": float(np.sum(np.abs(deltas[:, 1]))),
        "direct_displacement_px": float(np.linalg.norm(displacement)),
        "path_length_px": path_length,
        "mean_speed_px_s": path_length / duration,
        "max_speed_px_s": float(np.max(segment_speed)),
        "mean_speed_body_lengths_s": normalized_path / duration,
        "max_speed_body_lengths_s": float(normalized_speed[peak_offset]),
        "peak_speed_timestamp": float(timestamps[start + peak_offset]),
    }


def _prefixed_metrics(
    prefix: str,
    metrics: dict[str, float],
    start_timestamp: float,
) -> dict[str, float]:
    result = {f"{prefix}_{name}": value for name, value in metrics.items()}
    result[f"{prefix}_peak_speed_offset_seconds"] = (
        metrics["peak_speed_timestamp"] - start_timestamp
    )
    return result
