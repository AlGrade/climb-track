"""Automatic hand-move segmentation from refined 2D pose timelines."""

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from climbtrack.config import MoveDetectionConfig
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
BODY_SCALE_POINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
BODY_MOTION_POINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
)


@dataclass(frozen=True)
class MoveDetectionResult:
    """Canonical move rows plus scale and candidate diagnostics."""

    moves: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def detect_hand_moves(
    records: list[dict[str, Any]],
    config: MoveDetectionConfig,
) -> MoveDetectionResult:
    """Detect stable-hand to stable-hand transitions for each anatomical side."""
    if not records:
        raise ClimbTrackError("Automatic move detection requires refined pose observations")
    frame_count = max(int(record["frame_idx"]) for record in records) + 1
    timestamps = np.full(frame_count, np.nan, dtype=np.float64)
    wanted = set(BODY_MOTION_POINTS)
    for names in HAND_ANCHORS.values():
        wanted.update(names)
    coordinates = {name: np.full((frame_count, 2), np.nan, dtype=np.float64) for name in wanted}
    for record in records:
        frame_idx = int(record["frame_idx"])
        timestamp = float(record["timestamp"])
        previous = timestamps[frame_idx]
        if math.isfinite(previous) and not math.isclose(previous, timestamp, abs_tol=1e-6):
            raise ClimbTrackError(f"Pose frame {frame_idx} contains inconsistent timestamps")
        timestamps[frame_idx] = timestamp
        name = str(record["keypoint_name"])
        if name in coordinates and not bool(record["is_missing"]):
            coordinates[name][frame_idx] = (float(record["x"]), float(record["y"]))
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ClimbTrackError("Automatic move detection requires a complete forward timeline")

    body_scale = _body_scale(coordinates)
    detected: list[dict[str, Any]] = []
    side_diagnostics: dict[str, Any] = {}
    stable_runs_by_side: dict[str, list[tuple[int, int]]] = {}
    for side, anchors in HAND_ANCHORS.items():
        palm = _palm_center(coordinates, anchors)
        palm = _median_smooth(palm, config.position_smoothing_radius)
        speed = _normalized_speed(
            palm,
            timestamps,
            body_scale,
            config.speed_window_radius,
        )
        stable_runs = _stable_runs(
            speed,
            timestamps,
            threshold=config.stable_speed_body_lengths_per_second,
            minimum_seconds=config.minimum_stable_seconds,
        )
        stable_runs = _merge_same_hold_runs(stable_runs, palm, timestamps, body_scale, config)
        stable_runs_by_side[side] = stable_runs
        candidates = _transitions(
            stable_runs,
            palm,
            speed,
            timestamps,
            body_scale,
            side,
            config,
        )
        detected.extend(candidates)
        side_diagnostics[side] = {
            "stable_runs": len(stable_runs),
            "candidates": len(candidates),
            "median_speed_body_lengths_per_second": float(np.median(speed)),
            "maximum_speed_body_lengths_per_second": float(np.max(speed)),
        }

    body_positions = {
        name: _median_smooth(
            _interpolate_coordinates(coordinates[name]),
            config.position_smoothing_radius,
        )
        for name in BODY_MOTION_POINTS
    }
    body_motion = _body_motion(body_positions, timestamps, body_scale, config)
    body_stable_runs = _stable_runs(
        body_motion,
        timestamps,
        threshold=config.body_stable_speed_body_lengths_per_second,
        minimum_seconds=config.minimum_body_stable_seconds,
    )
    detected.sort(key=lambda move: (move["start_timestamp"], move["end_timestamp"]))
    _extend_completed_moves(detected, body_stable_runs, timestamps)
    terminal = _terminal_fall_move(
        detected,
        stable_runs_by_side,
        body_stable_runs,
        body_positions,
        timestamps,
        body_scale,
        config,
    )
    if terminal is not None:
        detected.append(terminal)
    detected.sort(key=lambda move: (move["start_timestamp"], move["end_timestamp"]))
    detected = [dict(move, move_id=index) for index, move in enumerate(detected, 1)]
    return MoveDetectionResult(
        moves=detected,
        diagnostics={
            "frames": frame_count,
            "body_scale_px": body_scale,
            "moves": len(detected),
            "completed_moves": sum(move["outcome"] == "completed" for move in detected),
            "fall_moves": sum(move["outcome"] == "fall" for move in detected),
            "body_stable_runs": len(body_stable_runs),
            "sides": side_diagnostics,
        },
    )


def _palm_center(
    coordinates: dict[str, np.ndarray],
    anchors: tuple[str, ...],
) -> np.ndarray:
    stack = np.stack([coordinates[name] for name in anchors], axis=1)
    with np.errstate(all="ignore"):
        center = np.nanmedian(stack, axis=1)
    if np.isnan(center).all(axis=1).any():
        raise ClimbTrackError("A hand is missing all palm anchors in at least one frame")
    return _interpolate_coordinates(center)


def _body_scale(coordinates: dict[str, np.ndarray]) -> float:
    points = {name: _interpolate_coordinates(coordinates[name]) for name in BODY_SCALE_POINTS}
    shoulder_center = (points["left_shoulder"] + points["right_shoulder"]) / 2.0
    hip_center = (points["left_hip"] + points["right_hip"]) / 2.0
    torso = np.linalg.norm(shoulder_center - hip_center, axis=1)
    shoulder_width = np.linalg.norm(points["left_shoulder"] - points["right_shoulder"], axis=1)
    scale = float(np.median(np.maximum(torso, shoulder_width)))
    if not math.isfinite(scale) or scale < 1.0:
        raise ClimbTrackError("Could not estimate a stable body scale for move detection")
    return scale


def _interpolate_coordinates(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    indices = np.arange(len(result))
    for dimension in range(2):
        valid = np.isfinite(result[:, dimension])
        if not valid.any():
            raise ClimbTrackError("Required move-detection keypoints are entirely missing")
        result[:, dimension] = np.interp(indices, indices[valid], result[valid, dimension])
    return result


def _median_smooth(values: np.ndarray, radius: int) -> np.ndarray:
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        result[index] = np.median(values[start:end], axis=0)
    return result


def _normalized_speed(
    values: np.ndarray,
    timestamps: np.ndarray,
    body_scale: float,
    radius: int,
) -> np.ndarray:
    speed = np.zeros(len(values), dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values) - 1, index + radius)
        seconds = timestamps[end] - timestamps[start]
        if seconds > 0:
            speed[index] = float(np.linalg.norm(values[end] - values[start])) / (
                seconds * body_scale
            )
    return speed


def _stable_runs(
    speed: np.ndarray,
    timestamps: np.ndarray,
    *,
    threshold: float,
    minimum_seconds: float,
) -> list[tuple[int, int]]:
    stable = speed < threshold
    runs = []
    index = 0
    while index < len(stable):
        if not stable[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(stable) and stable[index + 1]:
            index += 1
        end = index
        if timestamps[end] - timestamps[start] >= minimum_seconds:
            runs.append((start, end))
        index += 1
    return runs


def _merge_same_hold_runs(
    runs: list[tuple[int, int]],
    positions: np.ndarray,
    timestamps: np.ndarray,
    body_scale: float,
    config: MoveDetectionConfig,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged:
            previous_start, previous_end = merged[-1]
            gap = timestamps[start] - timestamps[previous_end]
            before = np.median(
                positions[max(previous_start, previous_end - 10) : previous_end + 1], axis=0
            )
            after = np.median(positions[start : min(end + 1, start + 11)], axis=0)
            displacement = float(np.linalg.norm(after - before)) / body_scale
            if (
                gap <= config.maximum_stable_gap_seconds
                and displacement <= config.same_hold_radius_body_lengths
            ):
                merged[-1] = (previous_start, end)
                continue
        merged.append((start, end))
    return merged


def _transitions(
    runs: list[tuple[int, int]],
    positions: np.ndarray,
    speed: np.ndarray,
    timestamps: np.ndarray,
    body_scale: float,
    side: str,
    config: MoveDetectionConfig,
) -> list[dict[str, Any]]:
    moves = []
    for previous, following in pairwise(runs):
        hand_start_frame, end_frame = previous[1], following[0]
        start_candidates = np.flatnonzero(
            speed[hand_start_frame : end_frame + 1] >= config.start_speed_body_lengths_per_second
        )
        start_frame = (
            hand_start_frame + int(start_candidates[0])
            if len(start_candidates)
            else hand_start_frame
        )
        duration = float(timestamps[end_frame] - timestamps[start_frame])
        before = np.median(
            positions[max(previous[0], previous[1] - 10) : previous[1] + 1],
            axis=0,
        )
        after = np.median(
            positions[following[0] : min(following[1] + 1, following[0] + 11)],
            axis=0,
        )
        displacement = float(np.linalg.norm(after - before)) / body_scale
        if not (
            config.minimum_move_seconds <= duration <= config.maximum_move_seconds
            and displacement >= config.minimum_displacement_body_lengths
        ):
            continue
        peak_speed = float(np.max(speed[start_frame : end_frame + 1]))
        confidence = min(
            0.90,
            0.55 + 0.08 * min(2.0, displacement) + 0.03 * min(4.0, peak_speed),
        )
        moves.append(
            {
                "move_id": 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_timestamp": float(timestamps[start_frame]),
                "end_timestamp": float(timestamps[end_frame]),
                "moving_hand": side,
                "confidence": confidence,
                "source": "automatic",
                "is_reviewed": False,
                "outcome": "completed",
            }
        )
    return moves


def _body_motion(
    positions: dict[str, np.ndarray],
    timestamps: np.ndarray,
    body_scale: float,
    config: MoveDetectionConfig,
) -> np.ndarray:
    speeds = np.stack(
        [
            _normalized_speed(
                positions[name],
                timestamps,
                body_scale,
                config.speed_window_radius,
            )
            for name in BODY_MOTION_POINTS
        ],
        axis=1,
    )
    return np.quantile(speeds, config.body_motion_quantile, axis=1)


def _extend_completed_moves(
    moves: list[dict[str, Any]],
    body_stable_runs: list[tuple[int, int]],
    timestamps: np.ndarray,
) -> None:
    for index, move in enumerate(moves):
        hand_end = int(move["end_frame"])
        next_start = int(moves[index + 1]["start_frame"]) if index + 1 < len(moves) else None
        completion: int | None = None
        for stable_start, stable_end in body_stable_runs:
            if stable_start <= hand_end <= stable_end:
                completion = hand_end
                break
            if stable_start > hand_end:
                completion = stable_start
                break
        if next_start is not None:
            completion = next_start if completion is None else min(completion, next_start)
        if completion is None:
            completion = hand_end
        move["end_frame"] = max(hand_end, completion)
        move["end_timestamp"] = float(timestamps[int(move["end_frame"])])


def _terminal_fall_move(
    completed: list[dict[str, Any]],
    hand_runs: dict[str, list[tuple[int, int]]],
    body_runs: list[tuple[int, int]],
    body_positions: dict[str, np.ndarray],
    timestamps: np.ndarray,
    body_scale: float,
    config: MoveDetectionConfig,
) -> dict[str, Any] | None:
    if not all(hand_runs.get(side) for side in ("left", "right")):
        return None
    departures = {side: runs[-1][1] for side, runs in hand_runs.items()}
    hand_start = min(departures.values())
    if completed and hand_start <= int(completed[-1]["end_frame"]):
        return None
    preceding_body_runs = [run for run in body_runs if run[1] <= hand_start]
    start_frame = preceding_body_runs[-1][1] if preceding_body_runs else hand_start
    torso_center = (
        body_positions["left_shoulder"]
        + body_positions["right_shoulder"]
        + body_positions["left_hip"]
        + body_positions["right_hip"]
    ) / 4.0
    fall_slice = torso_center[hand_start:]
    if not len(fall_slice):
        return None
    end_frame = hand_start + int(np.argmax(fall_slice[:, 1]))
    drop = float(torso_center[end_frame, 1] - torso_center[start_frame, 1]) / body_scale
    if end_frame <= hand_start or drop < config.fall_minimum_drop_body_lengths:
        return None
    ordered = sorted(departures.items(), key=lambda item: item[1])
    first_side, first_frame = ordered[0]
    _, second_frame = ordered[1]
    moving_hand = (
        "both" if timestamps[second_frame] - timestamps[first_frame] <= 0.10 else first_side
    )
    return {
        "move_id": 1,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_timestamp": float(timestamps[start_frame]),
        "end_timestamp": float(timestamps[end_frame]),
        "moving_hand": moving_hand,
        "confidence": 0.75,
        "source": "automatic",
        "is_reviewed": False,
        "outcome": "fall",
    }
