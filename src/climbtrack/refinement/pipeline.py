"""Pure temporal refinement logic over canonical long-form pose rows."""

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from climbtrack.config import RefineConfig
from climbtrack.refinement.one_euro import OneEuro2D


@dataclass(frozen=True)
class RefinementResult:
    """Refined records plus auditable repair counts."""

    records: list[dict[str, Any]]
    diagnostics: dict[str, int]


def refine_pose_records(
    records: list[dict[str, Any]],
    registry: dict[str, Any],
    crop_scales: dict[int, float],
    config: RefineConfig,
) -> RefinementResult:
    """Repair swaps/outliers, gate confidence, fill short gaps, then smooth."""
    copied = [dict(record) for record in records]
    by_frame: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in copied:
        by_frame[int(record["frame_idx"])][str(record["keypoint_name"])] = record
    frame_indices = sorted(by_frame)
    if not frame_indices:
        raise ValueError("At least one pose observation is required")

    diagnostics: Counter[str] = Counter()
    groups = {str(entry["name"]): str(entry["group"]) for entry in registry["keypoints"]}
    _repair_swaps(by_frame, frame_indices, registry, crop_scales, config, diagnostics)
    outliers = _segment_outliers(by_frame, frame_indices, registry, config)

    for frame_idx in frame_indices:
        for name, record in by_frame[frame_idx].items():
            confidence = float(record["confidence"])
            threshold = config.confidence_threshold_overrides.get(
                groups[name], config.confidence_threshold
            )
            reason = None
            if confidence < threshold:
                reason = "confidence_gated"
            elif (frame_idx, name) in outliers:
                reason = "segment_outliers"
            if reason is not None:
                _mark_missing(record)
                diagnostics[reason] += 1

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in copied:
        by_name[str(record["keypoint_name"])].append(record)
    for name, series in by_name.items():
        series.sort(key=lambda record: int(record["frame_idx"]))
        diagnostics["interpolated"] += _interpolate_short_gaps(
            series, config.maximum_interpolation_gap
        )
        if groups[name] in config.smoothing_groups:
            diagnostics["smoothed"] += _smooth_series(series, config)
    diagnostics["missing_final"] = sum(bool(record["is_missing"]) for record in copied)
    return RefinementResult(copied, dict(sorted(diagnostics.items())))


def _repair_swaps(
    by_frame: dict[int, dict[str, dict[str, Any]]],
    frame_indices: list[int],
    registry: dict[str, Any],
    crop_scales: dict[int, float],
    config: RefineConfig,
    diagnostics: Counter[str],
) -> None:
    entries = {str(entry["name"]): entry for entry in registry["keypoints"]}
    pairs = []
    for entry in registry["keypoints"]:
        partner = entry.get("swap")
        if partner is None or entry["group"] == "face":
            continue
        if int(entry["index"]) < int(entries[str(partner)]["index"]):
            pairs.append((str(entry["name"]), str(partner)))

    for previous_idx, frame_idx in pairwise(frame_indices):
        previous, current = by_frame[previous_idx], by_frame[frame_idx]
        scale = max(1.0, crop_scales.get(frame_idx, 1.0))
        for left, right in pairs:
            if not all(name in previous and name in current for name in (left, right)):
                continue
            previous_left, previous_right = previous[left], previous[right]
            current_left, current_right = current[left], current[right]
            direct = _distance(previous_left, current_left) + _distance(
                previous_right, current_right
            )
            swapped = _distance(previous_left, current_right) + _distance(
                previous_right, current_left
            )
            low_confidence = min(
                float(current_left["confidence"]), float(current_right["confidence"])
            )
            if (
                direct > config.swap_minimum_jump_scale * scale
                and swapped < direct * config.swap_cost_ratio
                and low_confidence <= config.outlier_confidence_ceiling
            ):
                for field in ("x", "y", "confidence"):
                    current_left[field], current_right[field] = (
                        current_right[field],
                        current_left[field],
                    )
                diagnostics["swap_pairs"] += 1


def _segment_outliers(
    by_frame: dict[int, dict[str, dict[str, Any]]],
    frame_indices: list[int],
    registry: dict[str, Any],
    config: RefineConfig,
) -> set[tuple[int, str]]:
    edges = [(str(left), str(right)) for left, right in registry["skeleton_edges"]]
    baselines: dict[tuple[str, str], float] = {}
    for edge in edges:
        lengths = []
        for frame_idx in frame_indices:
            first, second = by_frame[frame_idx][edge[0]], by_frame[frame_idx][edge[1]]
            if min(float(first["confidence"]), float(second["confidence"])) >= 0.5:
                lengths.append(_distance(first, second))
        if lengths:
            baselines[edge] = statistics.median(lengths)

    votes: Counter[tuple[int, str]] = Counter()
    index_by_frame = {frame_idx: index for index, frame_idx in enumerate(frame_indices)}
    for frame_idx in frame_indices:
        for edge, baseline in baselines.items():
            first, second = by_frame[frame_idx][edge[0]], by_frame[frame_idx][edge[1]]
            if baseline <= 0 or _distance(first, second) <= baseline * config.segment_maximum_ratio:
                continue
            first_confidence = float(first["confidence"])
            second_confidence = float(second["confidence"])
            if min(first_confidence, second_confidence) > config.outlier_confidence_ceiling:
                continue
            first_deviation = _temporal_deviation(
                by_frame, frame_indices, index_by_frame[frame_idx], edge[0]
            )
            second_deviation = _temporal_deviation(
                by_frame, frame_indices, index_by_frame[frame_idx], edge[1]
            )
            if not math.isclose(first_deviation, second_deviation, abs_tol=1e-6):
                suspect = edge[0] if first_deviation > second_deviation else edge[1]
            else:
                suspect = edge[0] if first_confidence <= second_confidence else edge[1]
            if (
                float(by_frame[frame_idx][suspect]["confidence"])
                > config.outlier_confidence_ceiling
            ):
                continue
            votes[(frame_idx, suspect)] += 1
    return set(votes)


def _temporal_deviation(
    by_frame: dict[int, dict[str, dict[str, Any]]],
    frame_indices: list[int],
    index: int,
    name: str,
) -> float:
    if index == 0 or index == len(frame_indices) - 1:
        return 0.0
    previous = by_frame[frame_indices[index - 1]][name]
    current = by_frame[frame_indices[index]][name]
    following = by_frame[frame_indices[index + 1]][name]
    start_time = float(previous["timestamp"])
    end_time = float(following["timestamp"])
    if end_time <= start_time:
        return 0.0
    fraction = (float(current["timestamp"]) - start_time) / (end_time - start_time)
    expected_x = float(previous["x"]) + fraction * (float(following["x"]) - float(previous["x"]))
    expected_y = float(previous["y"]) + fraction * (float(following["y"]) - float(previous["y"]))
    return math.hypot(float(current["x"]) - expected_x, float(current["y"]) - expected_y)


def _interpolate_short_gaps(series: list[dict[str, Any]], maximum_gap: int) -> int:
    interpolated = 0
    index = 0
    while index < len(series):
        if not bool(series[index]["is_missing"]):
            index += 1
            continue
        start = index
        while index < len(series) and bool(series[index]["is_missing"]):
            index += 1
        end = index
        gap = end - start
        if gap > maximum_gap or start == 0 or end == len(series):
            continue
        before, after = series[start - 1], series[end]
        start_time, end_time = float(before["timestamp"]), float(after["timestamp"])
        if end_time <= start_time:
            continue
        confidence = min(float(before["confidence"]), float(after["confidence"]))
        for record in series[start:end]:
            fraction = (float(record["timestamp"]) - start_time) / (end_time - start_time)
            record["x"] = float(before["x"]) + fraction * (float(after["x"]) - float(before["x"]))
            record["y"] = float(before["y"]) + fraction * (float(after["y"]) - float(before["y"]))
            record["confidence"] = confidence
            record["is_missing"] = False
            record["is_interpolated"] = True
            interpolated += 1
    return interpolated


def _smooth_series(series: list[dict[str, Any]], config: RefineConfig) -> int:
    filtered = 0
    one_euro: OneEuro2D | None = None
    for record in series:
        if bool(record["is_missing"]):
            one_euro = None
            continue
        if one_euro is None:
            one_euro = OneEuro2D(
                min_cutoff=config.one_euro_min_cutoff,
                beta=config.one_euro_beta,
                derivative_cutoff=config.one_euro_derivative_cutoff,
            )
        record["x"], record["y"] = one_euro.update(
            float(record["timestamp"]), float(record["x"]), float(record["y"])
        )
        filtered += 1
    return filtered


def _mark_missing(record: dict[str, Any]) -> None:
    record["x"] = None
    record["y"] = None
    record["confidence"] = None
    record["is_missing"] = True
    record["is_interpolated"] = False


def _distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    return math.hypot(
        float(first["x"]) - float(second["x"]),
        float(first["y"]) - float(second["y"]),
    )
