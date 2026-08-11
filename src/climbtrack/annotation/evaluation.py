"""Simple spatial metrics for the deliberately small reviewed frame set."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from climbtrack.annotation.session import AnnotationSession, load_session
from climbtrack.errors import ClimbTrackError
from climbtrack.schema.pose import read_pose_parquet

EVALUATION_SCHEMA_VERSION = "1.0.0"


def evaluate_session(
    session_path: Path,
    *,
    pck_threshold: float,
    oks_sigma: float,
    confidence_threshold: float,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate raw predictions against reviewed corrections and write JSON metrics."""
    session = load_session(session_path)
    reviewed = [frame for frame in session.frames if frame.reviewed]
    if not reviewed:
        raise ClimbTrackError("No annotation frame has been reviewed yet")

    samples: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for frame in reviewed:
        scale = max(frame.crop.x2 - frame.crop.x1, frame.crop.y2 - frame.crop.y1)
        for point in frame.points.values():
            if not point.visible or point.x is None or point.y is None:
                continue
            error = math.hypot(point.predicted_x - point.x, point.predicted_y - point.y)
            normalized = error / scale
            sample = {
                "error_px": error,
                "normalized_error": normalized,
                "pck": normalized <= pck_threshold,
                "oks": math.exp(-(normalized**2) / (2.0 * oks_sigma**2)),
                "low_confidence": point.predicted_confidence < confidence_threshold,
                "corrected": error > 0.5,
            }
            samples["overall"].append(sample)
            samples[point.group].append(sample)

    groups = {name: _summarize(values) for name, values in sorted(samples.items())}
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "annotation_schema_version": session.schema_version,
        "reviewed_frames": len(reviewed),
        "total_frames": len(session.frames),
        "pck_threshold": pck_threshold,
        "oks_sigma": oks_sigma,
        "normalization": "maximum annotation-crop dimension",
        "groups": groups,
    }
    output = session_path.with_name("evaluation.json")
    temporary = output.with_name(f".{output.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output, payload


def compare_pose_session(
    session_path: Path,
    refined_pose_path: Path,
    *,
    pck_threshold: float,
    oks_sigma: float,
    confidence_threshold: float,
) -> tuple[Path, dict[str, Any]]:
    """Compare cached raw and refined observations against the same reviewed ground truth."""
    session = load_session(session_path)
    reviewed = [frame for frame in session.frames if frame.reviewed]
    if not reviewed:
        raise ClimbTrackError("No annotation frame has been reviewed yet")

    raw_predictions = {
        (frame.frame_idx, point.name): {
            "x": point.predicted_x,
            "y": point.predicted_y,
            "confidence": point.predicted_confidence,
            "is_missing": False,
        }
        for frame in reviewed
        for point in frame.points.values()
    }
    wanted_frames = {frame.frame_idx for frame in reviewed}
    wanted_names = set(session.keypoint_names)
    refined_predictions = {
        (int(row["frame_idx"]), str(row["keypoint_name"])): row
        for row in read_pose_parquet(refined_pose_path)
        if int(row["frame_idx"]) in wanted_frames and str(row["keypoint_name"]) in wanted_names
    }
    raw = _evaluate_predictions(
        session,
        raw_predictions,
        pck_threshold=pck_threshold,
        oks_sigma=oks_sigma,
        confidence_threshold=confidence_threshold,
    )
    refined = _evaluate_predictions(
        session,
        refined_predictions,
        pck_threshold=pck_threshold,
        oks_sigma=oks_sigma,
        confidence_threshold=confidence_threshold,
    )
    raw_overall, refined_overall = raw["overall"], refined["overall"]
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "reviewed_frames": len(reviewed),
        "pck_threshold": pck_threshold,
        "oks_sigma": oks_sigma,
        "normalization": "maximum annotation-crop dimension",
        "raw": raw,
        "refined": refined,
        "delta_refined_minus_raw": {
            "mean_error_px": refined_overall["mean_error_px"] - raw_overall["mean_error_px"],
            "pck": refined_overall["pck"] - raw_overall["pck"],
            "oks": refined_overall["oks"] - raw_overall["oks"],
            "prediction_missing_rate": (
                refined_overall["prediction_missing_rate"] - raw_overall["prediction_missing_rate"]
            ),
        },
    }
    output = session_path.with_name("evaluation_refined.json")
    _write_json(output, payload)
    return output, payload


def _evaluate_predictions(
    session: AnnotationSession,
    predictions: dict[tuple[int, str], dict[str, Any]],
    *,
    pck_threshold: float,
    oks_sigma: float,
    confidence_threshold: float,
) -> dict[str, dict[str, float | int]]:
    samples: dict[str, list[dict[str, float | bool | None]]] = defaultdict(list)
    for frame in session.frames:
        if not frame.reviewed:
            continue
        scale = max(frame.crop.x2 - frame.crop.x1, frame.crop.y2 - frame.crop.y1)
        for truth in frame.points.values():
            if not truth.visible or truth.x is None or truth.y is None:
                continue
            prediction = predictions.get((frame.frame_idx, truth.name))
            missing = prediction is None or bool(prediction["is_missing"])
            if missing:
                sample = {
                    "error_px": None,
                    "normalized_error": None,
                    "pck": False,
                    "oks": 0.0,
                    "low_confidence": True,
                    "corrected": True,
                    "prediction_missing": True,
                }
            else:
                error = math.hypot(
                    float(prediction["x"]) - truth.x, float(prediction["y"]) - truth.y
                )
                normalized = error / scale
                sample = {
                    "error_px": error,
                    "normalized_error": normalized,
                    "pck": normalized <= pck_threshold,
                    "oks": math.exp(-(normalized**2) / (2.0 * oks_sigma**2)),
                    "low_confidence": float(prediction["confidence"]) < confidence_threshold,
                    "corrected": error > 0.5,
                    "prediction_missing": False,
                }
            samples["overall"].append(sample)
            samples[truth.group].append(sample)
    return {name: _summarize(values) for name, values in sorted(samples.items())}


def _summarize(samples: list[dict[str, float | bool | None]]) -> dict[str, float | int]:
    errors = [float(sample["error_px"]) for sample in samples if sample["error_px"] is not None]
    normalized = [
        float(sample["normalized_error"])
        for sample in samples
        if sample["normalized_error"] is not None
    ]
    return {
        "keypoints": len(samples),
        "observed_predictions": len(errors),
        "mean_error_px": statistics.fmean(errors) if errors else 0.0,
        "median_error_px": statistics.median(errors) if errors else 0.0,
        "p95_error_px": _percentile(errors, 0.95) if errors else 0.0,
        "mean_normalized_error": statistics.fmean(normalized) if normalized else 0.0,
        "pck": statistics.fmean(bool(sample["pck"]) for sample in samples),
        "oks": statistics.fmean(float(sample["oks"]) for sample in samples),
        "low_confidence_rate": statistics.fmean(
            bool(sample["low_confidence"]) for sample in samples
        ),
        "corrected_rate": statistics.fmean(bool(sample["corrected"]) for sample in samples),
        "prediction_missing_rate": statistics.fmean(
            bool(sample.get("prediction_missing", False)) for sample in samples
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
