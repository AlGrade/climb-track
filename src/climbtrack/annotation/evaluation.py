"""Simple spatial metrics for the deliberately small reviewed frame set."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from climbtrack.annotation.session import load_session
from climbtrack.errors import ClimbTrackError

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


def _summarize(samples: list[dict[str, float | bool]]) -> dict[str, float | int]:
    errors = [float(sample["error_px"]) for sample in samples]
    normalized = [float(sample["normalized_error"]) for sample in samples]
    return {
        "keypoints": len(samples),
        "mean_error_px": statistics.fmean(errors),
        "median_error_px": statistics.median(errors),
        "p95_error_px": _percentile(errors, 0.95),
        "mean_normalized_error": statistics.fmean(normalized),
        "pck": statistics.fmean(bool(sample["pck"]) for sample in samples),
        "oks": statistics.fmean(float(sample["oks"]) for sample in samples),
        "low_confidence_rate": statistics.fmean(
            bool(sample["low_confidence"]) for sample in samples
        ),
        "corrected_rate": statistics.fmean(bool(sample["corrected"]) for sample in samples),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
