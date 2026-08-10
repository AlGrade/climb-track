"""Multi-signal climber candidate scoring."""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise
from statistics import fmean
from typing import Any

from climbtrack.config import SelectionConfig


@dataclass(frozen=True)
class SelectionCandidate:
    """Explainable score for one person track."""

    track_id: int
    score: float
    observations: int
    first_frame: int
    last_frame: int
    span_frames: int
    continuity: float
    vertical_range: float
    motion: float
    center: float
    image_area: float
    eligible: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible candidate record."""
        return asdict(self)


def rank_candidates(
    tracks: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    config: SelectionConfig,
) -> list[SelectionCandidate]:
    """Combine length, continuity, motion, vertical, center and area signals."""
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        by_track[int(row["track_id"])].append(row)
    if not by_track:
        return []

    diagonal = math.hypot(image_width, image_height)
    image_area = float(image_width * image_height)
    raw: list[dict[str, Any]] = []
    for track_id, rows in by_track.items():
        ordered = sorted(rows, key=lambda row: int(row["frame_idx"]))
        frames = [int(row["frame_idx"]) for row in ordered]
        centers = [
            (
                (float(row["x1"]) + float(row["x2"])) / 2.0,
                (float(row["y1"]) + float(row["y2"])) / 2.0,
            )
            for row in ordered
        ]
        areas = [
            (float(row["x2"]) - float(row["x1"]))
            * (float(row["y2"]) - float(row["y1"]))
            / image_area
            for row in ordered
        ]
        span = frames[-1] - frames[0] + 1
        continuity = len(set(frames)) / span
        vertical_range = (
            max(center[1] for center in centers) - min(center[1] for center in centers)
        ) / image_height
        total_motion = sum(math.dist(left, right) for left, right in pairwise(centers))
        motion = min(total_motion / (2.0 * diagonal), 1.0)
        center_score = fmean(
            max(
                0.0,
                1.0
                - math.hypot(
                    (center[0] - image_width / 2.0) / (image_width / 2.0),
                    (center[1] - image_height / 2.0) / (image_height / 2.0),
                )
                / math.sqrt(2.0),
            )
            for center in centers
        )
        raw.append(
            {
                "track_id": track_id,
                "observations": len(set(frames)),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "span_frames": span,
                "continuity": continuity,
                "vertical_range": vertical_range,
                "motion": motion,
                "center": center_score,
                "image_area": fmean(areas),
            }
        )

    maxima = {
        name: max(float(candidate[name]) for candidate in raw)
        for name in ("observations", "vertical_range", "motion", "image_area")
    }
    weights = config.weights.model_dump()
    weight_sum = sum(weights.values())
    candidates: list[SelectionCandidate] = []
    for candidate in raw:
        normalized = {
            "length": _normalize(float(candidate["observations"]), maxima["observations"]),
            "continuity": float(candidate["continuity"]),
            "vertical_range": _normalize(
                float(candidate["vertical_range"]), maxima["vertical_range"]
            ),
            "motion": _normalize(float(candidate["motion"]), maxima["motion"]),
            "center": float(candidate["center"]),
            "image_area": _normalize(float(candidate["image_area"]), maxima["image_area"]),
        }
        score = sum(weights[name] * normalized[name] for name in weights) / weight_sum
        eligible = (
            int(candidate["observations"]) >= config.minimum_observations
            and float(candidate["continuity"]) >= config.minimum_continuity
        )
        candidates.append(
            SelectionCandidate(
                track_id=int(candidate["track_id"]),
                score=score,
                observations=int(candidate["observations"]),
                first_frame=int(candidate["first_frame"]),
                last_frame=int(candidate["last_frame"]),
                span_frames=int(candidate["span_frames"]),
                continuity=float(candidate["continuity"]),
                vertical_range=float(candidate["vertical_range"]),
                motion=float(candidate["motion"]),
                center=float(candidate["center"]),
                image_area=float(candidate["image_area"]),
                eligible=eligible,
            )
        )
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.track_id))


def _normalize(value: float, maximum: float) -> float:
    return value / maximum if maximum > 0 else 0.0
