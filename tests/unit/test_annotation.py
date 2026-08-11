import json
from itertools import pairwise
from pathlib import Path

import pytest

from climbtrack.annotation.evaluation import evaluate_session
from climbtrack.annotation.keypoints import ANNOTATED_KEYPOINTS
from climbtrack.annotation.selection import select_annotation_frames
from climbtrack.annotation.session import (
    AnnotationCrop,
    AnnotationFrame,
    AnnotationPoint,
    AnnotationSession,
    save_session,
)


def test_annotation_subset_has_forty_movement_keypoints() -> None:
    assert len(ANNOTATED_KEYPOINTS) == 40
    assert len(set(ANNOTATED_KEYPOINTS)) == 40
    assert "left_hip" in ANNOTATED_KEYPOINTS
    assert "right_pinky_finger4" in ANNOTATED_KEYPOINTS


def test_difficult_frame_sampling_is_deterministic_and_spaced() -> None:
    frames = [{"frame_idx": index, "timestamp": index * 0.1} for index in range(20)]
    rows = []
    for index in range(20):
        confidence = 0.05 if index == 18 else 0.9
        for offset, name in enumerate(("left_hip", "right_hip")):
            rows.append(
                {
                    "frame_idx": index,
                    "keypoint_name": name,
                    "confidence": confidence,
                    "x": index + offset,
                    "y": index,
                }
            )

    selected = select_annotation_frames(
        frames,
        rows,
        {"left_hip", "right_hip"},
        count=5,
        minimum_spacing_seconds=0.15,
    )

    assert len(selected) == 5
    assert 18 in selected
    assert all((right - left) * 0.1 >= 0.15 for left, right in pairwise(selected))


def test_evaluation_compares_predictions_with_reviewed_points(tmp_path: Path) -> None:
    points = {
        "left_hip": AnnotationPoint(
            name="left_hip",
            group="body",
            predicted_x=10,
            predicted_y=10,
            predicted_confidence=0.9,
            x=20,
            y=10,
        ),
        "right_hip": AnnotationPoint(
            name="right_hip",
            group="body",
            predicted_x=30,
            predicted_y=10,
            predicted_confidence=0.1,
            x=30,
            y=10,
        ),
    }
    session = AnnotationSession(
        created_at="2026-08-11T00:00:00+00:00",
        source_video_name="video.mp4",
        ingest_cache_key="ingest",
        selection_cache_key="selection",
        pose_cache_key="pose",
        keypoint_names=list(points),
        frames=[
            AnnotationFrame(
                frame_idx=1,
                timestamp=0.1,
                image_path="frames/000000001.png",
                crop=AnnotationCrop(x1=0, y1=0, x2=100, y2=80),
                reviewed=True,
                points=points,
            )
        ],
    )
    session_path = tmp_path / "ground_truth.json"
    save_session(session, session_path)

    output, metrics = evaluate_session(
        session_path,
        pck_threshold=0.2,
        oks_sigma=0.1,
        confidence_threshold=0.15,
    )

    overall = metrics["groups"]["overall"]
    assert overall["mean_error_px"] == pytest.approx(5.0)
    assert overall["pck"] == 1.0
    assert overall["corrected_rate"] == 0.5
    assert overall["low_confidence_rate"] == 0.5
    assert json.loads(output.read_text(encoding="utf-8"))["reviewed_frames"] == 1
