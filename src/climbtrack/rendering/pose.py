"""Shared confidence-aware pose drawing for raw and refined videos."""

from typing import Any

from climbtrack.config import AppConfig


def draw_pose(
    image: Any,
    frame_idx: int,
    rows: dict[str, dict[str, Any]],
    registry: dict[str, Any],
    crop: dict[str, Any] | None,
    config: AppConfig,
    *,
    label: str,
) -> None:
    """Draw available observations, omitting explicit missing values."""
    import cv2

    threshold = config.pose_render.confidence_threshold
    groups = {entry["name"]: entry["group"] for entry in registry["keypoints"]}
    primitives: list[tuple[str, tuple[Any, ...], float]] = []
    for left, right in registry["skeleton_edges"]:
        left_row, right_row = rows[left], rows[right]
        if not _present(left_row, threshold) or not _present(right_row, threshold):
            continue
        confidence = min(float(left_row["confidence"]), float(right_row["confidence"]))
        primitives.append(
            (
                "line",
                (
                    (round(float(left_row["x"])), round(float(left_row["y"]))),
                    (round(float(right_row["x"])), round(float(right_row["y"]))),
                ),
                confidence,
            )
        )
    for name, row in rows.items():
        if groups[name] == "face" and not config.pose_render.show_face_keypoints:
            continue
        if _present(row, threshold):
            primitives.append(
                (
                    "point",
                    ((round(float(row["x"])), round(float(row["y"]))),),
                    float(row["confidence"]),
                )
            )

    for bucket in range(5):
        selected = [
            item for item in primitives if min(4, int(min(1.0, max(0.0, item[2])) * 5)) == bucket
        ]
        if not selected:
            continue
        confidence = (bucket + 0.5) / 5
        color = (0, round(255 * confidence), round(255 * (1.0 - confidence)))
        layer = image.copy()
        for kind, points, _ in selected:
            if kind == "line":
                cv2.line(
                    layer,
                    points[0],
                    points[1],
                    color,
                    config.pose_render.line_thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    layer,
                    points[0],
                    config.pose_render.point_radius,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
        alpha = 0.25 + 0.75 * confidence
        cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0.0, image)

    if crop is not None and config.pose_render.show_pose_crop:
        x1, y1, x2, y2 = (round(float(crop[name])) for name in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    track_id = next(iter(rows.values()))["track_id"]
    cv2.putText(
        image,
        f"{label}  ID {track_id}  frame {frame_idx}",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.render.font_scale,
        (255, 255, 255),
        max(2, config.pose_render.line_thickness // 2),
        cv2.LINE_AA,
    )


def _present(row: dict[str, Any], threshold: float) -> bool:
    return (
        not bool(row["is_missing"])
        and row["x"] is not None
        and row["y"] is not None
        and row["confidence"] is not None
        and float(row["confidence"]) >= threshold
    )
