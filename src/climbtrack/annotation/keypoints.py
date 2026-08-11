"""Movement-relevant subset of the 308-point Sapiens registry."""

from typing import Any

BODY = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
FEET = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)
EXTRA = (
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
)
FINGERTIPS = tuple(
    f"{side}_{finger}4"
    for side in ("left", "right")
    for finger in ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")
)
ANNOTATED_KEYPOINTS = (*BODY, *FEET, *EXTRA, *FINGERTIPS)


def annotation_keypoints(registry: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Resolve the reviewed 40-point subset in stable registry order."""
    by_name = {entry["name"]: entry for entry in registry["keypoints"]}
    missing = set(ANNOTATED_KEYPOINTS) - by_name.keys()
    if missing:
        raise ValueError(f"Registry lacks annotation keypoints: {sorted(missing)}")
    wanted = set(ANNOTATED_KEYPOINTS)
    return tuple(entry for entry in registry["keypoints"] if entry["name"] in wanted)
