from climbtrack.config import MoveDetectionConfig
from climbtrack.moves import detect_hand_moves


def _record(frame_idx: int, timestamp: float, name: str, x: float, y: float) -> dict[str, object]:
    return {
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "track_id": 1,
        "keypoint_name": name,
        "x": x,
        "y": y,
        "confidence": 0.95,
        "is_missing": False,
        "is_interpolated": False,
        "source_backend": "test",
    }


def test_detects_stable_to_stable_hand_move_but_not_unfinished_fall() -> None:
    records = []
    hand_suffixes = (
        "wrist",
        "thumb_third_joint",
        "forefinger_third_joint",
        "middle_finger_third_joint",
        "ring_finger_third_joint",
        "pinky_finger_third_joint",
    )
    for frame_idx in range(101):
        timestamp = frame_idx * 0.1
        if frame_idx <= 30:
            left_x = 0.0
        elif frame_idx < 40:
            left_x = (frame_idx - 30) * 10.0
        else:
            left_x = 100.0
        right_x = 0.0 if frame_idx <= 80 else (frame_idx - 80) * 8.0
        for side, x in (("left", left_x), ("right", right_x)):
            for suffix in hand_suffixes:
                records.append(_record(frame_idx, timestamp, f"{side}_{suffix}", x, 0.0))
        for name, x, y in (
            ("left_shoulder", -50.0, 0.0),
            ("right_shoulder", 50.0, 0.0),
            ("left_hip", -40.0, 100.0),
            ("right_hip", 40.0, 100.0),
            ("left_knee", -35.0, 160.0),
            ("right_knee", 35.0, 160.0),
            ("left_ankle", -30.0, 220.0),
            ("right_ankle", 30.0, 220.0),
            ("left_big_toe", -25.0, 230.0),
            ("right_big_toe", 25.0, 230.0),
        ):
            records.append(_record(frame_idx, timestamp, name, x, y))

    result = detect_hand_moves(records, MoveDetectionConfig())

    assert len(result.moves) == 1
    move = result.moves[0]
    assert move["moving_hand"] == "left"
    assert 2.0 <= move["start_timestamp"] <= 3.5
    assert 3.5 <= move["end_timestamp"] <= 5.0
    assert move["source"] == "automatic"
    assert move["is_reviewed"] is False
    assert move["outcome"] == "completed"
    assert result.diagnostics["body_scale_px"] == 100.0


def test_detects_terminal_failed_move_until_lowest_fall_position() -> None:
    records = []
    hand_suffixes = (
        "wrist",
        "thumb_third_joint",
        "forefinger_third_joint",
        "middle_finger_third_joint",
        "ring_finger_third_joint",
        "pinky_finger_third_joint",
    )
    body_points = (
        ("left_shoulder", -50.0, 0.0),
        ("right_shoulder", 50.0, 0.0),
        ("left_hip", -40.0, 100.0),
        ("right_hip", 40.0, 100.0),
        ("left_knee", -35.0, 160.0),
        ("right_knee", 35.0, 160.0),
        ("left_ankle", -30.0, 220.0),
        ("right_ankle", 30.0, 220.0),
        ("left_big_toe", -25.0, 230.0),
        ("right_big_toe", 25.0, 230.0),
    )
    for frame_idx in range(101):
        timestamp = frame_idx * 0.1
        right_x = 0.0 if frame_idx <= 75 else (frame_idx - 75) * 8.0
        for side, x in (("left", 0.0), ("right", right_x)):
            for suffix in hand_suffixes:
                records.append(_record(frame_idx, timestamp, f"{side}_{suffix}", x, 0.0))
        fall_offset = max(0, frame_idx - 75) * 4.0
        for name, x, y in body_points:
            records.append(_record(frame_idx, timestamp, name, x, y + fall_offset))

    result = detect_hand_moves(records, MoveDetectionConfig())

    assert len(result.moves) == 1
    move = result.moves[0]
    assert move["moving_hand"] == "right"
    assert move["outcome"] == "fall"
    assert move["end_frame"] == 100
    assert move["end_timestamp"] == 10.0
    assert result.diagnostics["fall_moves"] == 1
