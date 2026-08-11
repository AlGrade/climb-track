from climbtrack.config import RefineConfig
from climbtrack.refinement import refine_pose_records
from climbtrack.refinement.one_euro import OneEuro2D


def _record(frame_idx: int, name: str, x: float, confidence: float = 1.0) -> dict[str, object]:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 60,
        "track_id": 1,
        "keypoint_name": name,
        "x": x,
        "y": 0.0,
        "confidence": confidence,
        "is_missing": False,
        "is_interpolated": False,
        "source_backend": "test",
    }


def _registry(names: list[str], edges: list[list[str]] | None = None) -> dict[str, object]:
    return {
        "keypoints": [
            {"index": index, "name": name, "group": "body", "swap": None}
            for index, name in enumerate(names)
        ],
        "skeleton_edges": edges or [],
    }


def test_short_confidence_gap_is_interpolated() -> None:
    records = [
        _record(index, "joint", float(index), 0.1 if index in {1, 2} else 0.9) for index in range(5)
    ]

    result = refine_pose_records(
        records,
        _registry(["joint"]),
        {index: 100.0 for index in range(5)},
        RefineConfig(maximum_interpolation_gap=2, confidence_threshold_overrides={}),
    )

    assert result.diagnostics["confidence_gated"] == 2
    assert result.diagnostics["interpolated"] == 2
    assert not any(record["is_missing"] for record in result.records)
    assert [record["is_interpolated"] for record in result.records] == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_long_gap_remains_missing() -> None:
    records = [
        _record(index, "joint", float(index), 0.1 if index in {1, 2} else 0.9) for index in range(4)
    ]

    result = refine_pose_records(
        records,
        _registry(["joint"]),
        {index: 100.0 for index in range(4)},
        RefineConfig(maximum_interpolation_gap=1, confidence_threshold_overrides={}),
    )

    assert result.diagnostics["missing_final"] == 2
    assert result.records[1]["x"] is None
    assert result.records[1]["confidence"] is None


def test_obvious_low_confidence_left_right_swap_is_repaired() -> None:
    registry = _registry(["left", "right"])
    registry["keypoints"][0]["swap"] = "right"
    registry["keypoints"][1]["swap"] = "left"
    records = [
        _record(0, "left", 0),
        _record(0, "right", 100),
        _record(1, "left", 100, 0.4),
        _record(1, "right", 0, 0.4),
    ]

    result = refine_pose_records(
        records,
        registry,
        {0: 100.0, 1: 100.0},
        RefineConfig(),
    )

    assert result.diagnostics["swap_pairs"] == 1
    second_frame = {row["keypoint_name"]: row for row in result.records if row["frame_idx"] == 1}
    assert second_frame["left"]["x"] == 0
    assert second_frame["right"]["x"] == 100


def test_segment_length_outlier_is_replaced_from_neighbors() -> None:
    records = []
    for index in range(5):
        records.append(_record(index, "anchor", 0, 0.1 if index == 2 else 0.9))
        records.append(_record(index, "tip", 100 if index == 2 else 10, 0.2 if index == 2 else 0.9))

    result = refine_pose_records(
        records,
        _registry(["anchor", "tip"], [["anchor", "tip"]]),
        {index: 100.0 for index in range(5)},
        RefineConfig(),
    )

    assert result.diagnostics["segment_outliers"] == 1
    assert result.diagnostics["interpolated"] == 1
    repaired = next(
        row for row in result.records if row["frame_idx"] == 2 and row["keypoint_name"] == "tip"
    )
    assert repaired["x"] == 10
    assert repaired["is_interpolated"] is True


def test_one_euro_reduces_stationary_jitter() -> None:
    filter_2d = OneEuro2D(min_cutoff=4.0, beta=0.01, derivative_cutoff=1.0)
    raw = [0.0, 2.0, -2.0, 2.0, -2.0, 0.0]
    filtered = [filter_2d.update(index / 60, value, 0.0)[0] for index, value in enumerate(raw)]

    assert max(filtered[1:-1]) - min(filtered[1:-1]) < max(raw[1:-1]) - min(raw[1:-1])
