import pytest

from climbtrack.errors import SchemaValidationError
from climbtrack.schema.keypoints import registry_from_sapiens_source, validate_registry


def _metadata_source() -> str:
    names = [f"point_{index}" for index in range(308)]
    keypoints = {index: {"name": name, "id": index, "swap": ""} for index, name in enumerate(names)}
    dataset = {
        "dataset_name": "goliath",
        "keypoint_info": keypoints,
        "skeleton_info": {0: {"link": (names[0], names[1]), "id": 0}},
        "remove_teeth": False,
        "body_keypoint_names": names,
        "foot_keypoint_names": [],
        "left_hand_keypoint_names": [],
        "right_hand_keypoint_names": [],
        "extra_keypoint_names": [],
        "face_keypoint_names": [],
    }
    return f"dataset_info = {dataset!r}\n"


def test_registry_is_statically_parsed_and_grouped() -> None:
    registry = registry_from_sapiens_source(
        _metadata_source(), source_url="https://example.test/metadata.py", source_sha256="abc"
    )

    assert len(registry["keypoints"]) == 308
    assert registry["keypoints"][42] == {
        "index": 42,
        "name": "point_42",
        "group": "body",
        "swap": None,
    }
    assert registry["skeleton_edges"] == [["point_0", "point_1"]]


def test_registry_parser_rejects_executable_syntax() -> None:
    source = "dataset_info = dict(dataset_name='goliath', keypoint_info=danger())\n"
    with pytest.raises(SchemaValidationError, match="Unsupported syntax"):
        registry_from_sapiens_source(source, source_url="https://example.test", source_sha256="x")


def test_registry_rejects_asymmetric_swaps() -> None:
    registry = registry_from_sapiens_source(
        _metadata_source(), source_url="https://example.test/metadata.py", source_sha256="abc"
    )
    registry["keypoints"][0]["swap"] = "point_1"
    with pytest.raises(SchemaValidationError, match="Asymmetric"):
        validate_registry(registry)
