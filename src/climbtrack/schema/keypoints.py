"""Versioned canonical keypoint registry derived from official Sapiens2 metadata."""

import ast
import json
from pathlib import Path
from typing import Any

from climbtrack.errors import SchemaValidationError

KEYPOINT_REGISTRY_VERSION = "1.0.0"
EXPECTED_KEYPOINTS = 308


def registry_from_sapiens_source(
    source: str, *, source_url: str, source_sha256: str
) -> dict[str, Any]:
    """Safely parse Meta's declarative metadata without executing downloaded code."""
    tree = ast.parse(source)
    dataset: dict[str, Any] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "dataset_info" for target in node.targets
        ):
            continue
        value = _static_value(node.value)
        if isinstance(value, dict) and value.get("dataset_name") == "goliath":
            dataset = value
            break
    if dataset is None:
        raise SchemaValidationError("Official Sapiens2 metadata lacks the Goliath dataset registry")

    raw = dataset.get("keypoint_info")
    if not isinstance(raw, dict):
        raise SchemaValidationError("Official Sapiens2 keypoint_info is invalid")
    entries = [dict(raw[index]) for index in sorted(raw)]
    if dataset.get("remove_teeth") is True:
        entries = [entry for entry in entries if not str(entry["name"]).startswith("teeth_")]
    if len(entries) != EXPECTED_KEYPOINTS:
        raise SchemaValidationError(
            f"Expected {EXPECTED_KEYPOINTS} Sapiens2 keypoints, found {len(entries)}"
        )

    group_names = {
        "body": set(dataset.get("body_keypoint_names", [])),
        "feet": set(dataset.get("foot_keypoint_names", [])),
        "left_hand": set(dataset.get("left_hand_keypoint_names", [])),
        "right_hand": set(dataset.get("right_hand_keypoint_names", [])),
        "extra": set(dataset.get("extra_keypoint_names", [])),
        "face": set(dataset.get("face_keypoint_names", [])),
    }
    keypoints = []
    for index, entry in enumerate(entries):
        name = str(entry["name"])
        groups = [group for group, names in group_names.items() if name in names]
        if len(groups) != 1:
            raise SchemaValidationError(f"Keypoint {name!r} maps to {len(groups)} groups")
        keypoints.append(
            {
                "index": index,
                "name": name,
                "group": groups[0],
                "swap": str(entry.get("swap", "")) or None,
            }
        )

    skeleton = dataset.get("skeleton_info")
    if not isinstance(skeleton, dict):
        raise SchemaValidationError("Official Sapiens2 skeleton_info is invalid")
    edges = [list(skeleton[index]["link"]) for index in sorted(skeleton)]
    names = {entry["name"] for entry in keypoints}
    if any(left not in names or right not in names for left, right in edges):
        raise SchemaValidationError("Official skeleton references an unknown keypoint")

    registry = {
        "schema_version": KEYPOINT_REGISTRY_VERSION,
        "backend": "sapiens2",
        "format": "sociopticon-308",
        "source_url": source_url,
        "source_sha256": source_sha256,
        "keypoints": keypoints,
        "skeleton_edges": edges,
    }
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    """Enforce stable indices, unique names and symmetric swap partners."""
    keypoints = registry.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != EXPECTED_KEYPOINTS:
        raise SchemaValidationError(f"Registry must contain {EXPECTED_KEYPOINTS} keypoints")
    indices = [entry.get("index") for entry in keypoints]
    if indices != list(range(EXPECTED_KEYPOINTS)):
        raise SchemaValidationError("Registry keypoint indices must be contiguous and ordered")
    by_name = {str(entry.get("name")): entry for entry in keypoints}
    if len(by_name) != EXPECTED_KEYPOINTS:
        raise SchemaValidationError("Registry keypoint names must be unique")
    for entry in keypoints:
        partner = entry.get("swap")
        if partner is None:
            continue
        if partner not in by_name or by_name[partner].get("swap") != entry["name"]:
            raise SchemaValidationError(f"Asymmetric keypoint swap mapping for {entry['name']}")


def write_registry(registry: dict[str, Any], path: Path) -> None:
    """Write validated canonical metadata as deterministic JSON."""
    validate_registry(registry)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_registry(path: Path) -> dict[str, Any]:
    """Read and validate canonical keypoint metadata."""
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"Invalid keypoint registry: {path}") from exc
    if not isinstance(registry, dict):
        raise SchemaValidationError(f"Keypoint registry must be an object: {path}")
    validate_registry(registry)
    return registry


def flip_pairs(registry: dict[str, Any]) -> list[list[int]]:
    """Return each left/right index pair exactly once."""
    by_name = {entry["name"]: entry for entry in registry["keypoints"]}
    pairs: list[list[int]] = []
    for entry in registry["keypoints"]:
        partner = entry["swap"]
        if partner is None:
            continue
        other = by_name[partner]
        if entry["index"] < other["index"]:
            pairs.append([entry["index"], other["index"]])
    return pairs


def _static_value(node: ast.AST) -> Any:
    """Evaluate only literals and ``dict(...)`` calls used by the metadata file."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {
            _static_value(key): _static_value(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.List):
        return [_static_value(value) for value in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_static_value(value) for value in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_static_value(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _static_value(node.left) * _static_value(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        result: dict[Any, Any] = {}
        for argument in node.args:
            result.update(_static_value(argument))
        result.update({keyword.arg: _static_value(keyword.value) for keyword in node.keywords})
        return result
    raise SchemaValidationError(f"Unsupported syntax in Sapiens2 metadata: {type(node).__name__}")
