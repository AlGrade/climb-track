from pathlib import Path

import pytest

from climbtrack.config import load_config
from climbtrack.errors import ClimbTrackError
from climbtrack.model_downloads import ensure_sapiens2_checkpoint, ensure_yolo11_checkpoint


def test_existing_sapiens_checkpoint_must_match_pinned_hash(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "default.yaml")
    model_dir = tmp_path / "sapiens2"
    model_dir.mkdir()
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "keypoints.json",
        "download.json",
    ):
        (model_dir / filename).write_text("wrong", encoding="utf-8")
    sapiens = config.models.sapiens2.model_copy(update={"model_dir": model_dir})
    models = config.models.model_copy(update={"sapiens2": sapiens})
    test_config = config.model_copy(update={"models": models})

    with pytest.raises(ClimbTrackError, match="SHA-256 mismatch"):
        ensure_sapiens2_checkpoint(test_config, project_root / "configs" / "default.yaml")


def test_existing_yolo_checkpoint_must_match_pinned_identity(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "default.yaml")
    checkpoint = tmp_path / "yolo11x.pt"
    checkpoint.write_text("wrong", encoding="utf-8")
    detection = config.detection.model_copy(update={"model_path": checkpoint})
    test_config = config.model_copy(update={"detection": detection})

    with pytest.raises(ClimbTrackError, match="identity mismatch"):
        ensure_yolo11_checkpoint(test_config, project_root / "configs" / "default.yaml")
