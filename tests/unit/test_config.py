from pathlib import Path

import pytest

from climbtrack.config import Device, load_config, resolve_cache_dir
from climbtrack.errors import ConfigurationError


def test_loads_strict_config(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "test.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
project:
  cache_dir: cache
  seed: 7
  device: mps
ingest:
  ffmpeg_path: ffmpeg
  ffprobe_path: ffprobe
  frame_format: png
  png_compression: 3
  hdr_policy: fail
  verify_cached_checksums: true
detection: {}
tracking: {}
selection: {}
render: {}
models:
  primary_pose_backend: sapiens2
  yolo11:
    model_id: yolo11x
    checkpoint_filename: yolo11x.pt
    checkpoint_sha256: 7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24
    checkpoint_size_bytes: 114636239
    source_url: https://example.test/yolo11x.pt
  sapiens2:
    model_id: facebook/sapiens2-pose-1b
    model_dir: models/sapiens2-pose-1b
    checkpoint_filename: model.safetensors
    checkpoint_sha256: 2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f
    checkpoint_size_bytes: 6079194752
    download_connections: 16
    download_segment_mb: 8
    revision: f5fed8b97b99698d5eea1d14ff0855d0b4c3f000
    keypoint_source_url: https://example.test/keypoints.py
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project.device is Device.MPS
    assert config.pose.flip_tta is True
    assert config.pose.multi_scale_tta == (1.0, 1.125)
    assert config.refine.smoothing_groups == ("left_hand", "right_hand")
    assert config.refine.confidence_threshold_overrides["feet"] == 0.05
    assert resolve_cache_dir(config, config_path) == tmp_path / "cache"


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("project: {device: cpu, surprise: true}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_config(config_path)
