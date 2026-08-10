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
    source_url: https://example.test/yolo11x.pt
  sapiens2:
    model_id: facebook/sapiens2-pose-1b
    checkpoint_filename: sapiens2_1b_pose.safetensors
    revision: null
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project.device is Device.MPS
    assert resolve_cache_dir(config, config_path) == tmp_path / "cache"


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("project: {device: cpu, surprise: true}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_config(config_path)
