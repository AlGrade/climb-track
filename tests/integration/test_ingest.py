import shutil
import subprocess
from pathlib import Path

import pytest

from climbtrack.config import AppConfig
from climbtrack.stages.ingest import ingest_video

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)
def test_ingest_and_resume_tiny_video(tmp_path: Path) -> None:
    video = tmp_path / "tiny.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5:duration=0.6",
            "-c:v",
            "ffv1",
            str(video),
        ],
        check=True,
    )
    config = AppConfig.model_validate(
        {
            "project": {"cache_dir": str(tmp_path / "cache"), "seed": 42, "device": "cpu"},
            "ingest": {
                "ffmpeg_path": "ffmpeg",
                "ffprobe_path": "ffprobe",
                "hdr_policy": "fail",
            },
            "detection": {},
            "tracking": {},
            "selection": {},
            "render": {},
            "models": {
                "primary_pose_backend": "sapiens2",
                "yolo11": {
                    "model_id": "yolo11x",
                    "checkpoint_filename": "yolo11x.pt",
                    "checkpoint_sha256": (
                        "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"
                    ),
                    "checkpoint_size_bytes": 114636239,
                    "source_url": "https://example.test/yolo11x.pt",
                },
                "sapiens2": {
                    "model_id": "facebook/sapiens2-pose-1b",
                    "model_dir": "models/sapiens2-pose-1b",
                    "checkpoint_filename": "model.safetensors",
                    "checkpoint_sha256": (
                        "2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f"
                    ),
                    "checkpoint_size_bytes": 6079194752,
                    "revision": "f5fed8b97b99698d5eea1d14ff0855d0b4c3f000",
                    "keypoint_source_url": "https://example.test/keypoints.py",
                },
            },
        }
    )

    first = ingest_video(
        video,
        config=config,
        cache_root=tmp_path / "cache",
        project_root=tmp_path,
    )
    second = ingest_video(
        video,
        config=config,
        cache_root=tmp_path / "cache",
        project_root=tmp_path,
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert len(list((first.path / "frames").glob("*.png"))) == 3
    assert (first.path / "frames.parquet").is_file()
    assert (first.path / "metadata.json").is_file()
