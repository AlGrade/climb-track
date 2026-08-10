"""Stage 00: content-addressed video ingest."""

import json
from pathlib import Path
from typing import Any

from climbtrack.cache import CacheResult, StageCache
from climbtrack.config import AppConfig
from climbtrack.errors import ConfigurationError
from climbtrack.hashing import fingerprint_file
from climbtrack.provenance import executable_version, git_state, runtime_state
from climbtrack.schema.frames import write_frame_index
from climbtrack.video.decode import decode_frames
from climbtrack.video.probe import run_ffprobe

STAGE_NAME = "00_ingest"
STAGE_VERSION = "1.0.0"


def ingest_video(
    video: Path,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Probe, decode and index one video with cache/resume semantics."""
    video = video.expanduser().resolve()
    if not video.is_file():
        raise ConfigurationError(f"Input video is not a readable file: {video}")

    input_fingerprint = fingerprint_file(video)
    effective_config = config.ingest.model_dump(mode="json")
    tools = {
        "ffmpeg": executable_version(config.ingest.ffmpeg_path),
        "ffprobe": executable_version(config.ingest.ffprobe_path),
    }
    runtime = runtime_state()
    git = git_state(project_root)
    cache = StageCache(cache_root, STAGE_NAME)
    cache_key = cache.make_key(
        stage=STAGE_NAME,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
    )

    def build(output: Path) -> None:
        probe = run_ffprobe(video, config.ingest.ffprobe_path)
        (output / "ffprobe.json").write_text(
            json.dumps(probe.raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        frames = decode_frames(video, output / "frames", probe.metadata, config.ingest)
        if len(frames) != len(probe.frames):
            raise AssertionError("Decoder and probe frame counts diverged after validation")
        write_frame_index(probe.frames, output / "frames.parquet")
        metadata: dict[str, Any] = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "stage_version": STAGE_VERSION,
            "source": input_fingerprint,
            "video": probe.metadata.as_dict(),
            "effective_config": effective_config,
            "tools": tools,
            "runtime": runtime,
            "git": git,
            "models": {},
            "ffprobe_output": "ffprobe.json",
            "frame_index": "frames.parquet",
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return cache.materialize(
        cache_key=cache_key,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
        runtime=runtime,
        git=git,
        verify_checksums=config.ingest.verify_cached_checksums,
        force=force,
        builder=build,
    )
