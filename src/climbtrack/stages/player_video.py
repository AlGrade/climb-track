"""Stage 90: browser-friendly proxy for responsive move-player seeking."""

import json
import subprocess
from pathlib import Path

from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.errors import ClimbTrackError, ExternalToolError
from climbtrack.provenance import executable_version, git_state, resolve_executable, runtime_state

STAGE_NAME = "90_player_video"
STAGE_VERSION = "1.0.0"
OUTPUT_NAME = "player_video.mp4"

PLAYER_VIDEO_CONFIG = {
    "max_width": 1440,
    "codec": "libx264",
    "preset": "veryfast",
    "crf": 21,
    "keyframe_interval_frames": 15,
    "pixel_format": "yuv420p",
    "audio": "copy",
}


def prepare_player_video(
    pose_overlay: CacheResult,
    *,
    ffmpeg_path: str,
    cache_root: Path,
    project_root: Path,
    force: bool = False,
) -> CacheResult:
    """Create or reuse a smaller short-GOP video for browser playback."""
    source = pose_overlay.path / "skeleton_raw_overlay.mp4"
    if not source.is_file():
        raise ClimbTrackError(f"Pose overlay video does not exist: {source}")

    effective_config = dict(PLAYER_VIDEO_CONFIG)
    tools = {"ffmpeg": executable_version(ffmpeg_path)}
    input_fingerprint = {"render_pose": upstream_fingerprint(pose_overlay.manifest)}
    cache = StageCache(cache_root, STAGE_NAME)
    cache_key = cache.make_key(
        stage=STAGE_NAME,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
    )

    def build(output: Path) -> None:
        destination = output / OUTPUT_NAME
        command = player_video_command(
            resolve_executable(ffmpeg_path),
            source,
            destination,
        )
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode != 0:
            detail = process.stderr.strip() or "no error output"
            raise ExternalToolError(f"Player video encoding failed: {detail}")
        summary = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "purpose": "responsive_local_browser_playback",
            "source": source.name,
            **effective_config,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return cache.materialize(
        cache_key=cache_key,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
        runtime=runtime_state(),
        git=git_state(project_root),
        verify_checksums=True,
        force=force,
        builder=build,
    )


def player_video_command(executable: Path, source: Path, destination: Path) -> list[str]:
    """Build the deterministic FFmpeg command for the browser proxy."""
    keyframes = str(PLAYER_VIDEO_CONFIG["keyframe_interval_frames"])
    return [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"scale=w='min({PLAYER_VIDEO_CONFIG['max_width']},iw)':h=-2",
        "-c:v",
        str(PLAYER_VIDEO_CONFIG["codec"]),
        "-preset",
        str(PLAYER_VIDEO_CONFIG["preset"]),
        "-crf",
        str(PLAYER_VIDEO_CONFIG["crf"]),
        "-g",
        keyframes,
        "-keyint_min",
        keyframes,
        "-sc_threshold",
        "0",
        "-pix_fmt",
        str(PLAYER_VIDEO_CONFIG["pixel_format"]),
        "-fps_mode",
        "vfr",
        "-c:a",
        str(PLAYER_VIDEO_CONFIG["audio"]),
        "-movflags",
        "+faststart",
        "-y",
        str(destination),
    ]
