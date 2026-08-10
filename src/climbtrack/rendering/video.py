"""Shared VFR timing and FFmpeg encoding utilities for overlay stages."""

import statistics
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

from climbtrack.config import AppConfig
from climbtrack.errors import ExternalToolError
from climbtrack.provenance import resolve_executable


def frame_durations(frames: list[dict[str, Any]]) -> list[float]:
    """Derive one positive display duration per variable-rate source frame."""
    if not frames:
        raise ValueError("At least one frame is required")
    if len(frames) == 1:
        duration = frames[0]["duration"]
        if duration is None or float(duration) <= 0:
            raise ValueError("A single frame requires a positive duration")
        return [float(duration)]
    timestamps = [float(frame["timestamp"]) for frame in frames]
    deltas = [right - left for left, right in pairwise(timestamps)]
    if any(delta <= 0 for delta in deltas):
        raise ValueError("Frame timestamps must be strictly increasing")
    fallback = statistics.median(deltas)
    last = frames[-1]["duration"]
    last_duration = float(last) if last is not None and float(last) > 0 else fallback
    return [*deltas, last_duration]


def escape_concat_path(path: Path) -> str:
    """Escape one absolute path for FFmpeg's concat demuxer."""
    return str(path.resolve()).replace("'", "'\\''")


def encode_overlay(
    concat_path: Path,
    source_video: Path,
    output_video: Path,
    duration_seconds: float,
    config: AppConfig,
) -> None:
    """Encode VFR overlay frames and preserve optional source audio."""
    executable = resolve_executable(config.render.ffmpeg_path)
    command = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        config.render.codec,
        "-preset",
        config.render.preset,
        "-crf",
        str(config.render.crf),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "vfr",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration_seconds:.9f}",
        "-movflags",
        "+faststart",
        "-y",
        str(output_video),
    ]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        detail = process.stderr.strip() or "no error output"
        raise ExternalToolError(f"ffmpeg overlay encoding failed: {detail}")
