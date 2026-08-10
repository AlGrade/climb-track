"""ffprobe execution and normalized video metadata."""

import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from climbtrack.errors import ExternalToolError
from climbtrack.provenance import resolve_executable

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


@dataclass(frozen=True)
class FrameTiming:
    """One decoded frame's source timeline information."""

    frame_idx: int
    timestamp: float
    duration: float | None
    source_pts: int | None


@dataclass(frozen=True)
class VideoMetadata:
    """Normalized metadata needed by downstream stages."""

    width: int
    height: int
    display_width: int
    display_height: int
    rotation_degrees: float
    codec_name: str
    pixel_format: str | None
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    nominal_fps: float
    average_fps: float
    frame_count: int
    duration_seconds: float
    start_timestamp: float
    variable_frame_rate: bool
    hdr: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    """Raw ffprobe data plus normalized metadata and timestamps."""

    raw: dict[str, Any]
    metadata: VideoMetadata
    frames: tuple[FrameTiming, ...]


def run_ffprobe(video: Path, ffprobe_path: str) -> ProbeResult:
    """Probe a video, including every decoded frame timestamp."""
    executable = resolve_executable(ffprobe_path)
    command = [
        str(executable),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_format",
        "-show_streams",
        "-show_frames",
        "-print_format",
        "json",
        str(video),
    ]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        detail = process.stderr.strip() or "no error output"
        raise ExternalToolError(f"ffprobe failed for {video}: {detail}")
    try:
        raw = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ExternalToolError(f"ffprobe returned invalid JSON for {video}") from exc
    return parse_probe(raw)


def parse_probe(raw: dict[str, Any]) -> ProbeResult:
    """Normalize ffprobe JSON without inventing missing timestamps."""
    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ExternalToolError("ffprobe did not report a video stream")
    stream = streams[0]

    raw_frames = raw.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ExternalToolError("ffprobe did not report decoded video frames")

    frames: list[FrameTiming] = []
    for frame_idx, frame in enumerate(raw_frames):
        timestamp_text = frame.get("best_effort_timestamp_time")
        if timestamp_text is None:
            timestamp_text = frame.get("pts_time")
        if timestamp_text is None:
            raise ExternalToolError(
                f"Frame {frame_idx} has no source timestamp; ingest will not approximate it"
            )
        timestamp = _finite_float(timestamp_text, f"frame {frame_idx} timestamp")
        duration_text = frame.get("pkt_duration_time") or frame.get("duration_time")
        duration = (
            _finite_float(duration_text, f"frame {frame_idx} duration")
            if duration_text is not None
            else None
        )
        source_pts_raw = frame.get("best_effort_timestamp")
        if source_pts_raw is None:
            source_pts_raw = frame.get("pts")
        source_pts = int(source_pts_raw) if source_pts_raw is not None else None
        frames.append(FrameTiming(frame_idx, timestamp, duration, source_pts))

    timestamps = [frame.timestamp for frame in frames]
    if any(right <= left for left, right in pairwise(timestamps)):
        raise ExternalToolError("Video frame timestamps are not strictly increasing")

    width = int(stream["width"])
    height = int(stream["height"])
    rotation = _rotation(stream)
    quarter_turn = math.isclose(abs(rotation) % 180.0, 90.0, abs_tol=0.01)
    display_width, display_height = (height, width) if quarter_turn else (width, height)

    nominal_fps = _rate(stream.get("r_frame_rate"), "nominal frame rate")
    average_fps = _rate(stream.get("avg_frame_rate"), "average frame rate")
    duration = _duration(raw, stream, frames)
    transfer = stream.get("color_transfer")
    metadata = VideoMetadata(
        width=width,
        height=height,
        display_width=display_width,
        display_height=display_height,
        rotation_degrees=rotation,
        codec_name=str(stream.get("codec_name", "unknown")),
        pixel_format=stream.get("pix_fmt"),
        color_primaries=stream.get("color_primaries"),
        color_transfer=transfer,
        color_space=stream.get("color_space"),
        nominal_fps=nominal_fps,
        average_fps=average_fps,
        frame_count=len(frames),
        duration_seconds=duration,
        start_timestamp=timestamps[0],
        variable_frame_rate=_is_vfr(timestamps, nominal_fps, average_fps),
        hdr=transfer in HDR_TRANSFERS,
    )
    return ProbeResult(raw=raw, metadata=metadata, frames=tuple(frames))


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExternalToolError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ExternalToolError(f"Non-finite {label}: {value!r}")
    return result


def _rate(value: Any, label: str) -> float:
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ExternalToolError(f"Invalid {label}: {value!r}") from exc
    if fraction <= 0:
        raise ExternalToolError(f"Non-positive {label}: {value!r}")
    return float(fraction)


def _rotation(stream: dict[str, Any]) -> float:
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        return _finite_float(tags["rotate"], "rotation") % 360.0
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return _finite_float(side_data["rotation"], "rotation") % 360.0
    return 0.0


def _duration(raw: dict[str, Any], stream: dict[str, Any], frames: list[FrameTiming]) -> float:
    for candidate in (stream.get("duration"), (raw.get("format") or {}).get("duration")):
        if candidate is not None:
            return _finite_float(candidate, "video duration")
    last_duration = frames[-1].duration or 0.0
    return frames[-1].timestamp - frames[0].timestamp + last_duration


def _is_vfr(timestamps: list[float], nominal_fps: float, average_fps: float) -> bool:
    if not math.isclose(nominal_fps, average_fps, rel_tol=1e-6, abs_tol=1e-6):
        return True
    if len(timestamps) < 3:
        return False
    deltas = [right - left for left, right in pairwise(timestamps)]
    median = statistics.median(deltas)
    tolerance = max(1e-6, median * 1e-3)
    return any(abs(delta - median) > tolerance for delta in deltas)
