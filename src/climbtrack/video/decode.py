"""Deterministic lossless frame extraction with explicit HDR policy."""

import subprocess
from pathlib import Path

from climbtrack.config import HdrPolicy, IngestConfig
from climbtrack.errors import ConfigurationError, ExternalToolError
from climbtrack.provenance import resolve_executable
from climbtrack.video.probe import VideoMetadata


def decode_frames(
    video: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    config: IngestConfig,
) -> tuple[Path, ...]:
    """Decode every source frame to a lossless, zero-based PNG sequence."""
    if metadata.hdr and config.hdr_policy == HdrPolicy.FAIL:
        raise ConfigurationError(
            "HDR input detected. Set ingest.hdr_policy to 'tonemap' or 'clip' explicitly; "
            "ingest will not silently discard HDR range."
        )

    executable = resolve_executable(config.ffmpeg_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_pattern = output_dir / "%09d.png"
    command = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+bitexact",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
        "-start_number",
        "0",
        "-threads",
        "1",
    ]
    video_filter = _video_filter(metadata, config.hdr_policy)
    if video_filter:
        command.extend(["-vf", video_filter])
    command.extend(
        [
            "-compression_level",
            str(config.png_compression),
            "-y",
            str(output_pattern),
        ]
    )

    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        detail = process.stderr.strip() or "no error output"
        raise ExternalToolError(f"ffmpeg frame extraction failed: {detail}")
    frames = tuple(sorted(output_dir.glob("*.png")))
    if len(frames) != metadata.frame_count:
        raise ExternalToolError(
            f"Frame count mismatch: ffprobe reported {metadata.frame_count}, "
            f"ffmpeg wrote {len(frames)}"
        )
    return frames


def _video_filter(metadata: VideoMetadata, policy: HdrPolicy) -> str:
    if not metadata.hdr:
        return "format=rgb24"
    if policy == HdrPolicy.CLIP:
        return "format=rgb24"
    if policy == HdrPolicy.TONEMAP:
        return (
            "zscale=transfer=linear:npl=100,format=gbrpf32le,"
            "zscale=primaries=bt709,tonemap=tonemap=hable:desat=0,"
            "zscale=transfer=bt709:matrix=bt709:range=tv,format=rgb24"
        )
    raise AssertionError(f"Unhandled HDR policy: {policy}")
