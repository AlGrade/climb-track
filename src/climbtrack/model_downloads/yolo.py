"""Atomic YOLO11 checkpoint download."""

import os
import urllib.error
import urllib.request
from pathlib import Path

from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from climbtrack.config import AppConfig, resolve_project_path
from climbtrack.errors import ClimbTrackError
from climbtrack.hashing import fingerprint_file


def ensure_yolo11_checkpoint(config: AppConfig, config_path: Path) -> tuple[Path, bool]:
    """Download the configured checkpoint only when explicitly requested."""
    destination = resolve_project_path(config.detection.model_path, config_path)
    if destination.is_file():
        verify_yolo11_checkpoint(
            destination,
            config.models.yolo11.checkpoint_sha256,
            config.models.yolo11.checkpoint_size_bytes,
        )
        return destination, False

    url = config.models.yolo11.source_url
    if not url.startswith("https://"):
        raise ClimbTrackError("YOLO checkpoint source must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.unlink(missing_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "climbtrack/0.1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0")) or None
            with (
                temporary.open("wb") as output,
                Progress(
                    TextColumn("YOLO11x"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                ) as progress,
            ):
                task = progress.add_task("download", total=total)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    progress.update(task, advance=len(chunk))
                output.flush()
                os.fsync(output.fileno())
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise ClimbTrackError(f"Could not download YOLO11x checkpoint: {exc}") from exc

    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise ClimbTrackError("Downloaded YOLO11x checkpoint is empty")
    verify_yolo11_checkpoint(
        temporary,
        config.models.yolo11.checkpoint_sha256,
        config.models.yolo11.checkpoint_size_bytes,
    )
    os.replace(temporary, destination)
    return destination, True


def verify_yolo11_checkpoint(
    path: Path,
    expected_sha256: str,
    expected_size: int,
) -> dict[str, int | str]:
    """Reject a YOLO checkpoint whose bytes differ from the pinned artifact."""
    fingerprint = fingerprint_file(path)
    actual_sha256 = str(fingerprint["sha256"])
    actual_size = int(fingerprint["size_bytes"])
    if actual_sha256 != expected_sha256 or actual_size != expected_size:
        raise ClimbTrackError(
            "YOLO11x checkpoint identity mismatch: "
            f"expected {expected_size} bytes / {expected_sha256}, "
            f"found {actual_size} bytes / {actual_sha256}"
        )
    return fingerprint
