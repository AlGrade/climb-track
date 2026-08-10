"""Explicit, atomic downloads for pinned model checkpoints."""

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from climbtrack.config import AppConfig, resolve_project_path
from climbtrack.errors import ClimbTrackError
from climbtrack.hashing import fingerprint_file
from climbtrack.schema.keypoints import registry_from_sapiens_source, write_registry


def ensure_yolo11_checkpoint(config: AppConfig, config_path: Path) -> tuple[Path, bool]:
    """Download the configured checkpoint only when the user invokes this operation."""
    destination = resolve_project_path(config.detection.model_path, config_path)
    if destination.is_file():
        fingerprint_file(destination)
        return destination, False

    url = config.models.yolo11.source_url
    if not url.startswith("https://"):
        raise ClimbTrackError("YOLO checkpoint source must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()

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
    os.replace(temporary, destination)
    fingerprint_file(destination)
    return destination, True


def ensure_sapiens2_checkpoint(config: AppConfig, config_path: Path) -> tuple[Path, bool]:
    """Explicitly download a pinned Transformers snapshot and canonical metadata."""
    model_config = config.models.sapiens2
    destination = resolve_project_path(model_config.model_dir, config_path)
    expected = (
        destination / "config.json",
        destination / "preprocessor_config.json",
        destination / model_config.checkpoint_filename,
        destination / "keypoints.json",
        destination / "download.json",
    )
    if destination.is_dir() and all(path.is_file() for path in expected):
        for path in expected:
            fingerprint_file(path)
        return destination, False
    if destination.exists():
        raise ClimbTrackError(
            f"Incomplete Sapiens2 model directory: {destination}. Move it aside and retry."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.download"
    temporary.mkdir(exist_ok=True)
    try:
        for filename in (
            "config.json",
            "preprocessor_config.json",
            model_config.checkpoint_filename,
        ):
            _download_huggingface_file(
                model_config.model_id,
                model_config.revision,
                filename,
                temporary / filename,
                expected_size=(
                    model_config.checkpoint_size_bytes
                    if filename == model_config.checkpoint_filename
                    else None
                ),
                connections=model_config.download_connections,
                segment_size=model_config.download_segment_mb * 1024 * 1024,
            )
        checkpoint = fingerprint_file(temporary / model_config.checkpoint_filename)
        if checkpoint["sha256"] != model_config.checkpoint_sha256:
            raise ClimbTrackError(
                "Sapiens2 checkpoint SHA-256 mismatch: "
                f"expected {model_config.checkpoint_sha256}, found {checkpoint['sha256']}"
            )
        source = _download_text(model_config.keypoint_source_url)
        source_sha256 = __import__("hashlib").sha256(source.encode()).hexdigest()
        registry = registry_from_sapiens_source(
            source,
            source_url=model_config.keypoint_source_url,
            source_sha256=source_sha256,
        )
        write_registry(registry, temporary / "keypoints.json")
        hub_cache = temporary / ".cache"
        if hub_cache.exists():
            shutil.rmtree(hub_cache)
        fingerprints = {
            path.name: fingerprint_file(path)
            for path in (
                temporary / "config.json",
                temporary / "preprocessor_config.json",
                temporary / model_config.checkpoint_filename,
                temporary / "keypoints.json",
            )
        }
        payload = {
            "schema_version": "1.0.0",
            "model_id": model_config.model_id,
            "revision": model_config.revision,
            "files": fingerprints,
        }
        (temporary / "download.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception as exc:
        if isinstance(exc, ClimbTrackError):
            raise
        raise ClimbTrackError(
            f"Could not download Sapiens2-1B: {exc}. Partial data was retained at {temporary}."
        ) from exc
    return destination, True


def _download_text(url: str) -> str:
    if not url.startswith("https://"):
        raise ClimbTrackError("Sapiens2 metadata source must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "climbtrack/0.1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except (OSError, UnicodeError, urllib.error.URLError) as exc:
        raise ClimbTrackError(f"Could not download Sapiens2 keypoint metadata: {exc}") from exc


def _download_huggingface_file(
    model_id: str,
    revision: str,
    filename: str,
    destination: Path,
    *,
    expected_size: int | None,
    connections: int,
    segment_size: int,
) -> None:
    """Download one immutable Hub file with curl resume and atomic publish."""
    if destination.is_file():
        return
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
    executable = shutil.which("curl")
    if executable is None:
        raise ClimbTrackError("Required resumable downloader 'curl' was not found in PATH")
    if expected_size is not None and connections > 1:
        _download_segmented(
            executable,
            url,
            destination,
            expected_size=expected_size,
            connections=connections,
            segment_size=segment_size,
        )
        return
    partial = destination.with_name(f".{destination.name}.part")
    process = subprocess.run(
        [
            executable,
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "15",
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=False,
    )
    if process.returncode != 0:
        raise ClimbTrackError(
            f"Could not download {filename} (curl exit {process.returncode}). "
            f"Partial file retained at {partial}."
        )
    if partial.stat().st_size == 0:
        raise ClimbTrackError(f"Downloaded Hugging Face file is empty: {filename}")
    os.replace(partial, destination)


def _download_segmented(
    executable: str,
    url: str,
    destination: Path,
    *,
    expected_size: int,
    connections: int,
    segment_size: int,
) -> None:
    """Mirror Xet's concurrent range strategy while keeping every chunk resumable."""
    segment_dir = destination.with_name(f".{destination.name}.segments")
    segment_dir.mkdir(exist_ok=True)
    ranges = [
        (index, start, min(expected_size - 1, start + segment_size - 1))
        for index, start in enumerate(range(0, expected_size, segment_size))
    ]
    complete = 0
    pending = []
    for index, start, end in ranges:
        path = segment_dir / f"{index:06d}.part"
        size = end - start + 1
        if path.is_file() and path.stat().st_size == size:
            complete += size
        else:
            pending.append((path, start, end))

    with Progress(
        TextColumn(destination.name),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    ) as progress:
        task = progress.add_task("parallel download", total=expected_size, completed=complete)
        with ThreadPoolExecutor(max_workers=connections) as executor:
            futures = {
                executor.submit(_download_range, executable, url, path, start, end): end - start + 1
                for path, start, end in pending
            }
            for future in as_completed(futures):
                future.result()
                progress.update(task, advance=futures[future])

    assembled = destination.with_name(f".{destination.name}.part")
    with assembled.open("wb") as output:
        for index, start, end in ranges:
            path = segment_dir / f"{index:06d}.part"
            expected = end - start + 1
            if path.stat().st_size != expected:
                raise ClimbTrackError(f"Sapiens2 segment has wrong size: {path}")
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    if assembled.stat().st_size != expected_size:
        raise ClimbTrackError("Assembled Sapiens2 checkpoint has the wrong byte size")
    os.replace(assembled, destination)
    shutil.rmtree(segment_dir)


def _download_range(executable: str, url: str, path: Path, start: int, end: int) -> None:
    temporary = path.with_suffix(".incomplete")
    expected = end - start + 1
    process = subprocess.run(
        [
            executable,
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "15",
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
            "--max-filesize",
            str(expected),
            "--range",
            f"{start}-{end}",
            "--output",
            str(temporary),
            "--write-out",
            "%{http_code}",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0 or process.stdout != "206":
        raise ClimbTrackError(
            f"Sapiens2 byte range {start}-{end} failed "
            f"(curl {process.returncode}, HTTP {process.stdout!r}): {process.stderr.strip()}"
        )
    if temporary.stat().st_size != expected:
        raise ClimbTrackError(f"Sapiens2 byte range {start}-{end} has the wrong size")
    os.replace(temporary, path)
