"""Resumable HTTPS helpers for immutable Hugging Face files."""

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from climbtrack.errors import ClimbTrackError


def download_huggingface_file(
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
    """Download independent byte ranges and retain every completed segment."""
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
        executor = ThreadPoolExecutor(max_workers=connections)
        futures = {
            executor.submit(_download_range, executable, url, path, start, end): end - start + 1
            for path, start, end in pending
        }
        try:
            for future in as_completed(futures):
                future.result()
                progress.update(task, advance=futures[future])
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

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
