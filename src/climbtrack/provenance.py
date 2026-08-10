"""Runtime and source provenance collection."""

import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from climbtrack.errors import ExternalToolError


def resolve_executable(binary: str) -> Path:
    """Resolve an executable path or raise an actionable error."""
    candidate = Path(binary).expanduser()
    if candidate.is_absolute() or "/" in binary:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate.resolve()
        raise ExternalToolError(f"Executable is missing or not executable: {candidate}")

    resolved = shutil.which(binary)
    if resolved is None:
        raise ExternalToolError(
            f"Required executable '{binary}' was not found in PATH. "
            "Install it or set its explicit path in configs/default.yaml."
        )
    return Path(resolved).resolve()


def executable_version(binary: str) -> dict[str, str]:
    """Capture the resolved executable and its first version line."""
    executable = resolve_executable(binary)
    process = subprocess.run(
        [str(executable), "-version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise ExternalToolError(f"Could not query {executable}: {detail}")
    first_line = process.stdout.splitlines()[0] if process.stdout else "unknown"
    return {"path": str(executable), "version": first_line}


def git_state(project_root: Path) -> dict[str, Any]:
    """Read the current Git commit and dirty status without failing outside Git."""
    commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        return {"commit": None, "dirty": None}
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def runtime_state() -> dict[str, str]:
    """Capture the Python and operating-system identity."""
    state = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in (
        "climbtrack",
        "lap",
        "numpy",
        "opencv-python",
        "pyarrow",
        "pydantic",
        "torch",
        "torchvision",
        "ultralytics",
    ):
        try:
            state[f"package:{package}"] = version(package)
        except PackageNotFoundError:
            state[f"package:{package}"] = "not-installed"
    return state
