"""Command-line interface for independently resumable pipeline stages."""

import json
import random
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from climbtrack.cache.manifest import CacheManifest
from climbtrack.config import AppConfig, load_config, resolve_cache_dir
from climbtrack.errors import ClimbTrackError
from climbtrack.provenance import executable_version
from climbtrack.stages.ingest import ingest_video

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Offline 2D skeleton tracking for climbing videos.",
)
console = Console()

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Strict YAML configuration.",
        dir_okay=False,
        readable=True,
    ),
]
VideoArgument = Annotated[
    Path,
    typer.Argument(help="Input climbing video.", dir_okay=False, readable=True),
]


@app.command("preflight")
def preflight(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    """Validate configuration and required Milestone-1 executables."""
    try:
        config = load_config(config_path)
        table = Table(title="Milestone 1 preflight")
        table.add_column("Check")
        table.add_column("Resolved value")
        table.add_row("device", config.project.device.value)
        for name, binary in (
            ("ffmpeg", config.ingest.ffmpeg_path),
            ("ffprobe", config.ingest.ffprobe_path),
        ):
            version = executable_version(binary)
            table.add_row(name, f"{version['path']}\n{version['version']}")
        table.add_row("cache", str(resolve_cache_dir(config, config_path)))
        console.print(table)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("ingest")
def ingest(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild and retain the previous entry as a backup."),
    ] = False,
) -> None:
    """Run Stage 00 and print the immutable cache location."""
    _execute_ingest(video, config_path, force)


@app.command("run-all")
def run_all(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild completed stages without deleting old entries."),
    ] = False,
) -> None:
    """Run every implemented stage in order (Milestone 1: ingest)."""
    _execute_ingest(video, config_path, force)
    console.print("[green]All currently implemented stages completed.[/green]")


@app.command("cache-list")
def cache_list(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    """List complete Stage-00 entries without modifying cache state."""
    try:
        config = load_config(config_path)
        stage_root = resolve_cache_dir(config, config_path) / "00_ingest"
        table = Table(title="Stage 00 cache")
        table.add_column("Key")
        table.add_column("Created")
        table.add_column("Input")
        found = False
        for manifest_path in sorted(stage_root.glob("*/manifest.json")):
            manifest = CacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            if manifest.status != "complete":
                continue
            found = True
            table.add_row(
                manifest.cache_key[:12],
                manifest.created_at.isoformat(),
                str(manifest.input_fingerprint.get("path", "unknown")),
            )
        if found:
            console.print(table)
        else:
            console.print("No complete Stage-00 cache entries.")
    except (ClimbTrackError, OSError, ValueError, json.JSONDecodeError) as exc:
        _abort(exc)


def _execute_ingest(video: Path, config_path: Path, force: bool) -> None:
    try:
        config = load_config(config_path)
        _seed(config)
        project_root = config_path.resolve().parent.parent
        result = ingest_video(
            video,
            config=config,
            cache_root=resolve_cache_dir(config, config_path),
            project_root=project_root,
            force=force,
        )
        state = "cache hit" if result.cache_hit else "created"
        console.print(f"[green]00_ingest {state}:[/green] {result.path}")
    except ClimbTrackError as exc:
        _abort(exc)


def _seed(config: AppConfig) -> None:
    random.seed(config.project.seed)
    np.random.seed(config.project.seed)


def _abort(exc: Exception) -> None:
    console.print(f"[bold red]Error:[/bold red] {exc}", highlight=False)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
