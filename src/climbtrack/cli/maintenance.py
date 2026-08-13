"""Commands that prepare and inspect the environment rather than analyse a video."""

import json

from rich.table import Table

from climbtrack.cache.manifest import CacheManifest
from climbtrack.cli.app import app
from climbtrack.cli.options import DEFAULT_CONFIG, ConfigOption
from climbtrack.cli.reporting import abort, console
from climbtrack.config import load_config, resolve_project_path
from climbtrack.device import require_torch_device
from climbtrack.errors import ClimbTrackError
from climbtrack.model_downloads import (
    ensure_sapiens2_checkpoint,
    ensure_yolo11_checkpoint,
    verify_sapiens2_checkpoint,
    verify_yolo11_checkpoint,
)
from climbtrack.pipeline import build_context
from climbtrack.provenance import executable_version


@app.command("preflight")
def preflight(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Validate tools, checkpoint, and the explicitly configured compute device."""
    try:
        context = build_context(config_path)
        table = Table(title="Milestone 3 preflight")
        table.add_column("Check")
        table.add_column("Resolved value")
        table.add_row("device", str(require_torch_device(context.config.project.device)))
        for name, binary in (
            ("ffmpeg", context.config.ingest.ffmpeg_path),
            ("ffprobe", context.config.ingest.ffprobe_path),
        ):
            tool = executable_version(binary)
            table.add_row(name, f"{tool['path']}\n{tool['version']}")
        model_path = resolve_project_path(context.config.detection.model_path, config_path)
        if not model_path.is_file():
            raise ClimbTrackError(
                f"YOLO11x checkpoint is missing: {model_path}. Run 'climbtrack download-yolo'."
            )
        model = verify_yolo11_checkpoint(
            model_path,
            context.config.models.yolo11.checkpoint_sha256,
            context.config.models.yolo11.checkpoint_size_bytes,
        )
        table.add_row("YOLO11x", f"{model_path}\nsha256 {model['sha256'][:16]}…")
        sapiens_dir = resolve_project_path(context.config.models.sapiens2.model_dir, config_path)
        sapiens_checkpoint = sapiens_dir / context.config.models.sapiens2.checkpoint_filename
        if not sapiens_checkpoint.is_file():
            raise ClimbTrackError(
                f"Sapiens2-1B checkpoint is missing: {sapiens_checkpoint}. "
                "Run 'climbtrack download-sapiens'."
            )
        sapiens = verify_sapiens2_checkpoint(
            sapiens_checkpoint,
            context.config.models.sapiens2.checkpoint_sha256,
        )
        table.add_row("Sapiens2-1B", f"{sapiens_dir}\nsha256 {sapiens['sha256'][:16]}…")
        table.add_row("cache", str(context.cache_root))
        console.print(table)
    except (ClimbTrackError, OSError) as exc:
        abort(exc)


@app.command("download-yolo")
def download_yolo(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Explicitly download the pinned YOLO11x checkpoint."""
    try:
        config = load_config(config_path)
        path, downloaded = ensure_yolo11_checkpoint(config, config_path)
        state = "downloaded" if downloaded else "already present"
        console.print(f"[green]YOLO11x {state}:[/green] {path}")
    except (ClimbTrackError, OSError) as exc:
        abort(exc)


@app.command("download-sapiens")
def download_sapiens(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Explicitly download the pinned 6.08-GB Sapiens2-1B snapshot."""
    try:
        config = load_config(config_path)
        path, downloaded = ensure_sapiens2_checkpoint(config, config_path)
        state = "downloaded" if downloaded else "already present"
        console.print(f"[green]Sapiens2-1B {state}:[/green] {path}")
    except (ClimbTrackError, OSError) as exc:
        abort(exc)


@app.command("cache-list")
def cache_list(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """List complete entries for every implemented stage."""
    try:
        context = build_context(config_path)
        table = Table(title="Pipeline cache")
        table.add_column("Stage")
        table.add_column("Key")
        table.add_column("Created")
        found = False
        for stage_root in sorted(context.cache_root.glob("[0-9][0-9]_*")):
            for manifest_path in sorted(stage_root.glob("*/manifest.json")):
                manifest = CacheManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                if manifest.status == "complete":
                    found = True
                    table.add_row(
                        manifest.stage,
                        manifest.cache_key[:12],
                        manifest.created_at.isoformat(),
                    )
        console.print(table if found else "No complete cache entries.")
    except (ClimbTrackError, OSError, ValueError, json.JSONDecodeError) as exc:
        abort(exc)
