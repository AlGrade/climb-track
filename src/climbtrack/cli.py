"""Command-line interface for independently resumable pipeline stages."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from climbtrack.cache import CacheResult
from climbtrack.cache.manifest import CacheManifest
from climbtrack.config import AppConfig, load_config, resolve_cache_dir, resolve_project_path
from climbtrack.device import require_torch_device, seed_torch
from climbtrack.errors import ClimbTrackError, SelectionUncertainError
from climbtrack.hashing import fingerprint_file
from climbtrack.models import ensure_yolo11_checkpoint
from climbtrack.provenance import executable_version
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.tracks import read_tracks
from climbtrack.selection.click import choose_track_by_click
from climbtrack.stages.detect import detect_people
from climbtrack.stages.ingest import ingest_video
from climbtrack.stages.render_tracks import render_tracking_overlay
from climbtrack.stages.select import select_climber
from climbtrack.stages.track import track_people

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Offline, reproducible person tracking for climbing videos.",
)
console = Console()

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Strict YAML configuration.", dir_okay=False),
]
ForceOption = Annotated[
    bool,
    typer.Option("--force", help="Rebuild the stage while retaining the previous entry."),
]
TrackIdOption = Annotated[
    int | None,
    typer.Option("--track-id", min=1, help="Explicit climber track ID."),
]
ClickOption = Annotated[
    bool,
    typer.Option("--click", help="Choose the climber by clicking a displayed bounding box."),
]
ReviewAllOption = Annotated[
    bool,
    typer.Option(
        "--review-all",
        help="Render every track ID without selecting a climber (for ambiguity review).",
    ),
]
VideoArgument = Annotated[
    Path,
    typer.Argument(help="Input climbing video.", dir_okay=False, readable=True),
]


@dataclass(frozen=True)
class PipelineContext:
    config: AppConfig
    config_path: Path
    project_root: Path
    cache_root: Path


@app.command("preflight")
def preflight(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    """Validate tools, checkpoint, and the explicitly configured compute device."""
    try:
        context = _context(config_path)
        table = Table(title="Milestone 2 preflight")
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
        model = fingerprint_file(model_path)
        table.add_row("YOLO11x", f"{model_path}\nsha256 {model['sha256'][:16]}…")
        table.add_row("cache", str(context.cache_root))
        console.print(table)
    except (ClimbTrackError, OSError) as exc:
        _abort(exc)


@app.command("download-yolo")
def download_yolo(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    """Explicitly download the pinned YOLO11x checkpoint."""
    try:
        config = load_config(config_path)
        path, downloaded = ensure_yolo11_checkpoint(config, config_path)
        state = "downloaded" if downloaded else "already present"
        console.print(f"[green]YOLO11x {state}:[/green] {path}")
    except (ClimbTrackError, OSError) as exc:
        _abort(exc)


@app.command("ingest")
def ingest(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    force: ForceOption = False,
) -> None:
    """Run Stage 00 and print the immutable cache location."""
    try:
        result = _run_ingest(video, _context(config_path), force)
        _report("00_ingest", result)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("detect")
def detect(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    force: ForceOption = False,
) -> None:
    """Run ingest and Stage 10 YOLO11x person detection."""
    try:
        context = _context(config_path)
        ingest_result = _run_ingest(video, context, False)
        result = _run_detect(ingest_result, context, force)
        _report("10_detect", result)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("track")
def track(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    force: ForceOption = False,
) -> None:
    """Run prerequisites and Stage 20 ByteTrack association."""
    try:
        _, _, result, _ = _pipeline_to_tracks(video, config_path, force)
        _report("20_track", result)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("select")
def select(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Select one climber automatically, by ID, or by clicking a track."""
    try:
        context, ingest_result, tracks_result, _ = _pipeline_to_tracks(video, config_path, False)
        chosen = _resolve_manual_track(track_id, click, ingest_result, tracks_result)
        result = select_climber(
            ingest_result,
            tracks_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            manual_track_id=chosen,
            force=force,
        )
        _report_selection(result)
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("render-tracks")
def render_tracks(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    review_all: ReviewAllOption = False,
    force: ForceOption = False,
) -> None:
    """Render a VFR-aware MP4 with person boxes and persistent track IDs."""
    try:
        context, ingest_result, tracks_result, _ = _pipeline_to_tracks(video, config_path, False)
        if review_all and (track_id is not None or click):
            raise ClimbTrackError("--review-all cannot be combined with --track-id or --click")
        if review_all:
            selection = None
        else:
            chosen = _resolve_manual_track(track_id, click, ingest_result, tracks_result)
            selection = select_climber(
                ingest_result,
                tracks_result,
                config=context.config,
                cache_root=context.cache_root,
                project_root=context.project_root,
                manual_track_id=chosen,
            )
        result = render_tracking_overlay(
            ingest_result,
            tracks_result,
            selection,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        _report("50_render_tracks", result)
        console.print(f"[bold green]Video:[/bold green] {result.path / 'tracking_overlay.mp4'}")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("run-all")
def run_all(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Run every Milestone-2 stage in dependency order."""
    try:
        context, ingest_result, tracks_result, detections = _pipeline_to_tracks(
            video, config_path, force
        )
        _report("00_ingest", ingest_result)
        _report("10_detect", detections)
        _report("20_track", tracks_result)
        chosen = _resolve_manual_track(track_id, click, ingest_result, tracks_result)
        selection = select_climber(
            ingest_result,
            tracks_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            manual_track_id=chosen,
            force=force,
        )
        _report_selection(selection)
        rendered = render_tracking_overlay(
            ingest_result,
            tracks_result,
            selection,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        _report("50_render_tracks", rendered)
        console.print(
            f"[bold green]Milestone 2 complete:[/bold green] "
            f"{rendered.path / 'tracking_overlay.mp4'}"
        )
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("cache-list")
def cache_list(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    """List complete entries for every implemented stage."""
    try:
        context = _context(config_path)
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
        _abort(exc)


def _context(config_path: Path) -> PipelineContext:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    _seed(config)
    return PipelineContext(
        config=config,
        config_path=config_path,
        project_root=config_path.parent.parent,
        cache_root=resolve_cache_dir(config, config_path),
    )


def _run_ingest(video: Path, context: PipelineContext, force: bool) -> CacheResult:
    return ingest_video(
        video.expanduser(),
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def _run_detect(ingest_result: CacheResult, context: PipelineContext, force: bool) -> CacheResult:
    model_path = resolve_project_path(context.config.detection.model_path, context.config_path)
    if not model_path.is_file():
        raise ClimbTrackError(
            f"YOLO11x checkpoint is missing: {model_path}. Run 'climbtrack download-yolo'."
        )
    return detect_people(
        ingest_result,
        model_path=model_path,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def _pipeline_to_tracks(
    video: Path, config_path: Path, force: bool
) -> tuple[PipelineContext, CacheResult, CacheResult, CacheResult]:
    context = _context(config_path)
    ingest_result = _run_ingest(video, context, force)
    detections = _run_detect(ingest_result, context, force)
    tracks = track_people(
        ingest_result,
        detections,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )
    return context, ingest_result, tracks, detections


def _resolve_manual_track(
    track_id: int | None,
    click: bool,
    ingest_result: CacheResult,
    tracks_result: CacheResult,
) -> int | None:
    if track_id is not None and click:
        raise ClimbTrackError("Use either --track-id or --click, not both")
    if not click:
        return track_id
    frames = read_frame_index(ingest_result.path / "frames.parquet")
    tracks = read_tracks(tracks_result.path / "tracks.parquet")
    return choose_track_by_click(ingest_result.path, frames, tracks)


def _report(stage: str, result: CacheResult) -> None:
    state = "cache hit" if result.cache_hit else "created"
    console.print(f"[green]{stage} {state}:[/green] {result.path}")


def _report_selection(result: CacheResult) -> None:
    _report("25_select", result)
    payload = json.loads((result.path / "selection.json").read_text(encoding="utf-8"))
    console.print(
        f"[bold green]Selected track ID {payload['track_id']}[/bold green] "
        f"({payload['method']}, score {payload['score']:.3f})"
    )


def _seed(config: AppConfig) -> None:
    random.seed(config.project.seed)
    np.random.seed(config.project.seed)
    seed_torch(config.project.seed)


def _abort_selection(exc: SelectionUncertainError) -> None:
    console.print(f"[bold yellow]Selection needs confirmation:[/bold yellow] {exc}")
    table = Table(title="Ranked climber candidates")
    columns = ("track_id", "score", "observations", "continuity", "eligible")
    for name in columns:
        table.add_column(name)
    for candidate in exc.candidates[:10]:
        table.add_row(*(str(candidate[name]) for name in columns))
    console.print(table)
    console.print("Re-run with --track-id ID or --click; no automatic guess was written.")
    raise typer.Exit(code=3)


def _abort(exc: Exception) -> None:
    console.print(f"[bold red]Error:[/bold red] {exc}", highlight=False)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
