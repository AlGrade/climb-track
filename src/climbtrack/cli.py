"""Command-line interface for independently resumable pipeline stages."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from climbtrack.annotation import compare_pose_session, evaluate_session, prepare_session
from climbtrack.annotation.tool import launch_annotation_tool
from climbtrack.cache import CacheResult
from climbtrack.cache.manifest import CacheManifest
from climbtrack.config import (
    AppConfig,
    load_config,
    resolve_annotation_dir,
    resolve_cache_dir,
    resolve_project_path,
)
from climbtrack.device import require_torch_device, seed_torch
from climbtrack.errors import ClimbTrackError, SelectionUncertainError
from climbtrack.model_downloads import (
    ensure_sapiens2_checkpoint,
    ensure_yolo11_checkpoint,
    verify_sapiens2_checkpoint,
    verify_yolo11_checkpoint,
)
from climbtrack.moves import prepare_move_session
from climbtrack.player import create_player_server, run_player_server
from climbtrack.provenance import executable_version
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.keypoints import read_registry
from climbtrack.schema.move_metrics import (
    read_move_metrics_parquet,
    read_move_speed_timeline_parquet,
)
from climbtrack.schema.moves import read_moves_parquet
from climbtrack.schema.tracks import read_tracks
from climbtrack.selection.click import choose_track_by_click
from climbtrack.stages.detect import detect_people
from climbtrack.stages.ingest import ingest_video
from climbtrack.stages.move_metrics import measure_moves as measure_move_metrics
from climbtrack.stages.moves import detect_moves
from climbtrack.stages.player_video import OUTPUT_NAME as PLAYER_VIDEO_NAME
from climbtrack.stages.player_video import prepare_player_video
from climbtrack.stages.pose import estimate_pose
from climbtrack.stages.refine import refine_pose
from climbtrack.stages.render_compare import render_pose_comparison
from climbtrack.stages.render_pose import render_pose_overlay
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
OpenBrowserOption = Annotated[
    bool,
    typer.Option(
        "--open-browser/--no-open-browser",
        help="Open the local move player in the default browser.",
    ),
]
PortOption = Annotated[
    int | None,
    typer.Option("--port", min=1024, max=65_535, help="Override the local player port."),
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


@app.command("download-sapiens")
def download_sapiens(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    """Explicitly download the pinned 6.08-GB Sapiens2-1B snapshot."""
    try:
        config = load_config(config_path)
        path, downloaded = ensure_sapiens2_checkpoint(config, config_path)
        state = "downloaded" if downloaded else "already present"
        console.print(f"[green]Sapiens2-1B {state}:[/green] {path}")
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


@app.command("pose")
def pose(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Run prerequisites and Stage 30 raw Sapiens2-1B inference."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        result = _run_pose(ingest_result, selection, context, force)
        _report("30_pose", result)
        console.print(f"[bold green]Raw poses:[/bold green] {result.path / 'pose_raw.parquet'}")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("refine")
def refine(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Run Stage 40 temporal repair using cached raw pose observations."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = _run_pose(ingest_result, selection, context, False)
        result = _run_refine(selection, pose_result, context, force)
        _report("40_refine", result)
        console.print(
            f"[bold green]Refined poses:[/bold green] {result.path / 'pose_refined.parquet'}"
        )
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("render-pose")
def render_pose(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Render the raw Sapiens2 skeleton over the source video."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = _run_pose(ingest_result, selection, context, False)
        result = render_pose_overlay(
            ingest_result,
            selection,
            pose_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        _report("50_render_pose", result)
        console.print(f"[bold green]Video:[/bold green] {result.path / 'skeleton_raw_overlay.mp4'}")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("detect-moves")
def detect_moves_command(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Automatically segment hand moves from cached refined poses."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = _run_pose(ingest_result, selection, context, False)
        refined = _run_refine(selection, pose_result, context, False)
        result = _run_moves(refined, context, force)
        _report("70_moves", result)
        console.print(f"[bold green]Moves:[/bold green] {result.path / 'moves_auto.parquet'}")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("measure-moves")
def measure_moves_command(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Calculate hand and body speeds for the current move annotations."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = _run_pose(ingest_result, selection, context, False)
        refined = _run_refine(selection, pose_result, context, False)
        automatic = _run_moves(refined, context, False)
        session_path, _, _ = prepare_move_session(
            ingest_result,
            annotation_root=resolve_annotation_dir(context.config, context.config_path),
            automatic_moves=read_moves_parquet(automatic.path / "moves_auto.parquet"),
            automatic_moves_cache_key=automatic.manifest.cache_key,
        )
        result = _run_move_metrics(refined, session_path.with_name("moves.parquet"), context, force)
        metrics = read_move_metrics_parquet(result.path / "move_metrics.parquet")
        _report("80_move_metrics", result)
        console.print(_move_metrics_table(metrics))
        console.print(f"[bold green]Metrics:[/bold green] {result.path / 'move_metrics.parquet'}")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except (ClimbTrackError, OSError) as exc:
        _abort(exc)


@app.command("render-comparison")
def render_comparison(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Render raw and refined skeletons side by side."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = _run_pose(ingest_result, selection, context, False)
        refined = _run_refine(selection, pose_result, context, False)
        result = render_pose_comparison(
            ingest_result,
            selection,
            pose_result,
            refined,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        _report("50_render_compare", result)
        console.print(f"[bold green]Video:[/bold green] {result.path / 'raw_vs_refined.mp4'}")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except ClimbTrackError as exc:
        _abort(exc)


@app.command("annotate")
def annotate(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
) -> None:
    """Review ten difficult frames in a small local ground-truth editor."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = _run_pose(ingest_result, selection, context, False)
        session_path, session, created = prepare_session(
            ingest_result,
            selection,
            pose_result,
            config=context.config,
            annotation_root=resolve_annotation_dir(context.config, context.config_path),
        )
        state = "created" if created else "resumed"
        console.print(f"[green]Annotation session {state}:[/green] {session_path}")
        console.print(
            "Drag wrong points, right-click invisible points, then press 'Bestätigen + weiter'."
        )
        registry = read_registry(pose_result.path / "keypoints.json")
        launch_annotation_tool(
            session,
            session_path,
            ingest_result.path,
            registry["skeleton_edges"],
        )
        reviewed = sum(frame.reviewed for frame in session.frames)
        console.print(f"[bold green]Reviewed:[/bold green] {reviewed}/{len(session.frames)} frames")
        if reviewed == len(session.frames):
            console.print(f"Next: climbtrack evaluate {session_path}")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except (ClimbTrackError, OSError) as exc:
        _abort(exc)


@app.command("player")
def player(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    open_browser: OpenBrowserOption = True,
    port: PortOption = None,
) -> None:
    """Open automatically detected moves in the local Phase-2 player."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = _run_pose(ingest_result, selection, context, False)
        refined = _run_refine(selection, pose_result, context, False)
        automatic, automatic_moves = _detect_moves_for_player(refined, context)
        skeleton = render_pose_overlay(
            ingest_result,
            selection,
            pose_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=False,
        )
        console.print("Preparing browser-optimized player video…")
        player_video = prepare_player_video(
            skeleton,
            ffmpeg_path=context.config.render.ffmpeg_path,
            cache_root=context.cache_root,
            project_root=context.project_root,
        )
        player_video_state = "reused" if player_video.cache_hit else "created"
        console.print(
            f"Browser video {player_video_state}: {player_video.path / PLAYER_VIDEO_NAME}"
        )
        session_path, session, created = prepare_move_session(
            ingest_result,
            annotation_root=resolve_annotation_dir(context.config, context.config_path),
            automatic_moves=automatic_moves,
            automatic_moves_cache_key=None if automatic is None else automatic.manifest.cache_key,
        )
        move_metrics, speed_timeline = _measure_moves_for_player(
            refined,
            session_path.with_name("moves.parquet"),
            context,
        )
        frames = read_frame_index(ingest_result.path / "frames.parquet")
        server = create_player_server(
            player_video.path / PLAYER_VIDEO_NAME,
            session_path,
            frames,
            context.config.move_player,
            move_metrics=move_metrics,
            speed_timeline=speed_timeline,
            port=port,
        )
        state = "created" if created else "resumed"
        console.print(f"[green]Move session {state}:[/green] {session_path}")
        console.print(
            f"[bold green]Automatic moves:[/bold green] {len(session.moves)} "
            + ("(manual correction is optional)" if session.moves else "(add moves manually)")
        )
        console.print(f"[bold green]Move metrics:[/bold green] {len(move_metrics)}")
        console.print(f"[bold green]Player:[/bold green] {server.url}")
        console.print("Press Ctrl+C in this terminal to stop the local player.")
        run_player_server(server, open_browser=open_browser)
        console.print("[green]Move player stopped.[/green]")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except (ClimbTrackError, OSError) as exc:
        _abort(exc)


@app.command("evaluate")
def evaluate(
    annotations: Annotated[
        Path,
        typer.Argument(help="ground_truth.json written by 'climbtrack annotate'.", dir_okay=False),
    ],
    config_path: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Measure raw pose accuracy against manually reviewed frames."""
    try:
        config = load_config(config_path.expanduser().resolve())
        output, metrics = evaluate_session(
            annotations.expanduser().resolve(),
            pck_threshold=config.annotation.pck_threshold,
            oks_sigma=config.annotation.oks_sigma,
            confidence_threshold=config.annotation.confidence_threshold,
        )
        table = Table(title=f"Ground truth: {metrics['reviewed_frames']} reviewed frames")
        for column in ("Group", "Points", "Mean px", "PCK@0.2", "OKS", "Corrected"):
            table.add_column(column)
        for group, values in metrics["groups"].items():
            table.add_row(
                group,
                str(values["keypoints"]),
                f"{values['mean_error_px']:.2f}",
                f"{values['pck']:.1%}",
                f"{values['oks']:.3f}",
                f"{values['corrected_rate']:.1%}",
            )
        console.print(table)
        console.print(f"[bold green]Metrics:[/bold green] {output}")
    except (ClimbTrackError, OSError) as exc:
        _abort(exc)


@app.command("evaluate-refined")
def evaluate_refined(
    video: VideoArgument,
    annotations: Annotated[
        Path,
        typer.Argument(help="Reviewed ground_truth.json.", dir_okay=False),
    ],
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
) -> None:
    """Compare raw and Stage-40 poses against the reviewed frames."""
    try:
        context, ingest_result, selection = _pipeline_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = _run_pose(ingest_result, selection, context, False)
        refined = _run_refine(selection, pose_result, context, False)
        output, metrics = compare_pose_session(
            annotations.expanduser().resolve(),
            refined.path / "pose_refined.parquet",
            pck_threshold=context.config.annotation.pck_threshold,
            oks_sigma=context.config.annotation.oks_sigma,
            confidence_threshold=context.config.annotation.confidence_threshold,
        )
        table = Table(title="Raw versus refined ground truth")
        for column in ("Group", "Raw px", "Refined px", "Raw PCK", "Refined PCK", "Missing"):
            table.add_column(column)
        for group in metrics["raw"]:
            raw = metrics["raw"][group]
            after = metrics["refined"][group]
            table.add_row(
                group,
                f"{raw['mean_error_px']:.2f}",
                f"{after['mean_error_px']:.2f}",
                f"{raw['pck']:.1%}",
                f"{after['pck']:.1%}",
                f"{after['prediction_missing_rate']:.1%}",
            )
        console.print(table)
        console.print(f"[bold green]Comparison:[/bold green] {output}")
    except SelectionUncertainError as exc:
        _abort_selection(exc)
    except (ClimbTrackError, OSError) as exc:
        _abort(exc)


@app.command("run-all")
def run_all(
    video: VideoArgument,
    config_path: ConfigOption = Path("configs/default.yaml"),
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Run every stage through Milestone 5 in dependency order."""
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
        pose_result = _run_pose(ingest_result, selection, context, force)
        _report("30_pose", pose_result)
        skeleton = render_pose_overlay(
            ingest_result,
            selection,
            pose_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        _report("50_render_pose", skeleton)
        refined = _run_refine(selection, pose_result, context, force)
        _report("40_refine", refined)
        comparison = render_pose_comparison(
            ingest_result,
            selection,
            pose_result,
            refined,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        _report("50_render_compare", comparison)
        console.print(
            f"[bold green]Milestone 5 complete:[/bold green] "
            f"{comparison.path / 'raw_vs_refined.mp4'}"
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


def _pipeline_to_selection(
    video: Path,
    config_path: Path,
    *,
    track_id: int | None,
    click: bool,
) -> tuple[PipelineContext, CacheResult, CacheResult]:
    context, ingest_result, tracks_result, _ = _pipeline_to_tracks(video, config_path, False)
    chosen = _resolve_manual_track(track_id, click, ingest_result, tracks_result)
    selection = select_climber(
        ingest_result,
        tracks_result,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        manual_track_id=chosen,
    )
    return context, ingest_result, selection


def _run_pose(
    ingest_result: CacheResult,
    selection: CacheResult,
    context: PipelineContext,
    force: bool,
) -> CacheResult:
    model_dir = resolve_project_path(context.config.models.sapiens2.model_dir, context.config_path)
    return estimate_pose(
        ingest_result,
        selection,
        model_dir=model_dir,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def _run_refine(
    selection: CacheResult,
    pose_result: CacheResult,
    context: PipelineContext,
    force: bool,
) -> CacheResult:
    return refine_pose(
        selection,
        pose_result,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def _run_moves(
    refined: CacheResult,
    context: PipelineContext,
    force: bool,
) -> CacheResult:
    return detect_moves(
        refined,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def _run_move_metrics(
    refined: CacheResult,
    moves_path: Path,
    context: PipelineContext,
    force: bool,
) -> CacheResult:
    return measure_move_metrics(
        refined,
        moves_path,
        config=context.config,
        cache_root=context.cache_root,
        project_root=context.project_root,
        force=force,
    )


def _detect_moves_for_player(
    refined: CacheResult,
    context: PipelineContext,
) -> tuple[CacheResult | None, list[dict[str, Any]]]:
    """Propose moves for the player, falling back to manual annotation.

    The player is the only place where boundaries can be corrected, so a video
    the detector refuses must not also lock the reviewer out of the correction
    tool. The dedicated 'detect-moves' command still fails loudly.
    """
    try:
        result = _run_moves(refined, context, False)
    except ClimbTrackError as exc:
        console.print(f"[bold yellow]Automatic move detection unavailable:[/bold yellow] {exc}")
        console.print("The player opens empty; add moves manually under 'Edit boundaries'.")
        return None, []
    return result, read_moves_parquet(result.path / "moves_auto.parquet")


def _measure_moves_for_player(
    refined: CacheResult,
    moves_path: Path,
    context: PipelineContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure the current move set, falling back to an empty speed view.

    Metrics only describe boundaries that the player exists to fix. An empty move
    set or a single segment with too few valid observations must therefore not
    stop the player from opening, or those boundaries could never be corrected.
    The dedicated 'measure-moves' command still fails loudly.
    """
    try:
        result = _run_move_metrics(refined, moves_path, context, False)
    except ClimbTrackError as exc:
        console.print(f"[bold yellow]Move metrics unavailable:[/bold yellow] {exc}")
        console.print(
            "Speeds stay empty until the boundaries are corrected and the player restarts."
        )
        return [], []
    return (
        read_move_metrics_parquet(result.path / "move_metrics.parquet"),
        read_move_speed_timeline_parquet(result.path / "move_speed_timeline.parquet"),
    )


def _move_metrics_table(metrics: list[dict[str, object]]) -> Table:
    table = Table(title="Per-move speed (relative to estimated body length)")
    for column in ("Move", "Result", "Hand max", "Hand mean", "Body max", "Body mean"):
        table.add_column(column)
    for row in metrics:
        table.add_row(
            str(row["move_id"]),
            str(row["outcome"]),
            f"{float(row['hand_max_speed_body_lengths_s']):.2f} BL/s",
            f"{float(row['hand_mean_speed_body_lengths_s']):.2f} BL/s",
            f"{float(row['body_max_speed_body_lengths_s']):.2f} BL/s",
            f"{float(row['body_mean_speed_body_lengths_s']):.2f} BL/s",
        )
    return table


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
