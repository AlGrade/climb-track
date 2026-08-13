"""Commands for move segmentation, per-move metrics, and the local move player."""

from pathlib import Path
from typing import Any

from climbtrack.cache import CacheResult
from climbtrack.cli.app import app
from climbtrack.cli.options import (
    DEFAULT_CONFIG,
    ClickOption,
    ConfigOption,
    ForceOption,
    OpenBrowserOption,
    PortOption,
    TrackIdOption,
    VideoArgument,
)
from climbtrack.cli.reporting import (
    abort,
    abort_selection,
    console,
    move_metrics_table,
    move_posture_table,
    report,
)
from climbtrack.config import resolve_annotation_dir
from climbtrack.errors import ClimbTrackError, SelectionUncertainError
from climbtrack.moves import prepare_move_session
from climbtrack.pipeline import (
    PipelineContext,
    run_move_metrics,
    run_moves,
    run_pose,
    run_refine,
    run_to_selection,
)
from climbtrack.player import create_player_server, run_player_server
from climbtrack.schema.frames import read_frame_index
from climbtrack.schema.move_metrics import (
    read_move_metrics_parquet,
    read_move_speed_timeline_parquet,
)
from climbtrack.schema.moves import read_moves_parquet
from climbtrack.stages.player_video import OUTPUT_NAME as PLAYER_VIDEO_NAME
from climbtrack.stages.player_video import prepare_player_video
from climbtrack.stages.render_pose import render_pose_overlay


@app.command("detect-moves")
def detect_moves_command(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Automatically segment hand moves from cached refined poses."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = run_pose(ingest_result, selection, context, False)
        refined = run_refine(selection, pose_result, context, False)
        result = run_moves(refined, context, force)
        report("70_moves", result)
        console.print(f"[bold green]Moves:[/bold green] {result.path / 'moves_auto.parquet'}")
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("measure-moves")
def measure_moves_command(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Calculate hand and body speeds for the current move annotations."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = run_pose(ingest_result, selection, context, False)
        refined = run_refine(selection, pose_result, context, False)
        automatic = run_moves(refined, context, False)
        session_path, _, _ = prepare_move_session(
            ingest_result,
            annotation_root=resolve_annotation_dir(context.config, context.config_path),
            automatic_moves=read_moves_parquet(automatic.path / "moves_auto.parquet"),
            automatic_moves_cache_key=automatic.manifest.cache_key,
        )
        result = run_move_metrics(refined, session_path.with_name("moves.parquet"), context, force)
        metrics = read_move_metrics_parquet(result.path / "move_metrics.parquet")
        report("80_move_metrics", result)
        console.print(move_metrics_table(metrics))
        console.print(move_posture_table(metrics))
        console.print(f"[bold green]Metrics:[/bold green] {result.path / 'move_metrics.parquet'}")
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except (ClimbTrackError, OSError) as exc:
        abort(exc)


@app.command("player")
def player(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    open_browser: OpenBrowserOption = True,
    port: PortOption = None,
) -> None:
    """Open automatically detected moves in the local Phase-2 player."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = run_pose(ingest_result, selection, context, False)
        refined = run_refine(selection, pose_result, context, False)
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
        abort_selection(exc)
    except (ClimbTrackError, OSError) as exc:
        abort(exc)


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
        result = run_moves(refined, context, False)
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
        result = run_move_metrics(refined, moves_path, context, False)
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
