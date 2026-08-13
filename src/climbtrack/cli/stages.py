"""Commands for the pose pipeline: one per stage, plus the full chain."""

from climbtrack.cli.app import app
from climbtrack.cli.options import (
    DEFAULT_CONFIG,
    ClickOption,
    ConfigOption,
    ForceOption,
    ReviewAllOption,
    TrackIdOption,
    VideoArgument,
)
from climbtrack.cli.reporting import abort, abort_selection, console, report, report_selection
from climbtrack.errors import ClimbTrackError, SelectionUncertainError
from climbtrack.pipeline import (
    build_context,
    resolve_manual_track,
    run_detect,
    run_ingest,
    run_pose,
    run_refine,
    run_to_selection,
    run_to_tracks,
)
from climbtrack.stages.render_compare import render_pose_comparison
from climbtrack.stages.render_pose import render_pose_overlay
from climbtrack.stages.render_tracks import render_tracking_overlay
from climbtrack.stages.select import select_climber


@app.command("ingest")
def ingest(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    force: ForceOption = False,
) -> None:
    """Run Stage 00 and print the immutable cache location."""
    try:
        result = run_ingest(video, build_context(config_path), force)
        report("00_ingest", result)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("detect")
def detect(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    force: ForceOption = False,
) -> None:
    """Run ingest and Stage 10 YOLO11x person detection."""
    try:
        context = build_context(config_path)
        ingest_result = run_ingest(video, context, False)
        result = run_detect(ingest_result, context, force)
        report("10_detect", result)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("track")
def track(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    force: ForceOption = False,
) -> None:
    """Run prerequisites and Stage 20 ByteTrack association."""
    try:
        _, _, result, _ = run_to_tracks(video, config_path, force)
        report("20_track", result)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("select")
def select(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Select one climber automatically, by ID, or by clicking a track."""
    try:
        context, ingest_result, tracks_result, _ = run_to_tracks(video, config_path, False)
        chosen = resolve_manual_track(track_id, click, ingest_result, tracks_result)
        result = select_climber(
            ingest_result,
            tracks_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            manual_track_id=chosen,
            force=force,
        )
        report_selection(result)
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("render-tracks")
def render_tracks(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    review_all: ReviewAllOption = False,
    force: ForceOption = False,
) -> None:
    """Render a VFR-aware MP4 with person boxes and persistent track IDs."""
    try:
        context, ingest_result, tracks_result, _ = run_to_tracks(video, config_path, False)
        if review_all and (track_id is not None or click):
            raise ClimbTrackError("--review-all cannot be combined with --track-id or --click")
        if review_all:
            selection = None
        else:
            chosen = resolve_manual_track(track_id, click, ingest_result, tracks_result)
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
        report("50_render_tracks", result)
        console.print(f"[bold green]Video:[/bold green] {result.path / 'tracking_overlay.mp4'}")
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("pose")
def pose(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Run prerequisites and Stage 30 raw Sapiens2-1B inference."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        result = run_pose(ingest_result, selection, context, force)
        report("30_pose", result)
        console.print(f"[bold green]Raw poses:[/bold green] {result.path / 'pose_raw.parquet'}")
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("refine")
def refine(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Run Stage 40 temporal repair using cached raw pose observations."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = run_pose(ingest_result, selection, context, False)
        result = run_refine(selection, pose_result, context, force)
        report("40_refine", result)
        console.print(
            f"[bold green]Refined poses:[/bold green] {result.path / 'pose_refined.parquet'}"
        )
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("render-pose")
def render_pose(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Render the raw Sapiens2 skeleton over the source video."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = run_pose(ingest_result, selection, context, False)
        result = render_pose_overlay(
            ingest_result,
            selection,
            pose_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        report("50_render_pose", result)
        console.print(f"[bold green]Video:[/bold green] {result.path / 'skeleton_raw_overlay.mp4'}")
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("render-comparison")
def render_comparison(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Render raw and refined skeletons side by side."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = run_pose(ingest_result, selection, context, False)
        refined = run_refine(selection, pose_result, context, False)
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
        report("50_render_compare", result)
        console.print(f"[bold green]Video:[/bold green] {result.path / 'raw_vs_refined.mp4'}")
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except ClimbTrackError as exc:
        abort(exc)


@app.command("run-all")
def run_all(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
    force: ForceOption = False,
) -> None:
    """Run every stage through Milestone 5 in dependency order."""
    try:
        context, ingest_result, tracks_result, detections = run_to_tracks(video, config_path, force)
        report("00_ingest", ingest_result)
        report("10_detect", detections)
        report("20_track", tracks_result)
        chosen = resolve_manual_track(track_id, click, ingest_result, tracks_result)
        selection = select_climber(
            ingest_result,
            tracks_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            manual_track_id=chosen,
            force=force,
        )
        report_selection(selection)
        rendered = render_tracking_overlay(
            ingest_result,
            tracks_result,
            selection,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        report("50_render_tracks", rendered)
        pose_result = run_pose(ingest_result, selection, context, force)
        report("30_pose", pose_result)
        skeleton = render_pose_overlay(
            ingest_result,
            selection,
            pose_result,
            config=context.config,
            cache_root=context.cache_root,
            project_root=context.project_root,
            force=force,
        )
        report("50_render_pose", skeleton)
        refined = run_refine(selection, pose_result, context, force)
        report("40_refine", refined)
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
        report("50_render_compare", comparison)
        console.print(
            f"[bold green]Milestone 5 complete:[/bold green] "
            f"{comparison.path / 'raw_vs_refined.mp4'}"
        )
    except SelectionUncertainError as exc:
        abort_selection(exc)
    except ClimbTrackError as exc:
        abort(exc)
