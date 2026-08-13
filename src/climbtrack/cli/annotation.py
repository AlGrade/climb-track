"""Commands for building ground truth and measuring accuracy against it."""

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from climbtrack.annotation import compare_pose_session, evaluate_session, prepare_session
from climbtrack.annotation.tool import launch_annotation_tool
from climbtrack.cli.app import app
from climbtrack.cli.options import (
    DEFAULT_CONFIG,
    ClickOption,
    ConfigOption,
    TrackIdOption,
    VideoArgument,
)
from climbtrack.cli.reporting import abort, abort_selection, console
from climbtrack.config import load_config, resolve_annotation_dir
from climbtrack.errors import ClimbTrackError, SelectionUncertainError
from climbtrack.pipeline import run_pose, run_refine, run_to_selection
from climbtrack.schema.keypoints import read_registry

GroundTruthArgument = Annotated[
    Path,
    typer.Argument(help="ground_truth.json written by 'climbtrack annotate'.", dir_okay=False),
]


@app.command("annotate")
def annotate(
    video: VideoArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
) -> None:
    """Review ten difficult frames in a small local ground-truth editor."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = run_pose(ingest_result, selection, context, False)
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
        abort_selection(exc)
    except (ClimbTrackError, OSError) as exc:
        abort(exc)


@app.command("evaluate")
def evaluate(
    annotations: GroundTruthArgument,
    config_path: ConfigOption = DEFAULT_CONFIG,
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
        abort(exc)


@app.command("evaluate-refined")
def evaluate_refined(
    video: VideoArgument,
    annotations: Annotated[
        Path,
        typer.Argument(help="Reviewed ground_truth.json.", dir_okay=False),
    ],
    config_path: ConfigOption = DEFAULT_CONFIG,
    track_id: TrackIdOption = None,
    click: ClickOption = False,
) -> None:
    """Compare raw and Stage-40 poses against the reviewed frames."""
    try:
        context, ingest_result, selection = run_to_selection(
            video, config_path, track_id=track_id, click=click
        )
        pose_result = run_pose(ingest_result, selection, context, False)
        refined = run_refine(selection, pose_result, context, False)
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
        abort_selection(exc)
    except (ClimbTrackError, OSError) as exc:
        abort(exc)
