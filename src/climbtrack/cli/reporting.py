"""Terminal output shared by the command modules.

Exit codes are part of the interface: 2 for a failed command, 3 for an ambiguous
climber that needs a human decision. Scripts can tell the two apart.
"""

import json

import typer
from rich.console import Console
from rich.table import Table

from climbtrack.cache import CacheResult
from climbtrack.errors import SelectionUncertainError

console = Console()


def report(stage: str, result: CacheResult) -> None:
    """Print whether a stage was computed or served from cache, and where it landed."""
    state = "cache hit" if result.cache_hit else "created"
    console.print(f"[green]{stage} {state}:[/green] {result.path}")


def report_selection(result: CacheResult) -> None:
    """Print the selected climber together with how the choice was made."""
    report("25_select", result)
    payload = json.loads((result.path / "selection.json").read_text(encoding="utf-8"))
    console.print(
        f"[bold green]Selected track ID {payload['track_id']}[/bold green] "
        f"({payload['method']}, score {payload['score']:.3f})"
    )


def move_metrics_table(metrics: list[dict[str, object]]) -> Table:
    """Build the per-move speed table."""
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


def move_posture_table(metrics: list[dict[str, object]]) -> Table:
    """Build the per-move posture and coordination table."""
    table = Table(title="Per-move posture and coordination")
    for column in ("Move", "Result", "Hand settles", "Hip rise", "Hip below hand", "Torso lead"):
        table.add_column(column)
    for row in metrics:
        lag = row["coordination_lag_seconds"]
        correlation = row["coordination_correlation"]
        lead = (
            "undefined"
            if lag is None or correlation is None
            else f"{float(lag) * 1000:+.0f} ms (r={float(correlation):.2f})"
        )
        table.add_row(
            str(row["move_id"]),
            str(row["outcome"]),
            f"{float(row['hand_settle_offset_seconds']):.2f} s",
            f"{float(row['hip_rise_body_lengths']):+.2f} BL",
            f"{float(row['hip_below_hand_body_lengths']):.2f} BL",
            lead,
        )
    return table


def abort_selection(exc: SelectionUncertainError) -> None:
    """Show the ranked candidates and exit without guessing a climber."""
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


def abort(exc: Exception) -> None:
    """Print a failure without a traceback and exit."""
    console.print(f"[bold red]Error:[/bold red] {exc}", highlight=False)
    raise typer.Exit(code=2)
