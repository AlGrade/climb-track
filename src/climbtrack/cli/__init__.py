"""Command-line interface for independently resumable pipeline stages.

Importing the command modules is what registers them on the Typer app, so the
imports below are load-bearing even though nothing references the names. Their
order is the order commands appear in `climbtrack --help`, which is why it is
set by hand -- set up first, then the pipeline, then what you do with a result.
"""

# ruff: isort: off
from climbtrack.cli.app import app
from climbtrack.cli import maintenance  # noqa: F401
from climbtrack.cli import stages  # noqa: F401
from climbtrack.cli import moves  # noqa: F401
from climbtrack.cli import annotation  # noqa: F401

# ruff: isort: on

__all__ = ["app", "main"]


def main() -> None:
    """Entry point for `python -m climbtrack.cli` and the console script."""
    app()


if __name__ == "__main__":
    main()
