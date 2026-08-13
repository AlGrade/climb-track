"""Option and argument types shared by the command modules."""

from pathlib import Path
from typing import Annotated

import typer

DEFAULT_CONFIG = Path("configs/default.yaml")

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
