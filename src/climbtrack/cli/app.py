"""The Typer application every command module registers itself on.

Kept in its own module so that command modules can import the app without
importing the package initializer that imports them back.
"""

import typer

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Offline, reproducible person tracking for climbing videos.",
)
