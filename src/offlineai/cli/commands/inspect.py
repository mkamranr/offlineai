"""``offlineai inspect`` - describe a bundle without installing it (section 60)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from offlineai.bundler.inspector import inspect_bundle
from offlineai.cli.main import Context
from offlineai.cli.main import register as _register


def register(app: typer.Typer) -> None:
    _register(app, "inspect", inspect)


def inspect(
    ctx: typer.Context,
    bundle: Annotated[Path, typer.Argument(help="Path to a .offlineai bundle.")],
) -> None:
    """Describe a bundle. Reads only its header, so this is instant at any size.

    Nothing is extracted and nothing is installed.
    """
    context: Context = ctx.obj
    context.output.inspect_result(inspect_bundle(bundle))
