"""``offlineai verify`` - check bundle integrity (sections 11 and 61)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from offlineai.bundler.verifier import verify_bundle
from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.cli.progress import select_reporter


def register(app: typer.Typer) -> None:
    _register(app, "verify", verify)


def verify(
    ctx: typer.Context,
    bundle: Annotated[Path, typer.Argument(help="Path to a .offlineai bundle.")],
) -> None:
    """Verify every artifact in a bundle against its manifest.

    Streams the whole archive, so the time taken scales with bundle size. Use
    'offlineai inspect' for a description without the integrity check.

    Exits 3 if verification fails.
    """
    context: Context = ctx.obj
    reporter = select_reporter(
        json=context.output.fmt.json,
        quiet=context.output.fmt.quiet,
        is_terminal=context.output.console.is_terminal,
    )
    # Streaming a 62 GB bundle takes minutes; silence for that long reads as
    # a hang.
    with reporter:
        result = verify_bundle(bundle, reporter=reporter)
    context.output.verify_result(result)
