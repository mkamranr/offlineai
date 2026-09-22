"""``offlineai build`` - construct a bundle (sections 41 and 42)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from offlineai.bundler.builder import BundleBuilder
from offlineai.bundler.results import BuildStep
from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.schema.manifest import Compression


def register(app: typer.Typer) -> None:
    _register(app, "build", build)


def build(
    ctx: typer.Context,
    target: Annotated[
        Path,
        typer.Argument(
            help="Directory containing offlineai.yaml, or the file itself.",
            metavar="PATH",
        ),
    ] = Path(),
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Bundle file or directory to write to."),
    ] = None,
    compression: Annotated[
        Compression,
        typer.Option(
            "--compress",
            help="Archive compression. 'none' is the default because model weights "
            "and container layers are already compressed.",
        ),
    ] = Compression.NONE,
    sign_key: Annotated[
        Path | None,
        typer.Option(
            "--sign-key",
            help="Ed25519 private key. Embeds the signature in the bundle header, "
            "which is free at build time.",
        ),
    ] = None,
    signer: Annotated[
        str | None, typer.Option("--signer", help="Identity recorded in the signature.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate the definition and report what would be fetched, "
            "without fetching or writing anything.",
        ),
    ] = False,
) -> None:
    """Build an .offlineai bundle from a package definition.

    Examples:

      offlineai build .

      offlineai build examples/hello-ai --output ./dist/
    """
    context: Context = ctx.obj
    output = context.output

    def on_step(step: BuildStep) -> None:
        output.build_step(step.index, step.total, step.name, step.status, step.detail)

    builder = BundleBuilder(context.settings, on_step=on_step)

    output.line(f"Building package from {target}")
    output.line()
    result = builder.build(
        target,
        output=output_path,
        compression=compression,
        sign_key=sign_key,
        signer=signer,
        dry_run=dry_run,
    )
    if dry_run:
        output.emit(result)
        output.line()
        output.line("Dry run: the definition is valid. Nothing was written.")
        return
    output.build_result(result)
