"""``offlineai build`` - construct a bundle (sections 41 and 42)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from offlineai.bundler.builder import BundleBuilder, LockMode
from offlineai.bundler.results import BuildStep
from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.cli.progress import select_reporter
from offlineai.errors import InvalidPackageError
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
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-j",
            help="Concurrent artifact downloads. Defaults to the configured "
            "value (4), kept conservative so a build does not saturate storage "
            "or the network.",
            min=1,
            max=64,
        ),
    ] = None,
    locked: Annotated[
        bool,
        typer.Option(
            "--locked",
            help="Require offlineai.lock to describe this build exactly. Fails on "
            "any drift and never rewrites the lock. Use this in CI.",
        ),
    ] = False,
    update_lock: Annotated[
        bool,
        typer.Option(
            "--update-lock",
            help="Ignore the existing pins, re-resolve everything, and write the "
            "result. This is how you deliberately take a newer version.",
        ),
    ] = False,
    no_lock: Annotated[
        bool,
        typer.Option("--no-lock", help="Neither read nor write offlineai.lock."),
    ] = False,
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

    if sum((locked, update_lock, no_lock)) > 1:
        raise InvalidPackageError(
            "--locked, --update-lock and --no-lock are mutually exclusive",
            action="Pick one. Without any of them a build applies the existing "
            "pins and then refreshes the lock, which is what you usually want.",
        )
    lock_mode: LockMode = (
        "locked" if locked else "update" if update_lock else "none" if no_lock else "refresh"
    )

    reporter = select_reporter(
        json=output.fmt.json,
        quiet=output.fmt.quiet,
        is_terminal=output.console.is_terminal,
    )
    builder = BundleBuilder(
        context.settings,
        on_step=on_step,
        reporter=reporter,
        workers=workers,
        lock_mode=lock_mode,
    )

    output.line(f"Building package from {target}")
    output.line()
    with reporter:
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
