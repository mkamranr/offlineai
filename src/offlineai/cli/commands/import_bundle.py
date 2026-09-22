"""``offlineai import`` - bring a bundle into the local registry (section 25)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from offlineai.bundler.verifier import verify_bundle
from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.registry.registry import Registry
from offlineai.utils.sizes import format_bytes


def register(app: typer.Typer) -> None:
    _register(app, "import", import_bundle)


def import_bundle(
    ctx: typer.Context,
    bundle: Annotated[Path, typer.Argument(help="Path to a .offlineai bundle.")],
    skip_verify: Annotated[
        bool,
        typer.Option(
            "--skip-verify",
            help="Do not verify checksums first. Not recommended: import streams "
            "artifacts into storage, and verification is what makes that safe.",
        ),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Re-import even if this version is present.")
    ] = False,
) -> None:
    """Verify a bundle and import it into the local registry.

    Artifacts stream straight into content-addressed storage rather than being
    extracted, so this needs room for the artifacts themselves and not for a
    second, temporary copy.
    """
    context: Context = ctx.obj
    output = context.output
    registry = Registry(context.settings)

    if not skip_verify:
        output.line(f"Verifying {bundle.name}")
        verify_bundle(bundle)
        output.line("Verification: OK", style="green")
        output.line()

    result = registry.import_bundle(bundle, force=force)

    if result.already_present:
        output.line(
            f"{result.package} {result.version} is already imported. Use --force to re-import."
        )
    else:
        output.line(f"Imported {result.package} {result.version}")
        output.line()
        output.line(f"  Artifacts:   {result.artifacts_imported}")
        output.line(f"  Stored:      {format_bytes(result.bytes_imported)}")
        if result.deduplicated:
            output.line(f"  Deduplicated: {result.deduplicated} artifact(s) already in storage")
        output.line()
        output.line(f"Install it with:\n  offlineai install {result.package}")

    output.emit_raw(
        {
            "package": result.package,
            "version": result.version,
            "artifacts_imported": result.artifacts_imported,
            "bytes_imported": result.bytes_imported,
            "deduplicated": result.deduplicated,
            "already_present": result.already_present,
        }
    )
