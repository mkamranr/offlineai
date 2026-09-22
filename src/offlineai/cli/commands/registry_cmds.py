"""``list``, ``search``, ``info`` and ``remove`` (section 23)."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.registry.registry import PackageRecord, Registry
from offlineai.utils.sizes import format_bytes


def register(app: typer.Typer) -> None:
    _register(app, "list", list_packages)
    _register(app, "search", search)
    _register(app, "info", info)
    _register(app, "remove", remove)


def _table(records: list[PackageRecord]) -> Table:
    table = Table(box=None, pad_edge=False)
    for column in ("NAME", "VERSION", "PLATFORM", "SIZE", "IMPORTED"):
        table.add_column(column)
    for record in records:
        table.add_row(
            record.name,
            record.version,
            ", ".join(record.platforms) or "-",
            format_bytes(record.total_size),
            record.imported_at[:10],
        )
    return table


def _emit(context: Context, records: list[PackageRecord]) -> None:
    context.output.emit_raw(
        {
            "packages": [
                {
                    "name": r.name,
                    "version": r.version,
                    "platforms": r.platforms,
                    "size": r.total_size,
                    "imported_at": r.imported_at,
                    "signed": r.signed,
                }
                for r in records
            ]
        }
    )


def list_packages(ctx: typer.Context) -> None:
    """List packages in the local registry."""
    context: Context = ctx.obj
    records = Registry(context.settings).list_packages()
    _emit(context, records)
    if not records:
        context.output.line("No packages imported.")
        context.output.line()
        context.output.line("Import one with:\n  offlineai import <bundle>.offlineai")
        return
    if not context.output.fmt.json and not context.output.fmt.quiet:
        context.output.console.print(_table(records))


def search(
    ctx: typer.Context,
    term: Annotated[str, typer.Argument(help="Substring to match against names.")],
) -> None:
    """Search the local registry."""
    context: Context = ctx.obj
    records = Registry(context.settings).search(term)
    _emit(context, records)
    if not records:
        context.output.line(f"No packages matching {term!r}.")
        return
    if not context.output.fmt.json and not context.output.fmt.quiet:
        context.output.console.print(_table(records))


def info(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Package name.")],
    version: Annotated[str | None, typer.Option("--version", help="Specific version.")] = None,
) -> None:
    """Show detail about an imported package."""
    context: Context = ctx.obj
    registry = Registry(context.settings)
    record = registry.require(name, version)
    manifest = record.manifest()

    context.output.emit_raw(
        {
            "name": record.name,
            "version": record.version,
            "platforms": record.platforms,
            "size": record.total_size,
            "imported_at": record.imported_at,
            "signed": record.signed,
            "bundle_sha256": record.bundle_sha256,
            "artifacts": len(manifest.artifacts),
            "required_secrets": manifest.required_secrets,
        }
    )

    out = context.output
    out.line(f"Package:   {record.name}")
    out.line(f"Version:   {record.version}")
    out.line(f"Platforms: {', '.join(record.platforms) or '-'}")
    out.line(f"Size:      {format_bytes(record.total_size)}")
    out.line(f"Imported:  {record.imported_at}")
    out.line(f"Signed:    {'yes' if record.signed else 'no'}")
    out.line(f"Bundle:    sha256:{record.bundle_sha256}")
    out.line()
    out.line("Artifacts:")
    for artifact_type, size in sorted(manifest.size_by_type().items()):
        count = len(manifest.artifacts_of_type(artifact_type))
        out.line(f"  {artifact_type.value:<16}{format_bytes(size):>10}  ({count} file(s))")
    if manifest.required_secrets:
        out.line()
        out.line("Required secrets (supplied externally):")
        for secret in manifest.required_secrets:
            out.line(f"  {secret}: REQUIRED")


def remove(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Package name.")],
    version: Annotated[str | None, typer.Option("--version", help="Specific version.")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not prompt.")] = False,
) -> None:
    """Remove a package from the local registry.

    Only artifacts no other package still references are deleted.
    """
    context: Context = ctx.obj
    registry = Registry(context.settings)
    record = registry.require(name, version)

    if not yes and not context.output.fmt.json:
        confirmed = typer.confirm(
            f"Remove {record.identifier} ({format_bytes(record.total_size)})?"
        )
        if not confirmed:
            context.output.line("Cancelled.")
            return

    freed, freed_bytes = registry.remove(record.name, record.version)
    context.output.emit_raw(
        {
            "removed": record.identifier,
            "artifacts_freed": freed,
            "bytes_freed": freed_bytes,
        }
    )
    context.output.line(f"Removed {record.identifier}")
    context.output.line(
        f"Freed {freed} artifact(s), {format_bytes(freed_bytes)} (shared artifacts were kept)"
    )
