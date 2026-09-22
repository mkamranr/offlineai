"""``offlineai graph``, ``offlineai diff`` and ``offlineai plugins``.

Sections 49, 58 and 59.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from offlineai.bundler.archive import BundleReader
from offlineai.bundler.differ import diff_manifests
from offlineai.bundler.graph import build_graph, to_ascii, to_dot
from offlineai.bundler.graph import to_json as graph_to_json
from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.plugins.loader import ENTRY_POINT_GROUP, load_sources
from offlineai.registry.registry import Registry
from offlineai.schema.manifest import Manifest
from offlineai.utils.sizes import format_bytes


class GraphFormat(StrEnum):
    ASCII = "ascii"
    JSON = "json"
    DOT = "dot"


def register(app: typer.Typer) -> None:
    _register(app, "graph", graph)
    _register(app, "diff", diff)
    _register(app, "plugins", plugins)


def _manifest_for(context: Context, target: str) -> Manifest:
    path = Path(target)
    if path.is_file():
        return BundleReader.peek_manifest(path)
    return Registry(context.settings).require(target).manifest()


def graph(
    ctx: typer.Context,
    target: Annotated[
        str, typer.Argument(help="A bundle path, or the name of an imported package.")
    ],
    output_format: Annotated[
        GraphFormat, typer.Option("--format", "-f", help="Output format.")
    ] = GraphFormat.ASCII,
) -> None:
    """Show what a package is made of, and where each piece came from.

    On an air-gapped host "where did this come from" cannot be looked up, so
    the bundle has to carry the answer.
    """
    context: Context = ctx.obj
    node = build_graph(_manifest_for(context, target))

    if output_format is GraphFormat.JSON or context.output.fmt.json:
        context.output.console.print_json(json.dumps(graph_to_json(node)))
        return
    if output_format is GraphFormat.DOT:
        context.output.console.print(to_dot(node), highlight=False, markup=False)
        return
    context.output.console.print(to_ascii(node), highlight=False, markup=False)


def diff(
    ctx: typer.Context,
    before: Annotated[Path, typer.Argument(help="The older bundle.")],
    after: Annotated[Path, typer.Argument(help="The newer bundle.")],
) -> None:
    """Compare two bundles.

    Reads only the manifests, so this is instant regardless of bundle size.

    Before promoting a new version into a secure environment, this is what
    turns "a new 81 GB file" into a reviewable statement.
    """
    context: Context = ctx.obj
    output = context.output

    result = diff_manifests(BundleReader.peek_manifest(before), BundleReader.peek_manifest(after))

    output.emit_raw(
        {
            "before": result.left,
            "after": result.right,
            "identical": result.identical,
            "added": [{"path": a.path, "size": a.size} for a in result.added],
            "removed": [{"path": a.path, "size": a.size} for a in result.removed],
            "changed": [
                {"path": c.path, "changes": c.what_changed, "size_delta": c.size_delta}
                for c in result.changed
            ],
            "requirements_changed": result.requirements_changed,
            "size_before": result.size_before,
            "size_after": result.size_after,
        }
    )

    output.line("Package Diff")
    output.line()
    output.line(f"  {result.left}  ->  {result.right}")

    if result.identical:
        output.line()
        output.line("No differences.", style="green")
        return

    if result.added:
        output.line()
        output.line("Added:", style="green")
        for artifact in result.added:
            output.line(f"  {_describe(artifact)}")
    if result.removed:
        output.line()
        output.line("Removed:", style="red")
        for artifact in result.removed:
            output.line(f"  {_describe(artifact)}")
    if result.changed:
        output.line()
        output.line("Changed:", style="yellow")
        for change in result.changed:
            output.line(f"  {change.path}")
            output.line(f"    {', '.join(change.what_changed)}", style="dim")
    if result.requirements_changed:
        output.line()
        output.line("Requirements:", style="yellow")
        for requirement_change in result.requirements_changed:
            output.line(f"  {requirement_change}")

    output.line()
    delta = result.size_delta
    sign = "+" if delta > 0 else ""
    output.line(
        f"Size:  {format_bytes(result.size_before)} -> "
        f"{format_bytes(result.size_after)}  ({sign}{format_bytes(abs(delta))})"
    )


def _describe(artifact: object) -> str:
    path = getattr(artifact, "path", "")
    size = getattr(artifact, "size", 0)
    metadata = getattr(artifact, "metadata", {}) or {}
    name = metadata.get("package") or metadata.get("image")
    version = metadata.get("version") or metadata.get("tag")
    label = f"{name} {version}" if name and version else str(path)
    return f"{label}  ({format_bytes(size)})"


def plugins(ctx: typer.Context) -> None:
    """List discovered artifact source plugins."""
    context: Context = ctx.obj
    loaded = load_sources()

    context.output.emit_raw(
        {
            "group": ENTRY_POINT_GROUP,
            "sources": loaded.names(),
            "failures": loaded.failures,
        }
    )
    context.output.line(f"Artifact sources  (entry-point group: {ENTRY_POINT_GROUP})")
    context.output.line()
    for name in loaded.names():
        context.output.line(f"  {name}")
    for name, reason in sorted(loaded.failures.items()):
        context.output.warn(f"{name} failed to load: {reason}")
    if not loaded.sources:
        context.output.line("  (none discovered)")
