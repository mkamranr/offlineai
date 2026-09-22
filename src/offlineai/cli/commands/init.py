"""``offlineai init`` - create the data directory tree (section 7.1)."""

from __future__ import annotations

from typing import Annotated

import typer

from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.config.settings import default_config_yaml


def register(app: typer.Typer) -> None:
    _register(app, "init", init)


def init(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing config.yaml.")
    ] = False,
) -> None:
    """Create ~/.offlineai (or the configured data directory) and a starter config."""
    context: Context = ctx.obj
    settings = context.settings
    output = context.output

    settings.ensure_directories()

    config_file = settings.config_file
    created = False
    if force or not config_file.exists():
        config_file.write_text(default_config_yaml())
        created = True

    output.emit_raw(
        {
            "home": str(settings.home),
            "data_dir": str(settings.data_dir),
            "cache_dir": str(settings.cache_dir),
            "registry_dir": str(settings.registry_dir),
            "config_file": str(config_file),
            "config_created": created,
        }
    )
    output.line(f"Initialised OfflineAI in {settings.home}")
    output.line()
    for label, path in (
        ("config", config_file),
        ("registry", settings.registry_dir),
        ("cache", settings.cache_dir),
        ("bundles", settings.bundles_dir),
        ("logs", settings.logs_dir),
    ):
        output.line(f"  {label:<10} {path}")
    if not created:
        output.line()
        output.line("Existing config.yaml left unchanged (use --force to replace it).")
