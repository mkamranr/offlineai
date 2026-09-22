"""``install``, ``start``, ``stop``, ``restart``, ``status``, ``logs``, ``rollback``.

Sections 26 to 28.
"""

from __future__ import annotations

from typing import Annotated

import typer
import yaml

from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.errors import InstallationError
from offlineai.installer.installer import Installer, InstallResult
from offlineai.installer.rollback import rollback_installation
from offlineai.registry.registry import PackageRecord, Registry
from offlineai.runtime.base import ContainerRuntime
from offlineai.runtime.docker import DockerRuntime
from offlineai.runtime.manager import RuntimeManager, image_overrides_from
from offlineai.schema.package import Package


def register(app: typer.Typer) -> None:
    _register(app, "install", install)
    _register(app, "start", start)
    _register(app, "stop", stop)
    _register(app, "restart", restart)
    _register(app, "status", status)
    _register(app, "logs", logs)
    _register(app, "rollback", rollback)


def _runtime(context: Context) -> ContainerRuntime:
    """Select the container engine for this invocation.

    Docker is the only implementation today; the setting exists so Podman or
    containerd can be selected without touching every command.
    """
    engine = context.settings.runtime.container_engine
    if engine == "docker":
        return DockerRuntime()
    raise InstallationError(
        f"unsupported container engine {engine!r}",
        action="Set runtime.container_engine to 'docker' in config.yaml.",
    )


def _load(context: Context, name: str, version: str | None) -> tuple[PackageRecord, Package]:
    record = Registry(context.settings).require(name, version)
    return record, Package.model_validate(yaml.safe_load(record.package_yaml))


def _report_install(context: Context, result: InstallResult) -> None:
    out = context.output
    out.emit_raw(
        {
            "package": result.package,
            "version": result.version,
            "installation_id": result.installation_id,
            "state": result.state.value,
            "services": result.services_started,
            "endpoints": result.endpoints,
            "healthy": result.healthy,
            "dev_mode": result.dev_mode,
            "warnings": result.warnings,
            "checks": [c.model_dump(mode="json") for c in result.checks],
        }
    )
    width = max((len(c.name) for c in result.checks), default=0) + 2
    for check in result.checks:
        detail = f"  {check.detail}" if check.detail else ""
        out.line(f"{(check.name + ':').ljust(width)}{check.status.value}{detail}")

    out.line()
    out.line(f"Installation: {result.installation_id}")
    out.line(f"State:        {result.state.value}")
    if result.services_started:
        out.line(f"Services:     {', '.join(result.services_started)}")
    if result.endpoints:
        out.line()
        out.line("Endpoints:")
        for endpoint in result.endpoints:
            out.line(f"  {endpoint}")
    for warning in result.warnings:
        out.warn(warning)
    if result.dev_mode:
        out.warn("Development mode: this is not a supported deployment platform.")
    out.line()
    out.line("Requirements satisfied. Runtime success is not guaranteed.", style="dim")


def install(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Package name, or a path to a bundle.")],
    version: Annotated[str | None, typer.Option("--version", help="Specific version.")] = None,
    no_start: Annotated[
        bool, typer.Option("--no-start", help="Install without starting services.")
    ] = False,
    gpus: Annotated[
        str | None,
        typer.Option("--gpus", help="Comma-separated GPU device ids, e.g. --gpus 0,1"),
    ] = None,
) -> None:
    """Install an imported package, then start and health-check it.

    Pass a bundle path to import and install in one step.
    """
    context: Context = ctx.obj
    registry = Registry(context.settings)

    from pathlib import Path

    candidate = Path(name)
    if candidate.is_file() and candidate.suffix == ".offlineai":
        context.output.line(f"Importing {candidate.name}")
        from offlineai.bundler.verifier import verify_bundle

        verify_bundle(candidate)
        imported = registry.import_bundle(candidate)
        name, version = imported.package, imported.version
        context.output.line()

    installer = Installer(context.settings, registry, _runtime(context))
    result = installer.install(
        name,
        version=version,
        start=not no_start,
        gpu_device_ids=gpus.split(",") if gpus else None,
    )
    _report_install(context, result)


def start(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Package name.")],
    version: Annotated[str | None, typer.Option("--version")] = None,
) -> None:
    """Start an installed package's services."""
    context: Context = ctx.obj
    record, package = _load(context, name, version)
    manager = RuntimeManager(_runtime(context))
    model_root = context.settings.data_dir / "installed" / record.name / "models"
    started = manager.start(
        package,
        model_root=model_root if model_root.is_dir() else None,
        image_overrides=image_overrides_from(record.manifest()),
    )
    context.output.emit_raw({"package": record.name, "started": started})
    context.output.line(f"Started {len(started)} service(s) for {record.name}")
    for service in started:
        context.output.line(f"  {service}")


def stop(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Package name.")],
    version: Annotated[str | None, typer.Option("--version")] = None,
    remove_containers: Annotated[
        bool, typer.Option("--remove", help="Remove containers as well as stopping them.")
    ] = False,
) -> None:
    """Stop an installed package's services."""
    context: Context = ctx.obj
    record, package = _load(context, name, version)
    stopped = RuntimeManager(_runtime(context)).stop(package, remove=remove_containers)
    context.output.emit_raw({"package": record.name, "stopped": stopped})
    context.output.line(f"Stopped {len(stopped)} service(s) for {record.name}")


def restart(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Package name.")],
    version: Annotated[str | None, typer.Option("--version")] = None,
) -> None:
    """Restart an installed package's services."""
    context: Context = ctx.obj
    record, package = _load(context, name, version)
    manager = RuntimeManager(_runtime(context))
    model_root = context.settings.data_dir / "installed" / record.name / "models"
    started = manager.restart(package, model_root=model_root if model_root.is_dir() else None)
    context.output.emit_raw({"package": record.name, "started": started})
    context.output.line(f"Restarted {record.name}")


def status(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Package name. Omit for all.")] = None,
    version: Annotated[str | None, typer.Option("--version")] = None,
) -> None:
    """Show the runtime status of an installed package."""
    context: Context = ctx.obj
    registry = Registry(context.settings)
    manager = RuntimeManager(_runtime(context))

    names = [name] if name else [r.name for r in registry.list_packages()]
    if not names:
        context.output.line("No packages imported.")
        context.output.emit_raw({"packages": []})
        return

    payloads = []
    for index, package_name in enumerate(names):
        record, package = _load(context, package_name, version if name else None)
        report = manager.status(package)
        payloads.append(
            {
                "package": report.package,
                "version": report.version,
                "status": report.status,
                "services": {s.name: s.status.value for s in report.services},
                "endpoints": report.endpoints,
            }
        )
        if index:
            context.output.line()
        context.output.line(f"Package: {report.package}")
        context.output.line()
        context.output.line(f"Status: {report.status}")
        if report.services:
            context.output.line()
            context.output.line("Services:")
            width = max(len(s.name) for s in report.services) + 2
            for service in report.services:
                context.output.line(f"  {service.name.ljust(width)}{service.status.value}")
        if report.endpoints:
            context.output.line()
            context.output.line("Endpoint:")
            for endpoint in report.endpoints:
                context.output.line(f"  {endpoint}")

    context.output.emit_raw(payloads[0] if name else {"packages": payloads})


def logs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Package name.")],
    service: Annotated[str | None, typer.Option("--service", help="Limit to one service.")] = None,
    tail: Annotated[
        int | None, typer.Option("--tail", "-n", help="Show only the last N lines.")
    ] = None,
    version: Annotated[str | None, typer.Option("--version")] = None,
) -> None:
    """Show logs for an installed package."""
    context: Context = ctx.obj
    _, package = _load(context, name, version)
    text = RuntimeManager(_runtime(context)).logs(package, service=service, tail=tail)
    context.output.emit_raw({"package": name, "logs": text})
    if not context.output.fmt.json:
        context.output.console.print(text, end="", highlight=False, markup=False)


def rollback(
    ctx: typer.Context,
    installation_id: Annotated[
        str, typer.Argument(help="Installation id, e.g. install-20260922-001.")
    ],
    remove_images: Annotated[
        bool,
        typer.Option("--remove-images", help="Also remove images this installation loaded."),
    ] = False,
) -> None:
    """Undo an installation.

    Replays each step's recorded inverse in reverse order. User-owned data is
    never deleted.
    """
    context: Context = ctx.obj
    registry = Registry(context.settings)
    result = rollback_installation(
        context.settings,
        registry,
        _runtime(context),
        installation_id,
        keep_images=not remove_images,
    )
    context.output.emit_raw(
        {
            "installation_id": result.installation_id,
            "package": result.package,
            "undone": result.undone,
            "failed": result.failed,
            "skipped": result.skipped,
            "complete": result.complete,
        }
    )
    context.output.line(f"Rolled back {result.installation_id} ({result.package})")
    for action in result.undone:
        context.output.line(f"  undone   {action}")
    for action in result.skipped:
        context.output.line(f"  skipped  {action}")
    for failure in result.failed:
        context.output.warn(f"could not undo: {failure}")
    if not result.complete:
        context.output.line()
        context.output.line("Rollback was incomplete. Review the warnings above.")
