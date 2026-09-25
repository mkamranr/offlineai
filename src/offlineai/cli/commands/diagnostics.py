"""``offlineai check`` and ``offlineai doctor`` (sections 21, 22 and 31)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from offlineai.bundler.archive import BundleReader
from offlineai.bundler.results import CheckResult, CheckStatus
from offlineai.cli.main import Context
from offlineai.cli.main import register as _register
from offlineai.hardware.compat import (
    COMPATIBILITY_CAVEAT,
    check_compatibility,
    estimate_storage,
)
from offlineai.hardware.detector import HardwareDetector
from offlineai.registry.registry import Registry
from offlineai.runtime.docker import DockerRuntime
from offlineai.schema.profile import Profile, load_profile
from offlineai.utils.fs import free_space
from offlineai.utils.sizes import format_bytes

_STYLE = {
    CheckStatus.OK: "green",
    CheckStatus.FAILED: "bold red",
    CheckStatus.WARNING: "yellow",
    CheckStatus.SKIPPED: "dim",
}


def register(app: typer.Typer) -> None:
    _register(app, "check", check)
    _register(app, "doctor", doctor)


def _print_checks(context: Context, checks: list[CheckResult]) -> None:
    """Render the check table.

    Goes through `output.line`, not `console.print`. The console is ungated,
    so printing through it directly emitted these rows *after* the JSON
    payload and made `offlineai --json doctor | jq` fail.
    """
    if not checks:
        return
    width = max(len(c.name) for c in checks) + 2
    for item in checks:
        detail = f"  {item.detail}" if item.detail else ""
        context.output.line(
            f"{(item.name + ':').ljust(width)}{item.status.value}{detail}",
            style=_STYLE[item.status],
        )


def check(
    ctx: typer.Context,
    target: Annotated[
        str, typer.Argument(help="A bundle path, or the name of an imported package.")
    ],
    profile_path: Annotated[
        Path | None,
        typer.Option(
            "--profile",
            "-p",
            help="Check against a described target environment instead of this "
            "machine. Capture one with 'offlineai doctor --save-profile'.",
        ),
    ] = None,
) -> None:
    """Check whether a host satisfies a bundle's requirements.

    Reads only the bundle header, so this is instant at any size and installs
    nothing.

    Without --profile this checks the machine you are on. On a builder that is
    the wrong machine: the target is air-gapped and elsewhere. Pass a profile
    to ask the question that matters.

      offlineai check qwen-vllm-1.0.0.offlineai --profile h100-server.yaml
    """
    context: Context = ctx.obj
    output = context.output

    path = Path(target)
    if path.is_file():
        manifest = BundleReader.peek_manifest(path)
        bundle_bytes = path.stat().st_size
    else:
        record = Registry(context.settings).require(target)
        manifest = record.manifest()
        bundle_bytes = record.total_size

    profile = load_profile(profile_path) if profile_path else None

    if profile is not None:
        hardware = profile.to_hardware_report()
        runtime = profile.to_runtime_availability()
        # A profile describes a class of machine, not how much space it has
        # free today, so there is nothing to check the estimate against. It is
        # still computed and reported, because the operator needs to know what
        # the bundle will cost on the target.
        storage = estimate_storage(manifest, bundle_bytes=bundle_bytes, available_bytes=0)
    else:
        hardware = HardwareDetector(disk_path=context.settings.data_dir).detect()
        runtime = DockerRuntime().availability()
        storage = estimate_storage(
            manifest,
            bundle_bytes=bundle_bytes,
            available_bytes=free_space(context.settings.data_dir),
        )

    report = check_compatibility(
        manifest,
        hardware,
        runtime=runtime,
        storage=None if profile is not None else storage,
    )

    output.emit_raw(
        {
            "package": manifest.package.name,
            "version": manifest.package.version,
            "compatible": report.compatible,
            "verdict": report.verdict,
            "checks": [c.model_dump(mode="json") for c in report.checks],
            "warnings": report.warnings,
            "profile": profile.name if profile else None,
            "evaluated_against": "profile" if profile else "this host",
            "storage": {
                "artifact_storage": storage.artifact_storage_bytes,
                "container_storage": storage.container_storage_bytes,
                "model_storage": storage.model_storage_bytes,
                "temporary": storage.temporary_bytes,
                "recommended_free": storage.recommended_bytes,
                "available": storage.available_bytes,
                "sufficient": storage.sufficient,
            },
        }
    )

    output.line("Hardware Compatibility")
    output.line()
    output.line(f"Package: {manifest.package.name} {manifest.package.version}")
    if profile is not None:
        # Unmistakable: a profile result mistaken for a real one is the
        # failure mode that would actually hurt someone.
        output.line()
        output.line(
            f"Evaluated against profile: {profile.name}   (NOT this host)",
            style="bold yellow",
        )
    output.line()
    _print_checks(context, report.checks)

    output.line()
    output.line("Installation Storage Estimate")
    output.line()
    for label, value in (
        ("Bundle", storage.bundle_bytes),
        ("Artifact storage", storage.artifact_storage_bytes),
        ("Container storage", storage.container_storage_bytes),
        ("Model storage", storage.model_storage_bytes),
        ("Temporary", storage.temporary_bytes),
    ):
        output.line(f"  {label:<20}{format_bytes(value):>12}")
    output.line("  " + "-" * 32)
    output.line(f"  {'Recommended free':<20}{format_bytes(storage.recommended_bytes):>12}")
    if profile is None:
        output.line(f"  {'Available':<20}{format_bytes(storage.available_bytes):>12}")
        if not storage.sufficient:
            output.line()
            output.warn(f"Insufficient disk space: {format_bytes(storage.shortfall_bytes)} short.")
    else:
        # A profile describes a class of machine, not how much space it has
        # free today. Printing "0 B available" and a shortfall would be a
        # fabricated failure, which is worse than saying nothing.
        output.line()
        output.line(
            "  A profile does not state free space, so this is what the target will\n"
            "  need rather than a check against what it has.",
            style="dim",
        )
    output.line()
    output.line(
        "  Import streams into content-addressed storage rather than extracting,",
        style="dim",
    )
    output.line("  so no separate extraction space is required.", style="dim")

    for warning in report.warnings:
        output.warn(warning)

    output.line()
    output.line(
        f"Result: {report.verdict}",
        style="bold green" if report.compatible else "bold red",
    )
    if report.compatible:
        output.line()
        if profile is not None:
            # A profile adds a second layer of "we have not actually looked":
            # nothing here verifies the target matches its own description.
            output.line(
                "Requirements satisfied against the profile. This does not verify "
                "that the\ntarget machine matches the profile, and runtime success "
                "is not guaranteed.",
                style="dim",
            )
        else:
            output.line(COMPATIBILITY_CAVEAT, style="dim")
    else:
        raise typer.Exit(4)


def doctor(
    ctx: typer.Context,
    save_profile: Annotated[
        Path | None,
        typer.Option(
            "--save-profile",
            help="Write this host's description to a file, for use with "
            "'offlineai check --profile' on a builder elsewhere.",
        ),
    ] = None,
    profile_name: Annotated[
        str | None,
        typer.Option("--profile-name", help="Name recorded in the saved profile."),
    ] = None,
) -> None:
    """Check that this host is ready to install and run packages.

    Bundle-independent: this reports on the environment, not on any particular
    workload.

    With --save-profile it also writes a description of this machine. Run it on
    the air-gapped target, carry the file back, and a builder can then check
    bundles against measured ground truth rather than a hand-written guess:

      offlineai doctor --save-profile h100-server.yaml
      offlineai check <bundle> --profile h100-server.yaml
    """
    context: Context = ctx.obj
    output = context.output

    hardware = HardwareDetector(disk_path=context.settings.data_dir).detect()
    runtime = DockerRuntime().availability()
    checks: list[CheckResult] = []
    warnings: list[str] = []

    checks.append(
        CheckResult(
            name="OS",
            status=CheckStatus.OK if hardware.is_linux else CheckStatus.WARNING,
            detail=f"{hardware.os_name} {hardware.os_version}"
            + ("" if hardware.is_linux else "  (not a supported target platform)"),
        )
    )
    if not hardware.is_linux:
        warnings.append(
            f"{hardware.os_name} is supported for building and for development "
            "only. Deploy onto Linux."
        )

    checks.append(
        CheckResult(
            name="Architecture",
            status=CheckStatus.OK,
            detail=f"{hardware.architecture} ({hardware.cpu.cores} cores)",
        )
    )

    checks.append(
        CheckResult(
            name="Memory",
            status=CheckStatus.OK if hardware.memory else CheckStatus.SKIPPED,
            detail=(
                f"{hardware.memory.total_gb:.1f} GB"
                if hardware.memory
                else "could not be determined on this host"
            ),
        )
    )

    if hardware.disk is None:
        checks.append(CheckResult(name="Disk", status=CheckStatus.SKIPPED, detail="not readable"))
    else:
        low = hardware.disk.free_bytes < 10 * 1000**3
        checks.append(
            CheckResult(
                name="Disk",
                status=CheckStatus.WARNING if low else CheckStatus.OK,
                detail=f"{format_bytes(hardware.disk.free_bytes)} free at {hardware.disk.path}",
            )
        )
        if low:
            warnings.append(
                "Less than 10 GB free. AI bundles are routinely far larger than "
                "that; consider --data-dir on a bigger volume."
            )

    checks.append(
        CheckResult(
            name="Container runtime",
            status=CheckStatus.OK if runtime.available else CheckStatus.FAILED,
            detail=(
                f"docker {runtime.version}"
                if runtime.available
                else (runtime.detail or "unavailable")
            ),
        )
    )

    checks.append(
        CheckResult(
            name="GPU runtime",
            status=CheckStatus.OK if runtime.gpu_support else CheckStatus.SKIPPED,
            detail=(
                "nvidia container runtime present"
                if runtime.gpu_support
                else "no GPU runtime registered with the container engine"
            ),
        )
    )

    if hardware.gpus:
        checks.append(
            CheckResult(
                name="GPU",
                status=CheckStatus.OK,
                detail=f"{len(hardware.gpus)} device(s), driver "
                f"{hardware.nvidia_driver or 'unknown'}",
            )
        )
    else:
        checks.append(
            CheckResult(
                name="GPU",
                status=CheckStatus.SKIPPED,
                detail=hardware.gpu_detail or "none detected",
            )
        )

    writable = _writable(context.settings.data_dir)
    checks.append(
        CheckResult(
            name="Permissions",
            status=CheckStatus.OK if writable else CheckStatus.FAILED,
            detail=f"{context.settings.data_dir} "
            + ("is writable" if writable else "is NOT writable"),
        )
    )

    # SQLite is in the standard library, so the only thing that can go wrong
    # is the location, which the permissions check above already covers.
    checks.append(
        CheckResult(
            name="Registry",
            status=CheckStatus.OK,
            detail=str(context.settings.registry_dir),
        )
    )

    failed = [c for c in checks if c.status is CheckStatus.FAILED]
    overall = "NOT READY" if failed else ("READY" if not warnings else "READY (with warnings)")

    output.emit_raw(
        {
            "ready": not failed,
            "overall": overall,
            "checks": [c.model_dump(mode="json") for c in checks],
            "warnings": warnings,
            "saved_profile": str(save_profile) if save_profile else None,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_gb": round(g.memory_gb, 1),
                    "driver": g.driver_version,
                    "compute_capability": g.compute_capability,
                }
                for g in hardware.gpus
            ],
        }
    )

    output.line("OfflineAI Doctor")
    output.line()
    _print_checks(context, checks)

    if hardware.gpus:
        output.line()
        output.line("GPU Environment")
        output.line()
        for device in hardware.gpus:
            output.line(f"GPU {device.index}:")
            output.line(f"  {device.name}")
            if device.driver_version:
                output.line(f"  Driver: {device.driver_version}")
            output.line(f"  VRAM: {device.memory_gb:.0f} GB")
            if device.compute_capability:
                output.line(f"  Compute capability: {device.compute_capability}")
        output.line()
        output.line(f"Detected: {len(hardware.gpus)} GPU(s)")

    if warnings:
        output.line()
        for warning in warnings:
            output.warn(warning)

    if save_profile is not None:
        import socket

        name = profile_name or socket.gethostname() or "captured-host"
        profile = Profile.from_hardware_report(hardware, name=name)
        save_profile.parent.mkdir(parents=True, exist_ok=True)
        save_profile.write_text(profile.to_yaml())
        output.line()
        output.line(f"Profile written: {save_profile}  (name: {name})")
        output.line(
            "Carry it to a builder and use it with:\n"
            f"  offlineai check <bundle> --profile {save_profile.name}"
        )

    output.line()
    output.line(f"Overall: {overall}", style="bold red" if failed else "bold green")
    if failed:
        raise typer.Exit(4)


def _writable(path: Path) -> bool:
    import os

    current = path.absolute()
    while not current.exists():
        parent = current.parent
        if parent == current:
            return False
        current = parent
    return os.access(current, os.W_OK)
