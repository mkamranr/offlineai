"""Hardware compatibility and storage planning (sections 21 and 22).

Two rules from the specification shape the output.

Section 21: *the tool must not automatically claim that a workload will work
merely because the minimum hardware requirement is met.* So a passing report
says "Requirements satisfied. Runtime success is not guaranteed." and nothing
stronger.

Also section 21, implicitly: a requirement that could not be evaluated is
``SKIPPED`` with a reason, never ``PASS``. "We could not look" and "it is fine"
are different answers and must stay different all the way to the terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from offlineai.bundler.results import CheckResult, CheckStatus
from offlineai.hardware.detector import HardwareReport, parse_driver_version
from offlineai.runtime.base import RuntimeAvailability
from offlineai.schema.manifest import ArtifactType, Manifest
from offlineai.utils.sizes import format_bytes

__all__ = [
    "COMPATIBILITY_CAVEAT",
    "CompatibilityReport",
    "StorageEstimate",
    "check_compatibility",
    "estimate_storage",
]

#: Printed whenever requirements are met. Section 21 asks for exactly this
#: distinction between "permitted to proceed" and "guaranteed to work".
COMPATIBILITY_CAVEAT = "Requirements satisfied. Runtime success is not guaranteed."


@dataclass(slots=True)
class CompatibilityReport:
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        return not any(c.status is CheckStatus.FAILED for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAILED]

    @property
    def verdict(self) -> str:
        return "COMPATIBLE" if self.compatible else "INCOMPATIBLE"


@dataclass(slots=True)
class StorageEstimate:
    """Space needed to install, broken down (section 22).

    The specification's example adds a full extraction pass to the total.
    OfflineAI does not extract - import streams bundle members straight into
    content-addressed storage - so that line is genuinely zero here, and
    saying so is more useful than padding the number to match the example.
    """

    bundle_bytes: int
    artifact_storage_bytes: int
    container_storage_bytes: int
    model_storage_bytes: int
    temporary_bytes: int
    available_bytes: int

    @property
    def recommended_bytes(self) -> int:
        return (
            self.artifact_storage_bytes
            + self.container_storage_bytes
            + self.model_storage_bytes
            + self.temporary_bytes
        )

    @property
    def sufficient(self) -> bool:
        return self.available_bytes >= self.recommended_bytes

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.recommended_bytes - self.available_bytes)


def check_compatibility(
    manifest: Manifest,
    hardware: HardwareReport,
    *,
    runtime: RuntimeAvailability | None = None,
    storage: StorageEstimate | None = None,
) -> CompatibilityReport:
    """Compare a bundle's declared requirements against the host."""
    report = CompatibilityReport()
    requirements = manifest.requirements

    # -- CPU architecture --------------------------------------------------
    wanted = {p.split("/")[-1] for p in manifest.platforms} or {"amd64"}
    if hardware.architecture in wanted:
        report.checks.append(
            CheckResult(
                name="CPU architecture", status=CheckStatus.OK, detail=hardware.architecture
            )
        )
    else:
        report.checks.append(
            CheckResult(
                name="CPU architecture",
                status=CheckStatus.FAILED,
                detail=f"requires {', '.join(sorted(wanted))}; "
                f"{hardware.describes} is {hardware.architecture}",
            )
        )

    # -- operating system --------------------------------------------------
    if hardware.is_linux:
        report.checks.append(
            CheckResult(
                name="Operating system",
                status=CheckStatus.OK,
                detail=f"{hardware.os_name} {hardware.os_version}",
            )
        )
    else:
        report.checks.append(
            CheckResult(
                name="Operating system",
                status=CheckStatus.WARNING,
                detail=f"{hardware.os_name} is not a supported target platform",
            )
        )
        report.warnings.append(
            f"{hardware.os_name} is a development convenience only. OS package "
            "and GPU steps will be skipped."
        )

    # -- CPU cores ---------------------------------------------------------
    if requirements.minimum_cpu_cores is not None:
        ok = hardware.cpu.cores >= requirements.minimum_cpu_cores
        report.checks.append(
            CheckResult(
                name="CPU cores",
                status=CheckStatus.OK if ok else CheckStatus.FAILED,
                detail=f"requires {requirements.minimum_cpu_cores}, "
                f"{hardware.describes} has {hardware.cpu.cores}",
            )
        )

    # -- RAM ---------------------------------------------------------------
    if requirements.minimum_ram_gb is not None:
        if hardware.memory is None:
            report.checks.append(
                CheckResult(
                    name="RAM",
                    status=CheckStatus.SKIPPED,
                    detail=f"not stated by {hardware.describes}",
                )
            )
        else:
            ok = hardware.memory.total_gb >= requirements.minimum_ram_gb
            report.checks.append(
                CheckResult(
                    name="RAM",
                    status=CheckStatus.OK if ok else CheckStatus.FAILED,
                    detail=f"requires {requirements.minimum_ram_gb} GB, "
                    f"{hardware.describes} has {hardware.memory.total_gb:.1f} GB",
                )
            )

    # -- disk --------------------------------------------------------------
    if storage is not None:
        report.checks.append(
            CheckResult(
                name="Disk space",
                status=CheckStatus.OK if storage.sufficient else CheckStatus.FAILED,
                detail=f"needs {format_bytes(storage.recommended_bytes)}, "
                f"{format_bytes(storage.available_bytes)} available",
            )
        )

    # -- container runtime -------------------------------------------------
    if requirements.docker is not None:
        if runtime is None:
            report.checks.append(
                CheckResult(
                    name="Container runtime",
                    status=CheckStatus.SKIPPED,
                    detail="not checked",
                )
            )
        elif not runtime.available:
            report.checks.append(
                CheckResult(
                    name="Container runtime",
                    status=CheckStatus.FAILED,
                    detail=runtime.detail or "unavailable",
                )
            )
        else:
            minimum = requirements.docker.minimum_version
            detail = runtime.version or "present"
            status = CheckStatus.OK
            if (
                minimum
                and runtime.version
                and parse_driver_version(runtime.version) < parse_driver_version(minimum)
            ):
                # A warning, not a failure: the bundle's minimum is the version
                # it was tested against, and an older daemon usually still works.
                # Refusing outright would be wrong.
                status = CheckStatus.WARNING
                detail = f"{runtime.version}, below the tested minimum {minimum}"
                report.warnings.append(
                    f"The container runtime is {runtime.version}; this bundle was "
                    f"built against {minimum} or newer."
                )
            report.checks.append(
                CheckResult(name="Container runtime", status=status, detail=detail)
            )

    # -- GPU ---------------------------------------------------------------
    gpu = requirements.gpu
    if gpu is None:
        report.checks.append(
            CheckResult(name="GPU", status=CheckStatus.SKIPPED, detail="not required")
        )
    elif not hardware.gpus:
        report.checks.append(
            CheckResult(
                name="GPU",
                status=CheckStatus.FAILED,
                detail=hardware.gpu_detail or "no GPU detected",
            )
        )
    else:
        report.checks.append(
            CheckResult(
                name="GPU",
                status=CheckStatus.OK,
                detail=f"{len(hardware.gpus)}x {hardware.gpus[0].name}",
            )
        )

        if gpu.count > len(hardware.gpus):
            report.checks.append(
                CheckResult(
                    name="GPU count",
                    status=CheckStatus.FAILED,
                    detail=f"requires {gpu.count}, {hardware.describes} has {len(hardware.gpus)}",
                )
            )

        if gpu.minimum_memory_gb is not None:
            # Compared per device, not summed: a model needing 48 GB on one
            # card is not satisfied by two 24 GB cards unless the workload
            # shards, which is an application decision we must not assume
            # (section 30).
            # Compared on the advertised (rounded) capacity: an 80 GB card
            # reports 79.6 GiB once the reserved region is excluded, and a
            # requirement of 80 must not reject the card it names.
            largest = max((g.advertised_gb for g in hardware.gpus), default=0)
            ok = largest >= gpu.minimum_memory_gb
            report.checks.append(
                CheckResult(
                    name="GPU VRAM",
                    status=CheckStatus.OK if ok else CheckStatus.FAILED,
                    detail=f"requires {gpu.minimum_memory_gb} GB per device, "
                    f"largest device has {largest} GB",
                )
            )

        if gpu.minimum_driver:
            if hardware.nvidia_driver is None:
                report.checks.append(
                    CheckResult(
                        name="GPU driver",
                        status=CheckStatus.SKIPPED,
                        detail="driver version could not be determined",
                    )
                )
            else:
                ok = parse_driver_version(hardware.nvidia_driver) >= parse_driver_version(
                    gpu.minimum_driver
                )
                report.checks.append(
                    CheckResult(
                        name="GPU driver",
                        status=CheckStatus.OK if ok else CheckStatus.FAILED,
                        detail=f"requires {gpu.minimum_driver} or newer, "
                        f"{hardware.describes} has {hardware.nvidia_driver}",
                    )
                )

    return report


def estimate_storage(
    manifest: Manifest,
    *,
    bundle_bytes: int,
    available_bytes: int,
    models_are_hardlinked: bool = True,
) -> StorageEstimate:
    """Estimate the space an install needs (section 22).

    Deliberately not the specification's naive sum. Import streams into
    content-addressed storage rather than extracting, and models are hard
    linked into place where the filesystem allows it, so two of the
    specification's five terms are genuinely zero. Reporting the real number
    is more useful than inflating it to match the example - an operator who
    is told they need 214 GB when 62 GB will do may simply not deploy.
    """
    sizes = manifest.size_by_type()
    container_bytes = sizes.get(ArtifactType.OCI_IMAGE, 0)
    model_bytes = sizes.get(ArtifactType.MODEL, 0)

    return StorageEstimate(
        bundle_bytes=bundle_bytes,
        artifact_storage_bytes=manifest.total_size,
        # Docker expands layers when loading an image tar; 1.3x is a
        # conservative allowance for that.
        container_storage_bytes=int(container_bytes * 1.3),
        # Hard links cost nothing; a copy costs the model again.
        model_storage_bytes=0 if models_are_hardlinked else model_bytes,
        temporary_bytes=min(1024**3, max(64 * 1024**2, manifest.total_size // 100)),
        available_bytes=available_bytes,
    )
