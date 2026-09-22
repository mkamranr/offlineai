"""The installation pipeline (sections 26, 27 and 32).

Order is chosen so that everything which can fail cheaply fails before anything
that changes the system: platform, disk, runtime and hardware are all checked
before the first image is loaded. An operator on an air-gapped host cannot
simply fetch the missing piece, so a failure needs to arrive before the machine
has been half-modified, not after.

Steps that change the system journal their own undo actions as they go, so
``offlineai rollback`` works even if the process is killed mid-install.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from offlineai.bundler.results import CheckResult, CheckStatus
from offlineai.config.settings import Settings
from offlineai.errors import (
    HardwareIncompatibleError,
    InstallationError,
    InsufficientDiskError,
    MissingArtifactError,
)
from offlineai.installer.transaction import (
    InstallState,
    InstallTransaction,
    Inverse,
    InverseAction,
    next_install_id,
)
from offlineai.logging import get_logger
from offlineai.registry.registry import PackageRecord, Registry
from offlineai.runtime.base import ContainerRuntime
from offlineai.runtime.manager import RuntimeManager, image_overrides_from, network_name
from offlineai.schema.manifest import ArtifactType
from offlineai.schema.package import Package
from offlineai.utils.fs import free_space
from offlineai.utils.sizes import format_bytes

__all__ = ["InstallResult", "Installer"]

logger = get_logger("installer")

#: The target platform the specification treats as supported (section 77.12).
SUPPORTED_OS = "linux"


@dataclass(slots=True)
class InstallResult:
    package: str
    version: str
    installation_id: str
    state: InstallState
    checks: list[CheckResult] = field(default_factory=list)
    services_started: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    healthy: bool | None = None
    health_detail: str | None = None
    warnings: list[str] = field(default_factory=list)
    dev_mode: bool = False

    @property
    def succeeded(self) -> bool:
        return self.state is InstallState.COMPLETED


class Installer:
    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        runtime: ContainerRuntime,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.runtime = runtime
        self.manager = RuntimeManager(runtime)

    # -- install ---------------------------------------------------------

    def install(
        self,
        name: str,
        *,
        version: str | None = None,
        start: bool = True,
        gpu_device_ids: list[str] | None = None,
        environment: dict[str, str] | None = None,
        health_sleep: float | None = None,
    ) -> InstallResult:
        record = self.registry.require(name, version)
        package = _package_from(record)

        install_id = next_install_id(self.registry.db_path)
        txn = InstallTransaction(self.registry.db_path, install_id, record.id)
        txn.begin({"gpu_device_ids": gpu_device_ids or []})

        result = InstallResult(
            package=record.name,
            version=record.version,
            installation_id=install_id,
            state=InstallState.PENDING,
        )

        try:
            txn.set_state(InstallState.VALIDATING)
            self._preflight(record, result)

            txn.set_state(InstallState.INSTALLING)
            self._load_images(record, package, txn, result)
            model_root = self._materialise_models(record, txn, result)
            self._install_system_packages(package, result)
            self._install_python(package, result)
            self._write_runtime_config(record, package, txn, result)

            if start:
                txn.set_state(InstallState.STARTING)
                self._start(
                    package,
                    txn,
                    result,
                    model_root,
                    gpu_device_ids,
                    environment,
                    image_overrides_from(record.manifest()),
                )

                txn.set_state(InstallState.HEALTH_CHECK)
                healthy, detail = self.manager.wait_for_health(package, sleep=health_sleep)
                result.healthy = healthy
                result.health_detail = detail
                if not healthy:
                    raise InstallationError(
                        "the workload did not become healthy",
                        details={"Detail": detail},
                        action=f"Inspect the logs:\n  offlineai logs {record.name}\n"
                        f"Then roll back if needed:\n  offlineai rollback {install_id}",
                    )

                report = self.manager.status(package)
                result.endpoints = report.endpoints

            txn.set_state(InstallState.COMPLETED)
            result.state = InstallState.COMPLETED
            return result

        except Exception as exc:
            txn.set_state(InstallState.FAILED, error=str(exc))
            result.state = InstallState.FAILED
            raise

    # -- pipeline stages -------------------------------------------------

    def _preflight(self, record: PackageRecord, result: InstallResult) -> None:
        manifest = record.manifest()
        checks = result.checks

        host_os = platform.system().lower()
        if host_os != SUPPORTED_OS:
            # Degraded development mode. Sections 21 and 77.12 make Linux the
            # supported target; running elsewhere is allowed but must be
            # labelled, never quietly reported as fine.
            result.dev_mode = True
            checks.append(
                CheckResult(
                    name="Operating system",
                    status=CheckStatus.WARNING,
                    detail=f"{host_os} is not a supported target platform (DEV MODE)",
                )
            )
            result.warnings.append(
                f"Running on {host_os}. This is a development convenience, not a "
                "supported deployment target; OS package and GPU steps will be skipped."
            )
        else:
            checks.append(
                CheckResult(name="Operating system", status=CheckStatus.OK, detail=host_os)
            )

        machine = platform.machine().lower()
        host_arch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
        wanted = {p.split("/")[-1] for p in manifest.platforms} or {"amd64"}
        if host_arch in wanted:
            checks.append(
                CheckResult(name="CPU architecture", status=CheckStatus.OK, detail=host_arch)
            )
        else:
            raise HardwareIncompatibleError(
                "the bundle does not support this CPU architecture",
                details={"Required": ", ".join(sorted(wanted)), "Detected": host_arch},
                action="Rebuild the bundle for this architecture on the builder machine.",
            )

        required = manifest.total_size
        available = free_space(self.settings.data_dir)
        if required > available:
            raise InsufficientDiskError(
                "not enough free space to install this package",
                details={
                    "Required": format_bytes(required),
                    "Available": format_bytes(available),
                },
                action="Free space, or install onto another volume with --data-dir.",
            )
        checks.append(
            CheckResult(
                name="Disk space",
                status=CheckStatus.OK,
                detail=f"{format_bytes(available)} available",
            )
        )

        availability = self.runtime.availability()
        if not availability.available:
            raise InstallationError(
                "no usable container runtime",
                details={"Detail": availability.detail or "unavailable"},
                action="Install and start Docker, then try again.",
            )
        checks.append(
            CheckResult(
                name="Container runtime",
                status=CheckStatus.OK,
                detail=f"{self.runtime.kind} {availability.version}",
            )
        )

        gpu = manifest.requirements.gpu
        if gpu is None:
            checks.append(
                CheckResult(name="GPU", status=CheckStatus.SKIPPED, detail="not required")
            )
        elif result.dev_mode:
            checks.append(
                CheckResult(
                    name="GPU",
                    status=CheckStatus.SKIPPED,
                    detail=f"not evaluated on {host_os}",
                )
            )
        elif not availability.gpu_support:
            checks.append(
                CheckResult(
                    name="GPU",
                    status=CheckStatus.WARNING,
                    detail="the container runtime reports no GPU support",
                )
            )
            result.warnings.append(
                "This package requests a GPU but the container runtime exposes none. "
                "The workload may fail to start."
            )
        else:
            checks.append(
                CheckResult(name="GPU", status=CheckStatus.OK, detail=gpu.vendor or "present")
            )

        missing_secrets = [
            secret for secret in manifest.required_secrets if not _env_present(secret)
        ]
        if missing_secrets:
            checks.append(
                CheckResult(
                    name="Required secrets",
                    status=CheckStatus.WARNING,
                    detail=", ".join(f"{s}: REQUIRED" for s in missing_secrets),
                )
            )
            result.warnings.append(
                "These environment variables are declared as externally supplied and "
                f"are not set: {', '.join(missing_secrets)}"
            )

    def _load_images(
        self,
        record: PackageRecord,
        package: Package,
        txn: InstallTransaction,
        result: InstallResult,
    ) -> None:
        archives = [
            (digest, bundle_path)
            for digest, _, bundle_path in self.registry.artifacts_for(record.id)
            if bundle_path.startswith("artifacts/containers/")
        ]

        if not archives:
            if package.containers:
                raise MissingArtifactError(
                    f"the bundle declares {len(package.containers)} container image(s) "
                    "but contains none",
                    details={"Expected": "\n".join(str(c.reference) for c in package.containers)},
                    action="Rebuild the bundle on a machine with a container runtime "
                    "so the images are included.",
                )
            result.checks.append(
                CheckResult(
                    name="Container images", status=CheckStatus.SKIPPED, detail="none declared"
                )
            )
            return

        with txn.step("load container images") as inverses:
            loaded: list[str] = []
            for digest, _ in archives:
                path = self.registry.artifact_path(digest)
                if not path.is_file():
                    raise MissingArtifactError(
                        f"image archive {digest[:12]} is missing from the local store",
                        action="Re-import the bundle.",
                    )
                for reference in self.runtime.load(path):
                    loaded.append(reference)
                    inverses.append(Inverse(action=InverseAction.REMOVE_IMAGE, target=reference))

        # Section 16: a tag is mutable, so confirm we got the image the bundle
        # recorded rather than something that happens to share its name.
        self._verify_digests(record, package, result)

        result.checks.append(
            CheckResult(
                name="Container images",
                status=CheckStatus.OK,
                detail=f"{len(loaded)} image(s) loaded",
            )
        )

    def _verify_digests(
        self, record: PackageRecord, package: Package, result: InstallResult
    ) -> None:
        manifest = record.manifest()
        recorded = {
            str(entry.metadata.get("container")): entry.digest
            for entry in manifest.artifacts_of_type(ArtifactType.OCI_IMAGE)
            if entry.digest
        }
        overrides = image_overrides_from(manifest)
        for container in package.containers:
            expected = recorded.get(container.name)
            if not expected:
                continue
            reference = overrides.get(container.name, str(container.reference))
            info = self.runtime.image_info(reference)
            if info is None or info.digest is None:
                continue
            if info.digest != expected:
                result.warnings.append(
                    f"image {reference} has digest {info.digest[:19]}… but the "
                    f"bundle recorded {expected[:19]}…. The loaded image is not the one "
                    "this bundle was built against."
                )

    def _materialise_models(
        self, record: PackageRecord, txn: InstallTransaction, result: InstallResult
    ) -> Path | None:
        entries = [
            (digest, bundle_path)
            for digest, _, bundle_path in self.registry.artifacts_for(record.id)
            if bundle_path.startswith("artifacts/models/")
        ]
        if not entries:
            result.checks.append(
                CheckResult(name="Models", status=CheckStatus.SKIPPED, detail="none declared")
            )
            return None

        root = self.settings.data_dir / "installed" / record.name / "models"
        with txn.step("install models") as inverses:
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
            inverses.append(Inverse(action=InverseAction.REMOVE_PATH, target=str(root)))

            for digest, bundle_path in entries:
                relative = bundle_path.removeprefix("artifacts/models/")
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = self.registry.artifact_path(digest)
                if not source.is_file():
                    raise MissingArtifactError(
                        f"model file {relative} is missing from the local store",
                        action="Re-import the bundle.",
                    )
                # Hard link where the filesystem allows it: a 48 GB checkpoint
                # should not be duplicated just to give it a friendly path.
                try:
                    destination.hardlink_to(source)
                except (OSError, NotImplementedError):
                    shutil.copy2(source, destination)

        result.checks.append(
            CheckResult(name="Models", status=CheckStatus.OK, detail=f"{len(entries)} file(s)")
        )
        return root

    def _install_system_packages(self, package: Package, result: InstallResult) -> None:
        declared = package.system.packages if package.system else []
        if not declared:
            result.checks.append(
                CheckResult(
                    name="System packages", status=CheckStatus.SKIPPED, detail="none declared"
                )
            )
            return
        reason = (
            "dpkg is unavailable on this platform"
            if result.dev_mode
            else "OS package installation arrives in a later phase"
        )
        result.checks.append(
            CheckResult(name="System packages", status=CheckStatus.SKIPPED, detail=reason)
        )
        result.warnings.append(f"{len(declared)} system package(s) were not installed: {reason}")

    def _install_python(self, package: Package, result: InstallResult) -> None:
        if not (package.python and package.python.requirements):
            result.checks.append(
                CheckResult(
                    name="Python dependencies",
                    status=CheckStatus.SKIPPED,
                    detail="none declared",
                )
            )
            return
        result.checks.append(
            CheckResult(
                name="Python dependencies",
                status=CheckStatus.SKIPPED,
                detail="wheel installation arrives in a later phase",
            )
        )

    def _write_runtime_config(
        self,
        record: PackageRecord,
        package: Package,
        txn: InstallTransaction,
        result: InstallResult,
    ) -> None:  # noqa: ARG002 - package is read below via package.services
        directory = self.settings.data_dir / "installed" / record.name
        with txn.step("generate runtime configuration") as inverses:
            directory.mkdir(parents=True, exist_ok=True)
            config_file = directory / "runtime.yaml"
            config_file.write_text(
                yaml.safe_dump(
                    {
                        "package": record.name,
                        "version": record.version,
                        "installation": result.installation_id,
                        "services": [s.name for s in package.services],
                        "network": network_name(record.name),
                    },
                    sort_keys=True,
                )
            )
            inverses.append(Inverse(action=InverseAction.REMOVE_PATH, target=str(config_file)))
        result.checks.append(CheckResult(name="Runtime configuration", status=CheckStatus.OK))

    def _start(
        self,
        package: Package,
        txn: InstallTransaction,
        result: InstallResult,
        model_root: Path | None,
        gpu_device_ids: list[str] | None,
        environment: dict[str, str] | None,
        image_overrides: dict[str, str] | None = None,
    ) -> None:
        if not package.services:
            result.checks.append(
                CheckResult(name="Services", status=CheckStatus.SKIPPED, detail="none declared")
            )
            return

        with txn.step("start services") as inverses:
            inverses.append(
                Inverse(
                    action=InverseAction.REMOVE_NETWORK,
                    target=network_name(package.metadata.name),
                )
            )
            names = self.manager.start(
                package,
                model_root=model_root,
                gpu_device_ids=gpu_device_ids,
                environment=environment,
                image_overrides=image_overrides,
            )
            for name in names:
                inverses.append(Inverse(action=InverseAction.REMOVE_CONTAINER, target=name))
            result.services_started = names

        result.checks.append(
            CheckResult(
                name="Services",
                status=CheckStatus.OK,
                detail=f"{len(result.services_started)} started",
            )
        )


def _package_from(record: PackageRecord) -> Package:
    if not record.package_yaml.strip():
        raise InstallationError(
            f"the registry has no package definition for {record.identifier}",
            action="Re-import the bundle.",
        )
    return Package.model_validate(yaml.safe_load(record.package_yaml))


def _env_present(name: str) -> bool:
    import os

    return bool(os.environ.get(name))
