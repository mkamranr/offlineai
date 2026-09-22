"""Runtime management (section 28): start, stop, restart, status, logs.

Containers are named ``offlineai-<package>-<service>`` and labelled, so an
operator can find them with plain ``docker ps`` and so we can find ours without
touching anything we did not create.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from offlineai.errors import RuntimeFailureError
from offlineai.logging import get_logger
from offlineai.runtime.base import ContainerRuntime, ContainerStatus, RunSpec
from offlineai.schema.manifest import ArtifactType, Manifest
from offlineai.schema.package import Package, ServiceSpec

__all__ = [
    "LABEL_PACKAGE",
    "RuntimeManager",
    "ServiceState",
    "StatusReport",
    "WHEEL_MOUNT",
    "image_overrides_from",
]

logger = get_logger("runtime.manager")

LABEL_PACKAGE = "ai.offlineai.package"
LABEL_VERSION = "ai.offlineai.version"
LABEL_SERVICE = "ai.offlineai.service"

#: Where the bundle's wheel closure appears inside a container.
WHEEL_MOUNT = "/opt/offlineai/wheels"


def container_name(package: str, service: str) -> str:
    return f"offlineai-{package}-{service}"


def network_name(package: str) -> str:
    return f"offlineai-{package}"


@dataclass(frozen=True, slots=True)
class ServiceState:
    name: str
    container: str
    status: ContainerStatus
    health: str | None = None
    ports: list[str] = field(default_factory=list)
    exit_code: int | None = None


@dataclass(slots=True)
class StatusReport:
    package: str
    version: str
    status: str
    services: list[ServiceState] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    healthy: bool | None = None


class RuntimeManager:
    def __init__(self, runtime: ContainerRuntime) -> None:
        self.runtime = runtime

    # -- lifecycle -------------------------------------------------------

    def start(
        self,
        package: Package,
        *,
        model_root: Path | None = None,
        gpu_device_ids: list[str] | None = None,
        environment: dict[str, str] | None = None,
        port_overrides: dict[str, list[str]] | None = None,
        image_overrides: dict[str, str] | None = None,
        wheel_root: Path | None = None,
    ) -> list[str]:
        """Start every service. Returns the container names created.

        ``image_overrides`` maps a container name to the image reference that
        should actually run. It exists because a container declaring
        ``dockerfile:`` does not run the image it names - that is its *base* -
        it runs the image the builder produced from that Dockerfile. Without
        this, install would start the base image and the application would
        never run.
        """
        name = package.metadata.name
        network = network_name(name)
        self.runtime.create_network(network)

        started: list[str] = []
        for service in _ordered_services(package):
            # Clear a container we previously created under the same name.
            # Reinstalling is an ordinary operation; failing on a name
            # collision with our own leftover container is not helpful.
            existing = container_name(name, service.name)
            if self.runtime.container_state(existing).status is not ContainerStatus.NOT_FOUND:
                logger.debug("removing existing container %s", existing)
                self.runtime.stop_container(existing)
                self.runtime.remove_container(existing)
            container = self._start_service(
                package,
                service,
                network=network,
                model_root=model_root,
                gpu_device_ids=gpu_device_ids,
                extra_environment=environment or {},
                port_overrides=(port_overrides or {}).get(service.name),
                image_overrides=image_overrides or {},
                wheel_root=wheel_root,
            )
            started.append(container)
        return started

    def _start_service(
        self,
        package: Package,
        service: ServiceSpec,
        *,
        network: str,
        model_root: Path | None,
        gpu_device_ids: list[str] | None,
        extra_environment: dict[str, str],
        port_overrides: list[str] | None,
        image_overrides: dict[str, str] | None = None,
        wheel_root: Path | None = None,
    ) -> str:
        container_spec = next((c for c in package.containers if c.name == service.container), None)
        if container_spec is None:
            raise RuntimeFailureError(
                f"service {service.name!r} references container "
                f"{service.container!r}, which is not declared"
            )

        environment = {**package.environment, **service.environment, **extra_environment}

        # Wheels from the bundle are mounted read-only and pip is pointed at
        # them with the index disabled. Any `pip install` inside the workload
        # then resolves from the bundle and fails loudly otherwise, rather than
        # quietly reaching for PyPI (section 32).
        if wheel_root is not None and wheel_root.is_dir():
            environment.setdefault("PIP_NO_INDEX", "1")
            environment.setdefault("PIP_FIND_LINKS", WHEEL_MOUNT)
            environment.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

        volumes: list[tuple[str, str, bool]] = []
        for volume in package.volumes:
            host = Path(volume.host).expanduser().absolute()
            host.mkdir(parents=True, exist_ok=True)
            volumes.append((str(host), volume.container, volume.read_only))

        # Models are mounted read-only. The workload should never be able to
        # modify the weights it was shipped, and read-only makes that explicit
        # rather than merely unlikely.
        if wheel_root is not None and wheel_root.is_dir():
            volumes.append((str(wheel_root), WHEEL_MOUNT, True))

        if model_root is not None:
            for model in package.models:
                if model.destination:
                    source = model_root / model.name
                    if source.is_dir():
                        volumes.append((str(source), model.destination, True))

        image = (image_overrides or {}).get(container_spec.name, str(container_spec.reference))

        name = container_name(package.metadata.name, service.name)
        # Returns the NAME, not the id the runtime hands back. Everything
        # downstream - rollback inverses, status lookups, logs - addresses
        # containers by name, and a rollback target that is an id would be
        # looked up by name and silently match nothing.
        container_id = self.runtime.run_container(
            RunSpec(
                name=name,
                image=image,
                command=service.command,
                environment=environment,
                ports=port_overrides if port_overrides is not None else service.ports,
                volumes=volumes,
                network=network,
                gpu_device_ids=gpu_device_ids,
                labels={
                    LABEL_PACKAGE: package.metadata.name,
                    LABEL_VERSION: package.metadata.version,
                    LABEL_SERVICE: service.name,
                },
            )
        )
        logger.debug("started %s as %s", name, container_id)
        return name

    def stop(self, package: Package, *, remove: bool = False) -> list[str]:
        stopped: list[str] = []
        for service in reversed(_ordered_services(package)):
            name = container_name(package.metadata.name, service.name)
            state = self.runtime.container_state(name)
            if state.status is ContainerStatus.NOT_FOUND:
                continue
            self.runtime.stop_container(name)
            if remove:
                self.runtime.remove_container(name)
            stopped.append(name)
        return stopped

    def restart(self, package: Package, **start_kwargs: object) -> list[str]:
        self.stop(package, remove=True)
        return self.start(package, **start_kwargs)  # type: ignore[arg-type]

    # -- observation -----------------------------------------------------

    def status(self, package: Package) -> StatusReport:
        services: list[ServiceState] = []
        endpoints: list[str] = []

        for service in _ordered_services(package):
            name = container_name(package.metadata.name, service.name)
            state = self.runtime.container_state(name)
            # Report the ports the container is ACTUALLY bound to, not the ones
            # the package declared. A --config override remaps them, and telling
            # an operator to curl a port nothing is listening on is worse than
            # saying nothing at all.
            bound = _bound_host_ports(state.ports)
            services.append(
                ServiceState(
                    name=service.name,
                    container=name,
                    status=state.status,
                    health=state.health,
                    ports=[str(p) for p in bound] or service.ports,
                    exit_code=state.exit_code,
                )
            )
            if state.running:
                for port in bound or service.host_ports():
                    endpoints.append(f"http://localhost:{port}")

        # Classified by how many services are running, not by which exact
        # container state they are in. `stop` without --remove leaves a
        # container EXITED rather than absent, and reporting that as DEGRADED
        # told the operator something was wrong when nothing was.
        running = sum(1 for s in services if s.status is ContainerStatus.RUNNING)
        if not services:
            overall = "NOT INSTALLED"
        elif running == len(services):
            overall = "RUNNING"
        elif running == 0:
            overall = "STOPPED"
        else:
            overall = "DEGRADED"

        return StatusReport(
            package=package.metadata.name,
            version=package.metadata.version,
            status=overall,
            services=services,
            endpoints=endpoints,
        )

    def logs(self, package: Package, *, service: str | None = None, tail: int | None = None) -> str:
        targets = [s for s in _ordered_services(package) if service in (None, s.name)]
        if not targets:
            known = ", ".join(s.name for s in package.services) or "none"
            raise RuntimeFailureError(
                f"package {package.metadata.name!r} has no service named {service!r}",
                details={"Services": known},
            )

        chunks: list[str] = []
        for target in targets:
            name = container_name(package.metadata.name, target.name)
            state = self.runtime.container_state(name)
            if state.status is ContainerStatus.NOT_FOUND:
                chunks.append(f"=== {target.name} ===\n(no container; is it running?)\n")
                continue
            header = f"=== {target.name} ===\n" if len(targets) > 1 else ""
            chunks.append(header + self.runtime.logs(name, tail=tail))
        return "\n".join(chunks)

    # -- health ----------------------------------------------------------

    def wait_for_health(self, package: Package, *, sleep: float | None = None) -> tuple[bool, str]:
        """Run the declared health check until it passes or the retries run out.

        Returns ``(healthy, detail)`` rather than raising: a workload that is
        installed but not yet healthy is a real state an operator needs
        reported, not an exception.
        """
        check = package.install.healthcheck
        if check is None or not check.command:
            return True, "no health check declared"

        service = package.services[0] if package.services else None
        if service is None:
            return True, "no services declared"

        name = container_name(package.metadata.name, service.name)
        interval = check.interval_seconds if sleep is None else sleep
        last = ""

        for attempt in range(1, check.retries + 1):
            state = self.runtime.container_state(name)
            if state.status is ContainerStatus.EXITED:
                return False, (
                    f"container {name} exited with code {state.exit_code} before becoming healthy"
                )
            code, output = self.runtime.exec_in(name, check.command)
            if code == 0:
                return True, f"healthy after {attempt} attempt(s)"
            last = output
            if attempt < check.retries:
                time.sleep(interval)

        return False, f"health check failed after {check.retries} attempts: {last}"


def _bound_host_ports(ports: dict[str, str]) -> list[int]:
    """Extract host ports from a runtime's port map.

    Values look like ``"0.0.0.0:8099"`` or ``":8099"`` depending on the engine
    and the binding, so the host port is whatever follows the last colon.
    """
    out: list[int] = []
    for binding in ports.values():
        candidate = binding.rsplit(":", 1)[-1].strip()
        if candidate.isdigit():
            out.append(int(candidate))
    return sorted(set(out))


def image_overrides_from(manifest: Manifest) -> dict[str, str]:
    """Map container name -> the image reference the bundle actually carries.

    Build time records the effective reference on each OCI artifact, which is
    the only place that knows whether an image was pulled as declared or built
    from a Dockerfile.
    """
    overrides: dict[str, str] = {}
    for entry in manifest.artifacts_of_type(ArtifactType.OCI_IMAGE):
        container = entry.metadata.get("container")
        if container and entry.source:
            overrides[str(container)] = entry.source
    return overrides


def _ordered_services(package: Package) -> list[ServiceSpec]:
    """Topologically order services by depends_on.

    Cycles are impossible: the schema validates that every dependency names a
    declared service, and a cycle would leave services unplaced, which is
    detected here rather than deadlocking.
    """
    remaining = {s.name: s for s in package.services}
    ordered: list[ServiceSpec] = []
    placed: set[str] = set()

    while remaining:
        ready = [s for s in remaining.values() if all(d in placed for d in s.depends_on)]
        if not ready:
            # A dependency cycle. Start them in declaration order rather than
            # refusing: the operator's stack may tolerate it, and a hang here
            # would be worse than a warning.
            logger.warning(
                "service dependency cycle among %s; starting in declaration order",
                ", ".join(sorted(remaining)),
            )
            ordered.extend(remaining.values())
            break
        for service in ready:
            ordered.append(service)
            placed.add(service.name)
            del remaining[service.name]

    return ordered
