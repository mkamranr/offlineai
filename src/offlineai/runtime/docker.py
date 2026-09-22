"""Docker implementation of :class:`~offlineai.runtime.base.ContainerRuntime`.

Drives the ``docker`` CLI rather than the HTTP API. On an air-gapped host the
CLI is what an operator already has, already trusts and can already run by
hand to check our work - and it avoids adding a Python dependency that would
have to be present on the target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from offlineai.errors import RuntimeFailureError
from offlineai.logging import get_logger
from offlineai.runtime.base import (
    ContainerState,
    ContainerStatus,
    ImageInfo,
    RunSpec,
    RuntimeAvailability,
)
from offlineai.utils.proc import CommandResult, have, run

__all__ = ["DockerRuntime"]

logger = get_logger("runtime.docker")

_STATUS_MAP = {
    "running": ContainerStatus.RUNNING,
    "exited": ContainerStatus.EXITED,
    "created": ContainerStatus.CREATED,
    "restarting": ContainerStatus.RESTARTING,
    "paused": ContainerStatus.STOPPED,
    "dead": ContainerStatus.EXITED,
    "removing": ContainerStatus.STOPPED,
}


class DockerRuntime:
    kind: ClassVar[str] = "docker"

    def __init__(self, executable: str = "docker") -> None:
        self.executable = executable

    # -- availability ----------------------------------------------------

    def availability(self) -> RuntimeAvailability:
        if not have(self.executable):
            return RuntimeAvailability(available=False, detail=f"{self.executable} is not on PATH")
        version = run([self.executable, "version", "--format", "{{.Server.Version}}"], timeout=30)
        if not version.ok:
            return RuntimeAvailability(
                available=False,
                detail="the Docker daemon is not reachable: " + version.output.strip(),
            )
        return RuntimeAvailability(
            available=True,
            version=version.first_line(),
            gpu_support=self._has_gpu_runtime(),
        )

    def _has_gpu_runtime(self) -> bool:
        result = run([self.executable, "info", "--format", "{{json .Runtimes}}"], timeout=30)
        if not result.ok:
            return False
        try:
            runtimes = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return False
        return "nvidia" in runtimes

    # -- images ----------------------------------------------------------

    def pull(self, reference: str, *, platform: str | None = None) -> ImageInfo:
        args = [self.executable, "pull"]
        if platform:
            args += ["--platform", platform]
        args.append(reference)
        self._require(run(args), f"failed to pull image {reference}")
        info = self.image_info(reference)
        if info is None:
            raise RuntimeFailureError(f"image {reference} is missing after a successful pull")
        return info

    def build(
        self, *, dockerfile: Path, context: Path, tag: str, platform: str | None = None
    ) -> ImageInfo:
        args = [self.executable, "build", "-f", str(dockerfile), "-t", tag]
        if platform:
            args += ["--platform", platform]
        args.append(str(context))
        self._require(run(args), f"failed to build image {tag}")
        info = self.image_info(tag)
        if info is None:
            raise RuntimeFailureError(f"image {tag} is missing after a successful build")
        return info

    def save(self, references: list[str], destination: Path) -> None:
        if not references:
            raise RuntimeFailureError("no image references given to save")
        destination.parent.mkdir(parents=True, exist_ok=True)
        args = [self.executable, "save", "-o", str(destination), *references]
        self._require(run(args), f"failed to save images to {destination}")

    def load(self, archive: Path) -> list[str]:
        result = run([self.executable, "load", "-i", str(archive)])
        self._require(result, f"failed to load images from {archive.name}")
        loaded: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Loaded image: "):
                loaded.append(line.removeprefix("Loaded image: ").strip())
            elif line.startswith("Loaded image ID: "):
                loaded.append(line.removeprefix("Loaded image ID: ").strip())
        return loaded

    def image_info(self, reference: str) -> ImageInfo | None:
        result = run([self.executable, "image", "inspect", reference], timeout=60)
        if not result.ok:
            return None
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        if not payload:
            return None
        entry = payload[0]
        return ImageInfo(
            reference=reference,
            digest=_first_digest(entry.get("RepoDigests") or []),
            image_id=entry.get("Id"),
            size=int(entry.get("Size") or 0),
        )

    def remove_image(self, reference: str) -> None:
        run([self.executable, "image", "rm", "-f", reference], timeout=120)

    # -- containers ------------------------------------------------------

    def run_container(self, spec: RunSpec) -> str:
        args = [self.executable, "run", "-d", "--name", spec.name]
        args += ["--restart", spec.restart_policy]

        for key, value in spec.environment.items():
            args += ["-e", f"{key}={value}"]
        for mapping in spec.ports:
            args += ["-p", mapping]
        for host, container, read_only in spec.volumes:
            args += ["-v", f"{host}:{container}:ro" if read_only else f"{host}:{container}"]
        for key, value in spec.labels.items():
            args += ["--label", f"{key}={value}"]
        if spec.network:
            args += ["--network", spec.network]
        if spec.gpu_device_ids is not None:
            # Section 30: expose exactly the requested devices, and make no
            # assumption about how the workload divides work between them.
            devices = ",".join(spec.gpu_device_ids) if spec.gpu_device_ids else "all"
            args += ["--gpus", f'"device={devices}"' if spec.gpu_device_ids else "all"]
        # Section 32: never let a run reach out for a missing image. If it is
        # not already loaded that is a bundle problem, and it must be reported
        # as one rather than papered over by a silent pull.
        args += ["--pull", "never"]
        args.append(spec.image)
        args += spec.command

        result = run(args)
        self._require(result, f"failed to start container {spec.name}")
        return result.first_line()

    def stop_container(self, name: str, *, timeout: int = 10) -> None:
        run([self.executable, "stop", "-t", str(timeout), name], timeout=timeout + 30)

    def remove_container(self, name: str) -> None:
        run([self.executable, "rm", "-f", name], timeout=120)

    def container_state(self, name: str) -> ContainerState:
        result = run([self.executable, "inspect", name], timeout=60)
        if not result.ok:
            return ContainerState(name=name, status=ContainerStatus.NOT_FOUND)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return ContainerState(name=name, status=ContainerStatus.UNKNOWN)
        if not payload:
            return ContainerState(name=name, status=ContainerStatus.NOT_FOUND)

        entry = payload[0]
        state = entry.get("State") or {}
        health = (state.get("Health") or {}).get("Status")
        return ContainerState(
            name=name,
            status=_STATUS_MAP.get(str(state.get("Status", "")).lower(), ContainerStatus.UNKNOWN),
            container_id=entry.get("Id"),
            health=health,
            exit_code=state.get("ExitCode"),
            ports=_parse_ports(entry),
        )

    def logs(self, name: str, *, tail: int | None = None) -> str:
        args = [self.executable, "logs"]
        if tail is not None:
            args += ["--tail", str(tail)]
        args.append(name)
        result = run(args, timeout=120)
        if not result.ok and "No such container" in result.stderr:
            raise RuntimeFailureError(
                f"container {name} does not exist",
                action="Has the package been installed and started?",
            )
        # Docker splits container output across both streams; callers want both.
        return result.stdout + result.stderr

    def exec_in(self, name: str, command: list[str]) -> tuple[int, str]:
        result = run([self.executable, "exec", name, *command], timeout=300)
        return result.returncode, result.output

    # -- networks --------------------------------------------------------

    def create_network(self, name: str) -> None:
        existing = run([self.executable, "network", "inspect", name], timeout=30)
        if existing.ok:
            return
        self._require(
            run([self.executable, "network", "create", name], timeout=60),
            f"failed to create network {name}",
        )

    def remove_network(self, name: str) -> None:
        run([self.executable, "network", "rm", name], timeout=60)

    # -- helpers ---------------------------------------------------------

    def _require(self, result: CommandResult, message: str) -> None:
        if result.ok:
            return
        raise RuntimeFailureError(
            message,
            details={"Command": " ".join(result.args), "Output": result.output or "(none)"},
            action=_advice(result.output),
        )


def _advice(output: str) -> str | None:
    lowered = output.lower()
    if "permission denied" in lowered and "docker.sock" in lowered:
        return (
            "The current user cannot reach the Docker socket. Add the user to the "
            "'docker' group, or re-run with sufficient privileges."
        )
    if "cannot connect to the docker daemon" in lowered:
        return "Start the Docker daemon and try again."
    if "pull access denied" in lowered or "manifest unknown" in lowered:
        return (
            "The image is not present locally and cannot be fetched. On an "
            "air-gapped host this means the bundle did not contain it - rebuild "
            "the bundle with the image included."
        )
    if "no space left on device" in lowered:
        return "The Docker storage volume is full. Free space and retry."
    return None


def _first_digest(repo_digests: list[str]) -> str | None:
    for entry in repo_digests:
        _, _, digest = entry.partition("@")
        if digest.startswith("sha256:"):
            return digest
    return None


def _parse_ports(entry: dict[str, Any]) -> dict[str, str]:
    ports = ((entry.get("NetworkSettings") or {}).get("Ports")) or {}
    out: dict[str, str] = {}
    for container_port, bindings in ports.items():
        if bindings:
            binding = bindings[0]
            out[container_port] = f"{binding.get('HostIp', '')}:{binding.get('HostPort', '')}"
    return out
