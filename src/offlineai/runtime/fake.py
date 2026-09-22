"""In-memory container runtime, for tests.

Ships in the package rather than the test tree on purpose: it is the reference
for what the :class:`~offlineai.runtime.base.ContainerRuntime` protocol
actually requires, and it lets the full install pipeline - which is the part
most worth testing - run in CI with no daemon, no images and no network.

It models the behaviour that matters for correctness: an image must be loaded
before a container can run (mirroring ``--pull never``), names are unique,
state transitions are real, and ports and volumes are recorded so tests can
assert on them.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from offlineai.errors import RuntimeFailureError
from offlineai.runtime.base import (
    ContainerState,
    ContainerStatus,
    ImageInfo,
    RunSpec,
    RuntimeAvailability,
)
from offlineai.utils.hashing import sha256_bytes

__all__ = ["FakeRuntime"]


@dataclass
class _Container:
    spec: RunSpec
    container_id: str
    status: ContainerStatus = ContainerStatus.RUNNING
    logs: list[str] = field(default_factory=list)
    exit_code: int | None = None


class FakeRuntime:
    kind: ClassVar[str] = "fake"

    def __init__(
        self,
        *,
        available: bool = True,
        version: str = "27.0.0",
        gpu_support: bool = False,
    ) -> None:
        self._available = available
        self._version = version
        self._gpu_support = gpu_support
        self.images: dict[str, ImageInfo] = {}
        self.containers: dict[str, _Container] = {}
        self.networks: set[str] = set()
        #: Every call, for tests that care about ordering or arguments.
        self.calls: list[tuple[str, str]] = []
        #: Set to make the next matching operation fail, for rollback tests.
        self.fail_on: set[str] = set()

    # -- availability ----------------------------------------------------

    def availability(self) -> RuntimeAvailability:
        return RuntimeAvailability(
            available=self._available,
            version=self._version if self._available else None,
            detail=None if self._available else "fake runtime configured as unavailable",
            gpu_support=self._gpu_support,
        )

    # -- images ----------------------------------------------------------

    def pull(self, reference: str, *, platform: str | None = None) -> ImageInfo:
        self._record("pull", reference)
        self._maybe_fail("pull", reference)
        info = ImageInfo(
            reference=reference,
            digest="sha256:" + sha256_bytes(reference.encode()),
            image_id="sha256:" + sha256_bytes(f"id:{reference}".encode()),
            size=1024 * 1024,
        )
        self.images[reference] = info
        return info

    def build(
        self, *, dockerfile: Path, context: Path, tag: str, platform: str | None = None
    ) -> ImageInfo:
        self._record("build", tag)
        self._maybe_fail("build", tag)
        if not dockerfile.is_file():
            raise RuntimeFailureError(f"{dockerfile} does not exist")
        info = ImageInfo(
            reference=tag,
            digest="sha256:" + sha256_bytes(f"built:{tag}".encode()),
            image_id="sha256:" + sha256_bytes(f"id:{tag}".encode()),
            size=2 * 1024 * 1024,
        )
        self.images[tag] = info
        return info

    def save(self, references: list[str], destination: Path) -> None:
        self._record("save", ",".join(references))
        self._maybe_fail("save", ",".join(references))
        missing = [r for r in references if r not in self.images]
        if missing:
            raise RuntimeFailureError(f"cannot save images that are not present: {missing}")

        # A real tar, so the bundle pipeline downstream is exercised for real.
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {r: self.images[r].digest for r in references}, sort_keys=True
        ).encode()
        with tarfile.open(destination, "w") as archive:
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            info.mtime = 0
            import io

            archive.addfile(info, io.BytesIO(payload))

    def load(self, archive: Path) -> list[str]:
        self._record("load", str(archive))
        self._maybe_fail("load", str(archive))
        if not archive.is_file():
            raise RuntimeFailureError(f"{archive} does not exist")
        with tarfile.open(archive, "r") as tar:
            member = tar.extractfile("manifest.json")
            if member is None:
                raise RuntimeFailureError(f"{archive} is not a recognised image archive")
            mapping = json.loads(member.read())
        for reference, digest in mapping.items():
            self.images[reference] = ImageInfo(
                reference=reference,
                digest=digest,
                image_id="sha256:" + sha256_bytes(f"id:{reference}".encode()),
                size=1024 * 1024,
            )
        return list(mapping)

    def image_info(self, reference: str) -> ImageInfo | None:
        return self.images.get(reference)

    def remove_image(self, reference: str) -> None:
        self._record("remove_image", reference)
        self.images.pop(reference, None)

    # -- containers ------------------------------------------------------

    def run_container(self, spec: RunSpec) -> str:
        self._record("run", spec.name)
        self._maybe_fail("run", spec.name)
        if spec.name in self.containers:
            raise RuntimeFailureError(f"container {spec.name} already exists")
        if spec.image not in self.images:
            # Mirrors --pull never: an absent image is a bundle problem, not a
            # reason to reach for the network.
            raise RuntimeFailureError(
                f"image {spec.image} is not present locally",
                action="The bundle did not contain this image. Rebuild it with the image included.",
            )
        container_id = "c" + sha256_bytes(spec.name.encode())[:12]
        self.containers[spec.name] = _Container(
            spec=spec,
            container_id=container_id,
            logs=[f"{spec.name} starting", f"{spec.name} listening"],
        )
        return container_id

    def stop_container(self, name: str, *, timeout: int = 10) -> None:
        self._record("stop", name)
        container = self.containers.get(name)
        if container is not None:
            container.status = ContainerStatus.EXITED
            container.exit_code = 0
            container.logs.append(f"{name} stopped")

    def remove_container(self, name: str) -> None:
        self._record("remove", name)
        self.containers.pop(name, None)

    def container_state(self, name: str) -> ContainerState:
        container = self.containers.get(name)
        if container is None:
            return ContainerState(name=name, status=ContainerStatus.NOT_FOUND)
        return ContainerState(
            name=name,
            status=container.status,
            container_id=container.container_id,
            health="healthy" if container.status is ContainerStatus.RUNNING else None,
            exit_code=container.exit_code,
            ports={p.split(":")[-1]: p.split(":")[0] for p in container.spec.ports},
        )

    def logs(self, name: str, *, tail: int | None = None) -> str:
        container = self.containers.get(name)
        if container is None:
            raise RuntimeFailureError(f"container {name} does not exist")
        lines = container.logs[-tail:] if tail else container.logs
        return "\n".join(lines) + "\n"

    def exec_in(self, name: str, command: list[str]) -> tuple[int, str]:
        self._record("exec", name)
        container = self.containers.get(name)
        if container is None:
            return 1, f"container {name} does not exist"
        if "exec" in self.fail_on or f"exec:{name}" in self.fail_on:
            return 1, "health check failed"
        return 0, "ok"

    # -- networks --------------------------------------------------------

    def create_network(self, name: str) -> None:
        self._record("create_network", name)
        self._maybe_fail("create_network", name)
        self.networks.add(name)

    def remove_network(self, name: str) -> None:
        self._record("remove_network", name)
        self.networks.discard(name)

    # -- test helpers ----------------------------------------------------

    def _record(self, operation: str, target: str) -> None:
        self.calls.append((operation, target))

    def _maybe_fail(self, operation: str, target: str) -> None:
        if operation in self.fail_on or f"{operation}:{target}" in self.fail_on:
            raise RuntimeFailureError(f"fake runtime was told to fail on {operation} {target}")

    def crash(self, name: str, exit_code: int = 1) -> None:
        """Simulate a container exiting on its own, for health-check tests."""
        container = self.containers.get(name)
        if container is not None:
            container.status = ContainerStatus.EXITED
            container.exit_code = exit_code
            container.logs.append(f"{name} exited with code {exit_code}")
