"""Container runtime abstraction (section 77.14).

Docker is the first implementation, but nothing above this interface knows
that. Podman and containerd are meant to slot in behind it later, and the test
suite uses an in-memory implementation so the installer can be exercised end
to end without a daemon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

__all__ = [
    "ContainerRuntime",
    "ContainerState",
    "ContainerStatus",
    "ImageInfo",
    "RunSpec",
    "RuntimeAvailability",
]


class ContainerStatus(StrEnum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    EXITED = "EXITED"
    RESTARTING = "RESTARTING"
    CREATED = "CREATED"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RuntimeAvailability:
    available: bool
    version: str | None = None
    detail: str | None = None
    #: Whether the runtime can expose GPUs (the NVIDIA container toolkit).
    gpu_support: bool = False


@dataclass(frozen=True, slots=True)
class ImageInfo:
    reference: str
    #: Immutable content digest. Section 16: never rely on a tag alone.
    digest: str | None = None
    image_id: str | None = None
    size: int = 0


@dataclass(frozen=True, slots=True)
class ContainerState:
    name: str
    status: ContainerStatus
    container_id: str | None = None
    health: str | None = None
    exit_code: int | None = None
    ports: dict[str, str] = field(default_factory=dict)

    @property
    def running(self) -> bool:
        return self.status is ContainerStatus.RUNNING


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything needed to start one container."""

    name: str
    image: str
    command: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    ports: list[str] = field(default_factory=list)
    volumes: list[tuple[str, str, bool]] = field(default_factory=list)
    network: str | None = None
    #: GPU device ids to expose, or None for no GPU access (section 30).
    gpu_device_ids: list[str] | None = None
    labels: dict[str, str] = field(default_factory=dict)
    restart_policy: str = "unless-stopped"


@runtime_checkable
class ContainerRuntime(Protocol):
    kind: ClassVar[str]

    def availability(self) -> RuntimeAvailability: ...

    # -- images ----------------------------------------------------------

    def pull(self, reference: str, *, platform: str | None = None) -> ImageInfo: ...

    def build(
        self, *, dockerfile: Path, context: Path, tag: str, platform: str | None = None
    ) -> ImageInfo: ...

    def save(self, references: list[str], destination: Path) -> None: ...

    def load(self, archive: Path) -> list[str]: ...

    def image_info(self, reference: str) -> ImageInfo | None: ...

    def remove_image(self, reference: str) -> None: ...

    # -- containers ------------------------------------------------------

    def run_container(self, spec: RunSpec) -> str: ...

    def stop_container(self, name: str, *, timeout: int = 10) -> None: ...

    def remove_container(self, name: str) -> None: ...

    def container_state(self, name: str) -> ContainerState: ...

    def logs(self, name: str, *, tail: int | None = None) -> str: ...

    def exec_in(self, name: str, command: list[str]) -> tuple[int, str]: ...

    # -- networks --------------------------------------------------------

    def create_network(self, name: str) -> None: ...

    def remove_network(self, name: str) -> None: ...
