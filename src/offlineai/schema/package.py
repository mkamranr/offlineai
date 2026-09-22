"""The ``offlineai.yaml`` package definition (section 8).

Two decisions shape this module.

**The schema is versioned.** ``apiVersion`` is checked exactly. Section 77.16
requires the bundle format to be versioned, and the package format is the half
of it that humans write, so it must be able to evolve without silently
changing meaning under an existing file.

**Unknown keys are rejected, everywhere.** This looks pedantic and is not. A
typo such as ``containters:`` that validates silently produces a bundle with no
container image in it, and the operator discovers that on the air-gapped side,
where they cannot simply rebuild. Failing at definition time on the builder is
the only cheap moment to catch it.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

__all__ = [
    "API_VERSION",
    "PORT_RE",
    "ContainerSpec",
    "GpuRequirement",
    "HardwareSpec",
    "ImageReference",
    "InstallSpec",
    "ModelSpec",
    "Package",
    "PackageMetadata",
    "PythonSpec",
    "SecretsSpec",
    "ServiceSpec",
    "SystemSpec",
    "VolumeSpec",
]

#: The only package API version this release understands.
API_VERSION = "offlineai/v1"

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,126}[a-z0-9])?$")
_VERSION_RE = re.compile(r"^[0-9]+(\.[0-9]+)*([-+][0-9A-Za-z.-]+)?$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: host:container, optionally ip-qualified and optionally /proto.
#: Exported so the override schema validates ports identically.
PORT_RE = re.compile(
    r"^(?:(?P<ip>\d{1,3}(?:\.\d{1,3}){3}):)?"
    r"(?P<host>\d{1,5}):(?P<container>\d{1,5})"
    r"(?:/(?P<proto>tcp|udp))?$"
)


Architecture: TypeAlias = Literal["amd64", "arm64"]


def _default_architectures() -> list[Architecture]:
    return ["amd64"]


class _Strict(BaseModel):
    """Base for every node in the document: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------


class ImageReference(_Strict):
    """A parsed OCI image reference.

    Section 16 is emphatic that ``image:tag`` alone is not enough, because a
    tag is mutable. The digest is recorded separately and preferred wherever
    one is available.
    """

    repository: str
    tag: str = "latest"
    digest: str | None = None

    @classmethod
    def parse(cls, raw: str) -> ImageReference:
        text = raw.strip()
        if not text:
            raise ValueError("image reference is empty")

        digest: str | None = None
        if "@" in text:
            text, _, digest = text.partition("@")
            if not _DIGEST_RE.match(digest):
                raise ValueError(f"malformed image digest: {digest!r}")

        tag = "latest"
        # A colon only introduces a tag when it is in the last path segment.
        # Otherwise it is a registry port, as in registry.internal:5000/app.
        head, sep, candidate = text.rpartition(":")
        if sep and "/" not in candidate:
            text, tag = head, candidate

        if not text:
            raise ValueError(f"malformed image reference: {raw!r}")
        return cls(repository=text, tag=tag, digest=digest)

    def __str__(self) -> str:
        if self.digest:
            return f"{self.repository}@{self.digest}"
        return f"{self.repository}:{self.tag}"

    @property
    def pinned(self) -> bool:
        """Whether this reference names immutable content."""
        return self.digest is not None


# ---------------------------------------------------------------------------
# Leaf specs
# ---------------------------------------------------------------------------


class PackageMetadata(_Strict):
    name: str
    version: str
    description: str | None = None
    license: str | None = None
    homepage: str | None = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        # The name becomes part of a filename and a registry key, so it has to
        # stay free of path separators and shell-significant characters.
        if not _NAME_RE.match(value):
            raise ValueError(
                f"{value!r} is not a valid package name. Use lowercase letters, "
                "digits and hyphens, starting and ending with an alphanumeric "
                "character (for example 'qwen3-30b')."
            )
        return value

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        if not _VERSION_RE.match(value):
            raise ValueError(
                f"{value!r} is not a valid version. Use a dotted numeric version "
                "with an optional pre-release or build suffix (for example '1.0.0')."
            )
        return value


class HuggingFaceModelSource(_Strict):
    type: Literal["huggingface"]
    repo: str
    revision: str | None = None
    #: Restrict which files are fetched. Empty means the whole repository.
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class HttpModelSource(_Strict):
    type: Literal["http"]
    url: str
    sha256: str | None = None


class LocalModelSource(_Strict):
    type: Literal["local"]
    path: str


ModelSource = Annotated[
    HuggingFaceModelSource | HttpModelSource | LocalModelSource,
    Field(discriminator="type"),
]


class ModelSpec(_Strict):
    name: str
    source: ModelSource
    #: Where the model is mounted inside the container.
    destination: str | None = None


class ContainerSpec(_Strict):
    name: str
    image: str
    platform: str = "linux/amd64"
    digest: str | None = None
    #: Build the image locally instead of pulling it (section 55).
    dockerfile: str | None = None
    context: str | None = None

    @field_validator("digest")
    @classmethod
    def _check_digest(cls, value: str | None) -> str | None:
        if value is not None and not _DIGEST_RE.match(value):
            raise ValueError(
                f"{value!r} is not a valid digest. Expected 'sha256:' followed by "
                "64 lowercase hex characters."
            )
        return value

    @property
    def reference(self) -> ImageReference:
        ref = ImageReference.parse(self.image)
        if self.digest and not ref.digest:
            return ImageReference(repository=ref.repository, tag=ref.tag, digest=self.digest)
        return ref


class ServiceSpec(_Strict):
    name: str
    container: str
    command: list[str] = Field(default_factory=list)
    ports: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("ports")
    @classmethod
    def _check_ports(cls, values: list[str]) -> list[str]:
        for value in values:
            match = PORT_RE.match(value)
            if not match:
                raise ValueError(
                    f"{value!r} is not a valid port mapping. Use 'HOST:CONTAINER', "
                    "optionally as 'IP:HOST:CONTAINER' or with a '/tcp' or '/udp' suffix."
                )
            for key in ("host", "container"):
                port = int(match.group(key))
                if not 1 <= port <= 65535:
                    raise ValueError(f"port {port} in {value!r} is outside 1-65535")
        return values

    def host_ports(self) -> list[int]:
        """Host-side ports, for reporting endpoints after start."""
        return [int(match.group("host")) for value in self.ports if (match := PORT_RE.match(value))]


class VolumeSpec(_Strict):
    host: str
    container: str
    read_only: bool = False


class PythonSpec(_Strict):
    requirements: list[str] = Field(default_factory=list)
    #: Interpreter the wheels must match on the target (section 18).
    version: str | None = None
    platform: str = "linux"
    architecture: str = "amd64"


class SystemSpec(_Strict):
    packages: list[str] = Field(default_factory=list)
    #: Section 19 requires the installer to distinguish these tiers.
    optional_packages: list[str] = Field(default_factory=list)
    recommended_packages: list[str] = Field(default_factory=list)


class GpuRequirement(_Strict):
    required: bool = False
    vendor: Literal["nvidia", "amd", "intel"] = "nvidia"
    minimum_vram_gb: int | None = Field(default=None, ge=0)
    minimum_driver: str | None = None
    count: int = Field(default=1, ge=1)


class HardwareSpec(_Strict):
    gpu: GpuRequirement | None = None
    minimum_ram_gb: int | None = Field(default=None, ge=0)
    minimum_disk_gb: int | None = Field(default=None, ge=0)
    minimum_cpu_cores: int | None = Field(default=None, ge=1)


class HealthcheckSpec(_Strict):
    command: list[str] = Field(default_factory=list)
    interval_seconds: int = Field(default=5, ge=1)
    timeout_seconds: int = Field(default=60, ge=1)
    retries: int = Field(default=12, ge=1)


class InstallSpec(_Strict):
    healthcheck: HealthcheckSpec | None = None


class SecretsSpec(_Strict):
    """Section 40: the bundle carries secret *names*, never values.

    ``external`` is a list of plain strings by design. Accepting a mapping here
    would invite somebody to write ``TOKEN: hunter2`` and ship it.
    """

    external: list[str] = Field(default_factory=list)

    @field_validator("external")
    @classmethod
    def _check_names(cls, values: list[Any]) -> list[str]:
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    "secrets.external must be a list of environment variable NAMES. "
                    "A bundle must never contain secret values."
                )
            if not _ENV_NAME_RE.match(value):
                raise ValueError(f"{value!r} is not a valid environment variable name")
        return values


class RuntimeSpec(_Strict):
    type: Literal["docker"] = "docker"


# ---------------------------------------------------------------------------
# Document root
# ---------------------------------------------------------------------------


class Package(_Strict):
    """A complete ``offlineai.yaml`` document."""

    api_version: str = Field(alias="apiVersion")
    kind: Literal["Package"]
    metadata: PackageMetadata

    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    architecture: list[Architecture] = Field(default_factory=_default_architectures)
    os: list[str] = Field(default_factory=list)

    models: list[ModelSpec] = Field(default_factory=list)
    containers: list[ContainerSpec] = Field(default_factory=list)
    python: PythonSpec | None = None
    system: SystemSpec | None = None

    environment: dict[str, str] = Field(default_factory=dict)
    volumes: list[VolumeSpec] = Field(default_factory=list)
    services: list[ServiceSpec] = Field(default_factory=list)

    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    install: InstallSpec = Field(default_factory=InstallSpec)
    secrets: SecretsSpec = Field(default_factory=SecretsSpec)

    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)

    @field_validator("api_version")
    @classmethod
    def _check_api_version(cls, value: str) -> str:
        if value != API_VERSION:
            raise ValueError(
                f"unsupported apiVersion {value!r}. This release understands {API_VERSION!r}."
            )
        return value

    @field_validator("environment", mode="before")
    @classmethod
    def _stringify_environment(cls, value: Any) -> Any:
        # YAML turns `WORKERS: 4` into an int, but container environments are
        # strings. Coerce rather than making users quote every number.
        if isinstance(value, dict):
            return {k: str(v) if v is not None else "" for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _check_cross_references(self) -> Package:
        _reject_duplicates([c.name for c in self.containers], "container")
        _reject_duplicates([m.name for m in self.models], "model")
        _reject_duplicates([s.name for s in self.services], "service")

        declared = {c.name for c in self.containers}
        for service in self.services:
            if service.container not in declared:
                known = ", ".join(sorted(declared)) or "none declared"
                raise ValueError(
                    f"service {service.name!r} references container "
                    f"{service.container!r}, which is not declared. "
                    f"Declared containers: {known}."
                )

        service_names = {s.name for s in self.services}
        for service in self.services:
            for dependency in service.depends_on:
                if dependency not in service_names:
                    raise ValueError(
                        f"service {service.name!r} depends on {dependency!r}, "
                        "which is not a declared service."
                    )
        return self

    def bundle_filename(self) -> str:
        """The conventional output name, ``<name>-<version>.offlineai``."""
        return f"{self.metadata.name}-{self.metadata.version}.offlineai"

    @property
    def identifier(self) -> str:
        return f"{self.metadata.name}:{self.metadata.version}"


def _reject_duplicates(names: list[str], kind: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate {kind} name {name!r}")
        seen.add(name)
