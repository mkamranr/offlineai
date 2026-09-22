"""The bundle manifest (section 10).

The manifest is the authoritative description of a bundle, and the root of its
trust chain: it records a SHA-256 for every artifact, so a signature over the
manifest transitively covers the entire payload. That is what makes
``verify-signature`` a header-only operation even on a 62 GB bundle.

Note that on the target side the manifest is *untrusted input* until its
signature or checksums have been confirmed, so artifact paths are validated
here exactly as archive member names are.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "FORMAT_VERSION",
    "ArtifactEntry",
    "ArtifactType",
    "BuilderInfo",
    "Compression",
    "DockerRequirement",
    "GpuRequirementSummary",
    "Manifest",
    "ManifestPackage",
    "Requirements",
]

#: Bundle format version. Bumped only for incompatible changes; the target
#: refuses a version it does not understand rather than guessing.
FORMAT_VERSION = "1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ArtifactType(StrEnum):
    MODEL = "model"
    OCI_IMAGE = "oci-image"
    PYTHON_WHEEL = "python-wheel"
    PYTHON_SDIST = "python-sdist"
    SYSTEM_PACKAGE = "system-package"
    CONFIG = "config"
    SCRIPT = "script"
    MISC = "misc"


class Compression(StrEnum):
    """How the bundle archive is compressed.

    ``NONE`` is the default. Model weights and container layers are already
    compressed, so gzipping a 62 GB bundle costs hours of CPU for close to no
    saving. Small bundles can opt in.
    """

    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"


class ArtifactEntry(_Base):
    """One file inside the bundle, with the hash that authenticates it."""

    id: str
    type: ArtifactType
    path: str
    size: int = Field(ge=0)
    sha256: str
    #: Where this came from, for provenance (a repo id, an image reference, a URL).
    source: str | None = None
    #: Immutable content digest at the origin, when the source provides one.
    digest: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError(
                f"{value!r} is not a SHA-256 digest (expected 64 lowercase hex characters)"
            )
        return value

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        # The manifest arrives from outside the security boundary. A path here
        # becomes a filesystem destination, so it gets the same scrutiny as an
        # archive member name.
        if not value or value.startswith("/") or "\\" in value or "\x00" in value:
            raise ValueError(f"{value!r} is not a valid bundle-relative path")
        if any(part == ".." for part in PurePosixPath(value).parts):
            raise ValueError(f"{value!r} escapes the bundle root")
        return value


class ManifestPackage(_Base):
    name: str
    version: str


class DockerRequirement(_Base):
    minimum_version: str | None = Field(default=None, alias="minimumVersion")


class GpuRequirementSummary(_Base):
    vendor: str | None = None
    minimum_driver: str | None = Field(default=None, alias="minimumDriver")
    minimum_memory_gb: int | None = Field(default=None, alias="minimumMemoryGB", ge=0)
    count: int = 1


class Requirements(_Base):
    docker: DockerRequirement | None = None
    gpu: GpuRequirementSummary | None = None
    minimum_ram_gb: int | None = Field(default=None, alias="minimumRamGB", ge=0)
    minimum_disk_gb: int | None = Field(default=None, alias="minimumDiskGB", ge=0)
    minimum_cpu_cores: int | None = Field(default=None, alias="minimumCpuCores", ge=1)
    python_version: str | None = Field(default=None, alias="pythonVersion")
    python_platform: str | None = Field(default=None, alias="pythonPlatform")
    python_architecture: str | None = Field(default=None, alias="pythonArchitecture")


class BuilderInfo(_Base):
    """Provenance of the build, per section 34."""

    offlineai_version: str = Field(alias="offlineaiVersion")
    os: str
    architecture: str
    python_version: str = Field(alias="pythonVersion")


class Manifest(_Base):
    format_version: Literal["1"] = Field(alias="formatVersion")
    package: ManifestPackage
    created_at: datetime = Field(alias="createdAt")
    platforms: list[str] = Field(default_factory=list)
    compression: Compression = Compression.NONE

    artifacts: list[ArtifactEntry] = Field(default_factory=list)
    requirements: Requirements = Field(default_factory=Requirements)

    #: Hash of the offlineai.yaml this bundle was built from (section 34).
    package_definition_sha256: str | None = Field(default=None, alias="packageDefinitionSha256")
    builder: BuilderInfo | None = None
    #: Environment variable names the workload expects to be supplied
    #: externally. Names only - never values (section 40).
    required_secrets: list[str] = Field(default_factory=list, alias="requiredSecrets")

    @model_validator(mode="after")
    def _check_unique(self) -> Manifest:
        for field, values in (
            ("id", [a.id for a in self.artifacts]),
            ("path", [a.path for a in self.artifacts]),
        ):
            seen: set[str] = set()
            for value in values:
                if value in seen:
                    raise ValueError(f"duplicate artifact {field}: {value!r}")
                seen.add(value)
        return self

    # -- lookups ----------------------------------------------------------

    @property
    def total_size(self) -> int:
        return sum(artifact.size for artifact in self.artifacts)

    def size_by_type(self) -> dict[ArtifactType, int]:
        totals: dict[ArtifactType, int] = defaultdict(int)
        for artifact in self.artifacts:
            totals[artifact.type] += artifact.size
        return dict(totals)

    def artifact_by_path(self, path: str) -> ArtifactEntry | None:
        return next((a for a in self.artifacts if a.path == path), None)

    def artifacts_of_type(self, artifact_type: ArtifactType) -> list[ArtifactEntry]:
        return [a for a in self.artifacts if a.type == artifact_type]

    @property
    def identifier(self) -> str:
        return f"{self.package.name}:{self.package.version}"

    # -- serialisation ----------------------------------------------------

    def canonical_bytes(self) -> bytes:
        """Deterministic serialisation, used as the signing input.

        Signing canonical JSON rather than the stored YAML bytes means
        reformatting the manifest cannot invalidate a signature, and reordering
        keys cannot be used to produce two documents with one signature.
        """
        payload = self.model_dump(by_alias=True, mode="json", exclude_none=True)
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def to_yaml(self) -> str:
        payload = self.model_dump(by_alias=True, mode="json", exclude_none=True)
        return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)

    @classmethod
    def from_yaml(cls, text: str | bytes) -> Manifest:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("manifest must be a YAML mapping")
        return cls.model_validate(data)
