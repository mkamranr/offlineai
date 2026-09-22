"""Typed results returned by bundler operations.

Commands return these rather than printing. One renderer turns them into human
text or JSON (section 47), which keeps ``--json`` honest and makes the command
logic testable without going through the CLI.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from offlineai.schema.manifest import ArtifactType, Compression

__all__ = [
    "BuildResult",
    "BuildStep",
    "CheckResult",
    "CheckStatus",
    "InspectResult",
    "VerifyResult",
]


class CheckStatus(StrEnum):
    OK = "OK"
    FAILED = "FAILED"
    WARNING = "WARNING"
    #: Never reported as OK. Section 21 is explicit that a requirement which
    #: was not actually evaluated must not be presented as satisfied.
    SKIPPED = "SKIPPED"


class CheckResult(BaseModel):
    name: str
    status: CheckStatus
    detail: str | None = None

    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.OK, CheckStatus.SKIPPED, CheckStatus.WARNING)


class VerifyResult(BaseModel):
    package: str
    version: str
    checks: list[CheckResult] = Field(default_factory=list)
    artifacts_verified: int = 0
    bytes_verified: int = 0
    verified: bool = False

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAILED]


class InspectResult(BaseModel):
    """Everything ``inspect`` reports, all of it from the bundle header."""

    package: str
    version: str
    format_version: str
    platforms: list[str] = Field(default_factory=list)
    compression: Compression = Compression.NONE
    created_at: str | None = None
    description: str | None = None

    artifact_counts: dict[ArtifactType, int] = Field(default_factory=dict)
    artifact_sizes: dict[ArtifactType, int] = Field(default_factory=dict)
    total_size: int = 0
    file_size: int = 0

    gpu_vendor: str | None = None
    gpu_minimum_vram_gb: int | None = None
    docker_minimum_version: str | None = None
    python_version: str | None = None

    signature_present: bool = False
    sbom_present: bool = False
    required_secrets: list[str] = Field(default_factory=list)
    builder: dict[str, str] = Field(default_factory=dict)
    #: Bytes actually read to produce this report, for the curious.
    header_bytes_read: int = 0


class BuildStep(BaseModel):
    index: int
    total: int
    name: str
    status: CheckStatus = CheckStatus.OK
    detail: str | None = None


class BuildResult(BaseModel):
    package: str
    version: str
    bundle_path: str
    size: int
    sha256: str
    artifact_count: int
    steps: list[BuildStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
