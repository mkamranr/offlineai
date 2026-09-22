"""Artifact source abstraction (sections 14 and 49).

The target environment must never depend on one provider, so everything that
knows how to obtain bytes sits behind this interface and is discovered through
entry points. Adding ModelScope, S3, NGC or an APT mirror later means writing a
source, not editing the CLI.

The split between :meth:`ArtifactSource.expand` and :meth:`ArtifactSource.fetch`
matters: one declaration in ``offlineai.yaml`` frequently becomes many files.
Section 13 is explicit that a model is not a single file - a sharded checkpoint
is eight safetensors plus a tokenizer, a config and generation settings - so
expansion is a first-class step rather than something each source improvises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from offlineai.schema.manifest import ArtifactEntry, ArtifactType

if TYPE_CHECKING:
    from offlineai.artifacts.cache import ArtifactCache

__all__ = ["ArtifactRequest", "ArtifactSource", "ResolvedArtifact", "SourceRef"]


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A declaration from the package definition, before expansion.

    ``kind`` selects the source ("huggingface", "oci", "local", ...);
    ``locator`` is whatever that source understands (a repo id, an image
    reference, a path).
    """

    kind: str
    locator: str
    name: str
    artifact_type: ArtifactType
    options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactRequest:
    """One concrete file to obtain and place in the bundle."""

    id: str
    artifact_type: ArtifactType
    bundle_path: str
    #: Source-specific locator for this individual file.
    locator: str
    source_kind: str
    #: Expected digest when the source can state one up front. Lets the cache
    #: answer a hit without contacting the origin at all.
    expected_sha256: str | None = None
    expected_size: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        return f"{self.source_kind}/{self.locator}"


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """A request that now has bytes on local disk, with its digest known."""

    request: ArtifactRequest
    local_path: Path
    sha256: str
    size: int
    #: Provenance recorded in the manifest and the SBOM.
    source: str | None = None
    #: Immutable digest at the origin (an OCI image digest, for example).
    digest: str | None = None
    license: str | None = None
    #: Whether this came from the cache rather than the network.
    cached: bool = False

    def to_manifest_entry(self) -> ArtifactEntry:
        return ArtifactEntry(
            id=self.request.id,
            type=self.request.artifact_type,
            path=self.request.bundle_path,
            size=self.size,
            sha256=self.sha256,
            source=self.source,
            digest=self.digest,
            license=self.license,
            metadata=dict(self.request.metadata),
        )


@runtime_checkable
class ArtifactSource(Protocol):
    """Obtains artifacts of one kind.

    Implementations are build-time only. Nothing on the air-gapped target ever
    calls a source - a target that needs one is a bundle that was built wrong,
    and that surfaces as :class:`~offlineai.errors.MissingArtifactError`.
    """

    #: Stable identifier, matching the ``type:`` in a package definition and
    #: the entry-point name.
    kind: ClassVar[str]

    def expand(self, ref: SourceRef) -> list[ArtifactRequest]:
        """Turn one declaration into the concrete files it implies."""
        ...

    def fetch(self, request: ArtifactRequest, cache: ArtifactCache) -> ResolvedArtifact:
        """Obtain one file, using and populating the cache."""
        ...
