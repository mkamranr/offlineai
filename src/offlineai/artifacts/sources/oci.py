"""Container image artifact source (sections 15 and 16).

Pulls an image on the builder and saves it into the cache as a tar that
``docker load`` can consume on the target.

Section 16 is the important constraint: a tag is mutable, so the immutable
content digest is resolved at build time and recorded in the manifest. The
target then knows exactly which image it is supposed to have, and a mismatch
is reported rather than silently tolerated.
"""

from __future__ import annotations

from typing import ClassVar

from offlineai import layout
from offlineai.artifacts.base import ArtifactRequest, ResolvedArtifact, SourceRef
from offlineai.artifacts.cache import ArtifactCache
from offlineai.errors import SourceError
from offlineai.logging import get_logger
from offlineai.runtime.base import ContainerRuntime
from offlineai.schema.manifest import ArtifactType
from offlineai.schema.package import ImageReference

__all__ = ["OciSource"]

logger = get_logger("artifacts.oci")


class OciSource:
    kind: ClassVar[str] = "oci"

    def __init__(self, runtime: ContainerRuntime) -> None:
        self.runtime = runtime

    def expand(self, ref: SourceRef) -> list[ArtifactRequest]:
        """One image becomes one archive.

        Saving each image separately rather than batching them into a single
        tar costs a little duplication of shared layers, but means the cache
        can dedupe an image reused across bundles, and a single changed image
        does not invalidate the rest.
        """
        reference = ImageReference.parse(ref.locator)
        filename = _safe_filename(ref.name)
        return [
            ArtifactRequest(
                id=f"image-{ref.name}",
                artifact_type=ArtifactType.OCI_IMAGE,
                bundle_path=layout.artifact_path(ArtifactType.OCI_IMAGE, f"{filename}.tar"),
                locator=str(reference),
                source_kind=self.kind,
                metadata={
                    "image": reference.repository,
                    "tag": reference.tag,
                    "platform": str(ref.options.get("platform", "linux/amd64")),
                    "container": ref.name,
                },
            )
        ]

    def fetch(self, request: ArtifactRequest, cache: ArtifactCache) -> ResolvedArtifact:
        reference = request.locator
        platform = str(request.metadata.get("platform") or "linux/amd64")

        availability = self.runtime.availability()
        if not availability.available:
            raise SourceError(
                "a container runtime is required to package container images",
                details={"Detail": availability.detail or "unavailable"},
                action="Install Docker on the builder machine, or remove the "
                "container declarations from the package definition.",
            )

        # Prefer what is already local. A builder that has just built the image
        # should not have to push it somewhere first.
        info = self.runtime.image_info(reference)
        if info is None:
            logger.info("pulling %s", reference)
            info = self.runtime.pull(reference, platform=platform)

        # Cache on the immutable digest where we have one, so a rebuild with an
        # unchanged image is a cache hit rather than another multi-GB save.
        cache_key = info.digest or reference
        cached = cache.lookup_ref(self.kind, cache_key)
        if cached is not None:
            return ResolvedArtifact(
                request=request,
                local_path=cached.path,
                sha256=cached.sha256,
                size=cached.size,
                source=reference,
                digest=info.digest,
                cached=True,
            )

        cache.ensure()
        with cache.partial(f"{self.kind}:{cache_key}") as staging:
            self.runtime.save([reference], staging)
            entry = cache.store_file(staging, move=True)

        cache.put_ref(self.kind, cache_key, entry.sha256)
        return ResolvedArtifact(
            request=request,
            local_path=entry.path,
            sha256=entry.sha256,
            size=entry.size,
            source=reference,
            digest=info.digest,
        )


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)
