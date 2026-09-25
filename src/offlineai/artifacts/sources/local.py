"""Local filesystem artifact source.

The simplest source, and the one the test suite leans on: it needs no network,
so build/verify/import/install can be exercised end to end under the offline
guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from offlineai.artifacts.base import ArtifactRequest, ResolvedArtifact, SourceRef
from offlineai.artifacts.cache import ArtifactCache
from offlineai.errors import SourceError
from offlineai.progress import ProgressReporter
from offlineai.schema.manifest import ArtifactType

__all__ = ["LocalSource"]


class LocalSource:
    kind: ClassVar[str] = "local"

    def __init__(self, base_dir: Path | str | None = None) -> None:
        #: Relative locators resolve against this, normally the directory
        #: containing offlineai.yaml.
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def expand(self, ref: SourceRef) -> list[ArtifactRequest]:
        source = self._resolve(ref.locator)
        if source.is_file():
            return [self._request(ref, source, source.name)]
        if source.is_dir():
            files = sorted(p for p in source.rglob("*") if p.is_file())
            if not files:
                raise SourceError(
                    f"{source} contains no files",
                    action="An empty directory produces an empty bundle entry. "
                    "Remove the declaration or point it at real content.",
                )
            return [
                self._request(ref, f, str(f.relative_to(source)).replace("\\", "/")) for f in files
            ]
        raise SourceError(
            f"{source} does not exist",
            details={"Declared as": ref.locator},
            action="Check the path in your package definition.",
        )

    def fetch(
        self,
        request: ArtifactRequest,
        cache: ArtifactCache,
        *,
        progress: ProgressReporter | None = None,  # noqa: ARG002 - a local copy completes faster than a bar would render
    ) -> ResolvedArtifact:
        source = Path(request.locator)
        if not source.is_file():
            raise SourceError(f"{source} disappeared during the build")
        entry = cache.store_file(source)
        # Deliberately no put_ref. A local file's cache key is its path, and a
        # path does not change when its content does - so a developer editing
        # a model and rebuilding would silently get the previous bytes back.
        # store_file already deduplicates by content, so the reference index
        # adds nothing here but the chance to be wrong.
        return ResolvedArtifact(
            request=request,
            local_path=entry.path,
            sha256=entry.sha256,
            size=entry.size,
            source=f"file://{source}",
        )

    # -- helpers ---------------------------------------------------------

    def _resolve(self, locator: str) -> Path:
        path = Path(locator).expanduser()
        return path if path.is_absolute() else (self.base_dir / path)

    def _request(self, ref: SourceRef, file: Path, relative: str) -> ArtifactRequest:
        from offlineai import layout

        if ref.artifact_type is ArtifactType.MODEL:
            bundle_path = layout.artifact_path(ref.artifact_type, ref.name, relative)
        else:
            bundle_path = layout.artifact_path(ref.artifact_type, relative)
        return ArtifactRequest(
            id=f"{ref.artifact_type.value}-{ref.name}-{relative}".replace("/", "-"),
            artifact_type=ref.artifact_type,
            bundle_path=bundle_path,
            locator=str(file),
            source_kind=self.kind,
            metadata={"declared_as": ref.locator},
        )
