"""Plain HTTP(S) artifact source.

For artifacts that live at a URL rather than in a model hub - an internal
mirror, a vendor download, a dataset tarball. Shares the resumable download
machinery with the Hugging Face source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from urllib.parse import unquote, urlparse

from offlineai import layout
from offlineai.artifacts.base import ArtifactRequest, ResolvedArtifact, SourceRef
from offlineai.artifacts.cache import ArtifactCache
from offlineai.artifacts.download import download_resumable
from offlineai.errors import SourceError
from offlineai.progress import ProgressReporter, download_callback

if TYPE_CHECKING:
    import httpx

__all__ = ["HttpSource"]


class HttpSource:
    kind: ClassVar[str] = "http"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client

    def expand(self, ref: SourceRef) -> list[ArtifactRequest]:
        parsed = urlparse(ref.locator)
        if parsed.scheme not in ("http", "https"):
            raise SourceError(
                f"{ref.locator!r} is not an http or https URL",
                action="Use a full URL, for example https://example.internal/model.tar",
            )
        filename = unquote(parsed.path.rsplit("/", 1)[-1]) or "download"
        if ref.artifact_type.value == "model":
            bundle_path = layout.artifact_path(ref.artifact_type, ref.name, filename)
        else:
            bundle_path = layout.artifact_path(ref.artifact_type, filename)

        return [
            ArtifactRequest(
                id=f"{ref.artifact_type.value}-{ref.name}",
                artifact_type=ref.artifact_type,
                bundle_path=bundle_path,
                locator=ref.locator,
                source_kind=self.kind,
                expected_sha256=_optional_str(ref.options.get("sha256")),
                metadata={"url": ref.locator, "name": ref.name},
            )
        ]

    def fetch(
        self,
        request: ArtifactRequest,
        cache: ArtifactCache,
        *,
        progress: ProgressReporter | None = None,
    ) -> ResolvedArtifact:
        cache.ensure()
        with cache.partial(f"{self.kind}:{request.locator}") as staging:
            result = download_resumable(
                request.locator,
                staging,
                client=self._http(),
                expected_sha256=request.expected_sha256,
                expected_size=request.expected_size,
                progress=(download_callback(progress, request.id) if progress else None),
            )
            entry = cache.store_file(result.path, move=True)

        cache.put_ref(self.kind, request.cache_key, entry.sha256)
        return ResolvedArtifact(
            request=request,
            local_path=entry.path,
            sha256=entry.sha256,
            size=entry.size,
            source=request.locator,
        )

    def _http(self) -> httpx.Client:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise SourceError(
                    "downloading artifacts over HTTP requires the builder extra",
                    action="Install it on the builder machine:\n  pip install 'offlineai[builder]'",
                ) from exc
            self._client = httpx.Client(timeout=httpx.Timeout(30.0, read=300.0))
        return self._client


def _optional_str(value: object) -> str | None:
    return str(value) if value else None
